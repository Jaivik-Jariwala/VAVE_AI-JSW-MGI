"""
VAVE U-Net CycleGAN Training Script
=====================================
Architecture: U-Net Generator + CycleGAN cycle-consistency loss + PatchGAN Discriminator
Text Conditioning: CLIP embeddings injected at the U-Net bottleneck

Why this is better than the original GAN:
  - Old GAN: Noise -> Image (impossible to learn, no structure)
  - This:    Real Car Image + Text -> Modified Car Image (preserves vehicle structure)

Cycle Consistency Loss ensures the model learns reversible transformations:
  G_AB: Original Vehicle -> Modified Version
  G_BA: Modified Version -> Reconstructed Original
  Loss = ||G_BA(G_AB(original)) - original|| + adversarial

GPU: STRONGLY RECOMMENDED (12-24h on CPU, 1-2h on GPU)
     Runs on CPU too - just much slower per epoch.

Usage:
    python advanced_gan/train_unet_cyclegan.py
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
class Config:
    IMAGE_SIZE   = 256
    BATCH_SIZE   = 2  # Keep low for CPU; increase to 8 on GPU
    EPOCHS       = 50
    LR           = 0.0002
    B1, B2       = 0.5, 0.999
    LAMBDA_CYCLE = 10.0   # Cycle-consistency weight
    LAMBDA_ID    = 5.0    # Identity loss (keeps unchanged areas stable)
    LAMBDA_PIXEL = 20.0   # Pixel-wise L1 (structural fidelity)
    TEXT_DIM     = 512    # CLIP output dimension

    # Paths — script lives one level below project root
    BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR     = os.path.join(BASE_DIR, "images", "mg")
    EXCEL_PATH   = os.path.join(BASE_DIR, "AIML Dummy Ideas Data.xlsx")
    OUTPUT_DIR   = os.path.join(BASE_DIR, "advanced_gan", "unet_cyclegan_samples")
    WEIGHTS_DIR  = os.path.join(BASE_DIR, "advanced_gan", "unet_cyclegan_weights")
    # Final weight saved here so vlm_engine can load it
    FINAL_G_PATH = os.path.join(BASE_DIR, "saved_models", "unet_generator_final.pth")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.FINAL_G_PATH), exist_ok=True)

if Config.DEVICE.type == "cpu":
    logger.warning("GPU not detected — training on CPU. This will be slow.")
    logger.warning("Each epoch may take 30-60 minutes on CPU.")
    logger.warning("For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
else:
    logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class VehicleIdeaDataset(Dataset):
    def __init__(self, excel_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        try:
            df = pd.read_excel(excel_path)
            for idx, row in df.iterrows():
                text = str(row.get("Cost Reduction Idea Proposal") or "").strip()
                if len(text) < 5 or text.lower() == "nan":
                    continue
                img_path = os.path.join(img_dir, f"{idx + 1}.jpg")
                if os.path.exists(img_path):
                    self.data.append({"img": img_path, "text": text})
            logger.info(f"Loaded {len(self.data)} image-text training pairs.")
        except Exception as e:
            logger.error(f"Dataset load failed: {e}")

        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["img"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tok = self.tokenizer(item["text"], padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt")
        return img, tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0), item["text"]


# ─── U-Net Blocks ────────────────────────────────────────────────────────────
class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        return torch.cat((self.model(x), skip), 1)


# ─── U-Net Generator (Text-Conditioned Image-to-Image) ───────────────────────
class GeneratorUNet(nn.Module):
    """
    Input : vehicle image (3, H, W) + CLIP text embedding (TEXT_DIM,)
    Output: proposed modified vehicle image (3, H, W)

    Text is injected at the bottleneck — the deepest U-Net layer —
    so it globally guides the style/modification without disturbing local edges.
    """
    def __init__(self, text_dim=512):
        super().__init__()
        self.text_fc = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU())

        self.d1 = UNetDown(3,    64,  normalize=False)
        self.d2 = UNetDown(64,   128)
        self.d3 = UNetDown(128,  256)
        self.d4 = UNetDown(256,  512, dropout=0.5)
        self.d5 = UNetDown(512,  512, dropout=0.5)
        self.d6 = UNetDown(512,  512, dropout=0.5)  # Bottleneck: [B, 512, 4, 4]

        # up1 gets bottleneck(512) + text(256) = 768, skip from d5(512)
        self.u1 = UNetUp(512 + 256, 512, dropout=0.5)  # out: 512, cat d5 -> 1024
        self.u2 = UNetUp(1024, 512, dropout=0.5)        # out: 512, cat d4 -> 1024
        self.u3 = UNetUp(1024, 256)                     # out: 256, cat d3 -> 512
        self.u4 = UNetUp(512,  128)                     # out: 128, cat d2 -> 256
        self.u5 = UNetUp(256,   64)                     # out:  64, cat d1 -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, img, text_emb):
        d1 = self.d1(img);  d2 = self.d2(d1);  d3 = self.d3(d2)
        d4 = self.d4(d3);   d5 = self.d5(d4);  d6 = self.d6(d5)

        tf = self.text_fc(text_emb)                         # [B, 256]
        tf = tf.view(tf.size(0), 256, 1, 1).expand(-1, -1, d6.size(2), d6.size(3))
        bottleneck = torch.cat((d6, tf), 1)                 # [B, 768, 4, 4]

        u1 = self.u1(bottleneck, d5);  u2 = self.u2(u1, d4)
        u3 = self.u3(u2, d3);          u4 = self.u4(u3, d2)
        u5 = self.u5(u4, d1)
        return self.final(u5)


# ─── PatchGAN Discriminator ───────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    PatchGAN — classifies whether 70x70 patches of the image are real or fake.
    Much more stable than a fully-connected discriminator.
    """
    def __init__(self, text_dim=512):
        super().__init__()
        self.text_fc = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU())

        def block(in_f, out_f, norm=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1))]
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3,    64,  norm=False),
            *block(64,   128),
            *block(128,  256),
            *block(256,  512),
        )
        self.final = nn.utils.spectral_norm(nn.Conv2d(512 + 256, 1, 3, padding=1))

    def forward(self, img, text_emb):
        feat = self.model(img)
        tf = self.text_fc(text_emb)
        tf = tf.view(tf.size(0), 256, 1, 1).expand(-1, -1, feat.size(2), feat.size(3))
        return self.final(torch.cat((feat, tf), 1))


# ─── Training Loop ────────────────────────────────────────────────────────────
def train():
    logger.info(f"Starting U-Net CycleGAN — Device: {Config.DEVICE}")

    from transformers import CLIPTextModel
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(Config.DEVICE)
    clip.eval()

    # Two generators: original→proposal, proposal→original (for cycle-consistency)
    G_AB = GeneratorUNet(Config.TEXT_DIM).to(Config.DEVICE)  # Real -> Modified
    G_BA = GeneratorUNet(Config.TEXT_DIM).to(Config.DEVICE)  # Modified -> Real

    D_A  = Discriminator(Config.TEXT_DIM).to(Config.DEVICE)  # Real discriminator
    D_B  = Discriminator(Config.TEXT_DIM).to(Config.DEVICE)  # Modified discriminator

    # Losses
    crit_gan   = nn.MSELoss()    # LSGAN (more stable than BCE)
    crit_cycle = nn.L1Loss()     # Cycle consistency
    crit_id    = nn.L1Loss()     # Identity
    crit_pix   = nn.L1Loss()     # Pixel-wise fidelity

    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=Config.LR, betas=(Config.B1, Config.B2)
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=Config.LR * 0.5, betas=(Config.B1, Config.B2))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=Config.LR * 0.5, betas=(Config.B1, Config.B2))

    # LR Schedulers: decay after half the epochs
    def lr_lambda(e): return 1.0 - max(0, e - Config.EPOCHS // 2) / float(Config.EPOCHS // 2)
    sched_G   = torch.optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda)
    sched_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda)
    sched_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda)

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset    = VehicleIdeaDataset(Config.EXCEL_PATH, Config.DATA_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=0)

    logger.info(f"Training {Config.EPOCHS} epochs | {len(dataloader)} batches/epoch")

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        for i, (real_A, ids, mask, _) in enumerate(dataloader):
            real_A = real_A.to(Config.DEVICE)
            ids    = ids.to(Config.DEVICE)
            mask   = mask.to(Config.DEVICE)

            with torch.no_grad():
                text_emb = clip(input_ids=ids, attention_mask=mask).pooler_output

            patch_shape = (real_A.size(0), 1,
                           real_A.size(2) // 16, real_A.size(3) // 16)
            valid = torch.ones(*patch_shape, device=Config.DEVICE)
            fake  = torch.zeros(*patch_shape, device=Config.DEVICE)

            # ── Generators ──────────────────────────────────────────────────
            opt_G.zero_grad()

            fake_B  = G_AB(real_A, text_emb)            # A -> B (proposed)
            recov_A = G_BA(fake_B, text_emb)            # B -> A (reconstructed)
            fake_A  = G_BA(real_A, text_emb)            # Identity: B -> A on A
            recov_B = G_AB(fake_A, text_emb)            # Identity cycle

            # Adversarial
            loss_adv_AB = crit_gan(D_B(fake_B, text_emb), valid)
            loss_adv_BA = crit_gan(D_A(fake_A, text_emb), valid)
            loss_adv    = (loss_adv_AB + loss_adv_BA) * 0.5

            # Cycle consistency
            loss_cyc = (crit_cycle(recov_A, real_A) + crit_cycle(recov_B, fake_B)) * 0.5 * Config.LAMBDA_CYCLE

            # Identity: G_BA(real_A) should ≈ real_A
            loss_id = crit_id(fake_A, real_A) * Config.LAMBDA_ID

            # Pixel fidelity (keep the vehicle shape)
            loss_pix = crit_pix(fake_B, real_A) * Config.LAMBDA_PIXEL

            g_loss = loss_adv + loss_cyc + loss_id + loss_pix
            g_loss.backward()
            opt_G.step()

            # ── Discriminators ───────────────────────────────────────────────
            opt_D_A.zero_grad()
            loss_D_A = (crit_gan(D_A(real_A, text_emb), valid) +
                        crit_gan(D_A(fake_A.detach(), text_emb), fake)) * 0.5
            loss_D_A.backward(); opt_D_A.step()

            opt_D_B.zero_grad()
            loss_D_B = (crit_gan(D_B(real_A, text_emb), valid) +           # real in B domain
                        crit_gan(D_B(fake_B.detach(), text_emb), fake)) * 0.5
            loss_D_B.backward(); opt_D_B.step()

            if i % 5 == 0:
                logger.info(
                    f"[Ep {epoch:02d}/{Config.EPOCHS}] [Batch {i:02d}/{len(dataloader)}] "
                    f"G:{g_loss.item():.4f}  Adv:{loss_adv.item():.4f}  "
                    f"Cyc:{loss_cyc.item():.4f}  Pix:{loss_pix.item():.4f}  "
                    f"D_A:{loss_D_A.item():.4f}  D_B:{loss_D_B.item():.4f}"
                )

        sched_G.step(); sched_D_A.step(); sched_D_B.step()
        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {elapsed:.0f}s")

        # Save samples + checkpoints every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                comparison = torch.cat([real_A[:2], fake_B[:2]], dim=-1)
                save_image(comparison, f"{Config.OUTPUT_DIR}/ep{epoch:03d}_compare.png",
                           nrow=2, normalize=True)

            torch.save(G_AB.state_dict(), f"{Config.WEIGHTS_DIR}/G_AB_ep{epoch}.pth")
            torch.save(G_BA.state_dict(), f"{Config.WEIGHTS_DIR}/G_BA_ep{epoch}.pth")
            torch.save(D_A.state_dict(),  f"{Config.WEIGHTS_DIR}/D_A_ep{epoch}.pth")
            logger.info(f"Checkpoint saved (epoch {epoch})")

    # ── SAVE FINAL WEIGHTS ──────────────────────────────────────────────────
    torch.save(G_AB.state_dict(), Config.FINAL_G_PATH)
    torch.save(G_AB.state_dict(), f"{Config.WEIGHTS_DIR}/G_AB_FINAL.pth")
    torch.save(G_BA.state_dict(), f"{Config.WEIGHTS_DIR}/G_BA_FINAL.pth")
    logger.info(f"Training complete! G_AB saved to {Config.FINAL_G_PATH}")
    logger.info("Update vlm_engine.py: set MODEL_PATH = 'saved_models/unet_generator_final.pth'")


if __name__ == "__main__":
    train()
