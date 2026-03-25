import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import logging
import time
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 0. HARD REQUIREMENT: ENFORCE GPU ---
if not torch.cuda.is_available():
    logger.error("CRITICAL ERROR: CUDA/GPU is NOT available!")
    logger.error("Advanced Image-to-Image GANs require a GPU to train.")
    logger.error("Please restart this script after you have fixed your PyTorch installation to use CUDA.")
    logger.error("Installation command: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# --- 1. CONFIGURATION ---
class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 8 # Better for stability
    EPOCHS = 100
    LR_G = 0.0002
    LR_D = 0.0001 # Discriminator learns slightly slower to prevent overpowering
    B1 = 0.5
    B2 = 0.999
    TEXT_EMBED_DIM = 512
    # Adjust paths since we are in a subfolder 'advanced_gan'
    DATA_DIR = "../images/mg" 
    EXCEL_PATH = "../AIML Dummy Ideas Data.xlsx"
    OUTPUT_DIR = "training_samples"
    MODEL_SAVE_DIR = "weights"
    DEVICE = torch.device("cuda")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

# --- 2. DATASET (Reads from Original Excel) ---
class ConditionalVehicleDataset(Dataset):
    def __init__(self, excel_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        
        try:
            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
                for index, row in df.iterrows():
                    idea_text = str(row.get("Cost Reduction Idea Proposal") or "")
                    if len(idea_text) < 5 or idea_text.lower() == "nan": continue
                        
                    img_id = index + 1
                    img_path = os.path.join(img_dir, f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        self.data.append({"img_path": img_path, "text": idea_text})
            logger.info(f"Loaded {len(self.data)} training pairs.")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        if self.transform: img = self.transform(img)
            
        tokens = self.tokenizer(item["text"], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return img, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)

# --- 3. U-NET GENERATOR (Image to Image Conditioned on Text) ---
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """
    Takes an input image of the vehicle, and a text embedding, and outputs the modified vehicle.
    """
    def __init__(self, text_dim=512):
        super(GeneratorUNet, self).__init__()
        
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )

        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512 + 256, 512, dropout=0.5) # +256 for Text Injection
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, text_embedding):
        # Image encoding
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5) # Bottleneck: [Batch, 512, 4, 4] for 256x256 image

        # Text Injection at Bottleneck
        text_f = self.text_fc(text_embedding) # [Batch, 256]
        text_f = text_f.view(text_f.size(0), 256, 1, 1).expand(-1, -1, d6.size(2), d6.size(3))
        
        # Combine image bottleneck and text
        bottleneck = torch.cat((d6, text_f), 1) # [Batch, 768, 4, 4]

        # Decoding
        u1 = self.up1(bottleneck, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)

# --- 4. SPECTRAL NORMALIZED DISCRIMINATOR (Prevents Mode Collapse) ---
class Discriminator(nn.Module):
    def __init__(self, text_dim=512):
        super(Discriminator, self).__init__()
        
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )

        def discriminator_block(in_filters, out_filters, normalization=True):
            # Spectral constraint on weights to prevent exploding gradients (0.69 stagnation)
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512) # Output size: [Batch, 512, 16, 16]
        )

        # PatchGAN style classifier output
        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(512 + 256, 1, 3, padding=1))

    def forward(self, img, text_embedding):
        img_features = self.model(img)
        
        text_f = self.text_fc(text_embedding)
        text_f = text_f.view(text_f.size(0), 256, 1, 1).expand(-1, -1, img_features.size(2), img_features.size(3))
        
        combined = torch.cat((img_features, text_f), 1)
        validity = self.final_conv(combined)
        return validity # LSGAN / Hinge Loss does not use Sigmoid here

# --- 5. TRAINING LOOP (With Least Squares / Hinge Loss) ---
def train_unet_gan():
    logger.info(f"Warming up Advanced U-Net GAN on Device: {Config.DEVICE}")

    generator = GeneratorUNet().to(Config.DEVICE)
    discriminator = Discriminator().to(Config.DEVICE)
    
    logger.info("Loading CLIP Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(Config.DEVICE)
    text_encoder.eval() 
    
    # We use MSELoss (Least Squares GAN) which performs significantly better than BCE for image translation
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss() # Encourages structural similarity to the original vehicle
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Config.LR_G, betas=(Config.B1, Config.B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.LR_D, betas=(Config.B1, Config.B2))

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ConditionalVehicleDataset(Config.EXCEL_PATH, Config.DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    lambda_pixel = 100 # Weight for structural similarity

    for epoch in range(Config.EPOCHS):
        for i, (imgs, input_ids, attention_mask) in enumerate(dataloader):
            
            real_imgs = imgs.to(Config.DEVICE)
            input_ids = input_ids.to(Config.DEVICE)
            attention_mask = attention_mask.to(Config.DEVICE)
            
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

            # Create labels based on output size of discriminator
            valid = torch.ones((real_imgs.size(0), 1, real_imgs.size(2)//16, real_imgs.size(3)//16)).to(Config.DEVICE)
            fake = torch.zeros((real_imgs.size(0), 1, real_imgs.size(2)//16, real_imgs.size(3)//16)).to(Config.DEVICE)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            gen_imgs = generator(real_imgs, text_embeddings)

            # GAN loss
            pred_fake = discriminator(gen_imgs, text_embeddings)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss (Keep it looking like the same vehicle)
            loss_pixel = criterion_pixelwise(gen_imgs, real_imgs)

            g_loss = loss_GAN + lambda_pixel * loss_pixel
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(real_imgs, text_embeddings)
            loss_real = criterion_GAN(pred_real, valid)
            
            pred_fake = discriminator(gen_imgs.detach(), text_embeddings)
            loss_fake = criterion_GAN(pred_fake, fake)

            d_loss = 0.5 * (loss_real + loss_fake)
            d_loss.backward()
            optimizer_D.step()

            if i % 5 == 0:
                logger.info(f"[Epoch {epoch}/{Config.EPOCHS}] [Batch {i}/{len(dataloader)}] "
                           f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                           f"[Pixel L1: {loss_pixel.item():.4f}]")

        if epoch % 5 == 0:
            # Save combination of Original and Modified for comparison
            comparison = torch.cat((real_imgs.data[:4], gen_imgs.data[:4]), -2)
            save_image(comparison, f"{Config.OUTPUT_DIR}/epoch_{epoch}_comparison.png", nrow=4, normalize=True)
            
            torch.save(generator.state_dict(), f"{Config.MODEL_SAVE_DIR}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{Config.MODEL_SAVE_DIR}/discriminator_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_unet_gan()
