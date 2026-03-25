import os
import sys
import itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import logging
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 0. HARD REQUIREMENT: ENFORCE GPU ---
if not torch.cuda.is_available():
    logger.error("CRITICAL ERROR: CUDA/GPU is NOT available!")
    logger.error("CycleGAN requires a GPU to train efficiently.")
    sys.exit(1)

# --- 1. CONFIGURATION ---
class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 4 
    EPOCHS = 200 # CycleGAN needs more epochs
    LR = 0.0002
    B1 = 0.5
    B2 = 0.999
    TEXT_EMBED_DIM = 512
    LAMBDA_CYC = 10.0 # Weight for cycle-consistency loss
    LAMBDA_ID = 5.0   # Weight for identity loss
    DATA_DIR = "../images/mg" 
    EXCEL_PATH = "../AIML Dummy Ideas Data.xlsx"
    OUTPUT_DIR = "cycle_gan_samples"
    MODEL_SAVE_DIR = "cycle_gan_weights"
    DEVICE = torch.device("cuda")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

# --- 2. DATASET ---
class CycleVehicleDataset(Dataset):
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
            logger.info(f"Loaded {len(self.data)} images for Domain A (Original) and Domain B (Conditioned).")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_A = Image.open(item["img_path"]).convert("RGB")
        
        # For true CycleGAN we usually need two distinct datasets (A and B).
        # Since we only have images of A, we use the same image, but we are teaching
        # Generator A->B to modify it based on text, and B->A to remove the modification.
        
        if self.transform: 
            img_A = self.transform(img_A)
            
        tokens = self.tokenizer(item["text"], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return img_A, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


# --- 3. RESNET BLOCKS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

# --- 4. GENERATOR (ResNet Architecture) ---
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9, text_dim=512):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        # Text Injection Projection
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )

        # Initial convolution block (accepts image + expanded text embedding)
        out_features = 64
        # We add 256 to in_channels to account for the text feature map we will concatenate
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels + 256, out_features, 7), 
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        in_features = out_features
        out_features = in_features * 2
        
        down_blocks = []
        for _ in range(2):
            down_blocks += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
            
        self.downsampling = nn.Sequential(*down_blocks)

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks += [ResidualBlock(in_features)]
        self.residuals = nn.Sequential(*res_blocks)

        # Upsampling
        out_features = in_features // 2
        up_blocks = []
        for _ in range(2):
            up_blocks += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
            
        self.upsampling = nn.Sequential(*up_blocks)

        # Output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        )

    def forward(self, x, text_embedding):
        # 1. Expand text embedding to match image spatial dimensions
        text_f = self.text_fc(text_embedding) # [Batch, 256]
        text_f = text_f.view(text_f.size(0), 256, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        
        # 2. Concatenate Image and Text
        x_concat = torch.cat((x, text_f), 1)

        # 3. Pass through network
        out = self.initial(x_concat)
        out = self.downsampling(out)
        out = self.residuals(out)
        out = self.upsampling(out)
        out = self.final(out)
        return out

# --- 5. DISCRIMINATOR ---
class Discriminator(nn.Module):
    def __init__(self, input_shape, text_dim=512):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )
        
        # PatchGAN style classifier output, accepting image features + text
        self.final_conv = nn.Conv2d(512 + 256, 1, 3, padding=1)

    def forward(self, img, text_embedding):
        img_features = self.model(img)
        
        text_f = self.text_fc(text_embedding)
        text_f = text_f.view(text_f.size(0), 256, 1, 1).expand(-1, -1, img_features.size(2), img_features.size(3))
        
        combined = torch.cat((img_features, text_f), 1)
        return self.final_conv(combined)

# --- 6. TRAINING LOOP ---
def train_cycle_gan():
    logger.info(f"Starting Text-Conditioned CycleGAN on Device: {Config.DEVICE}")

    # Initialize generator and discriminator
    # G_AB: Generates Modified Image (B) from Original Image (A) + Text
    G_AB = GeneratorResNet((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(Config.DEVICE)
    # G_BA: Reconstructs Original Image (A) from Modified Image (B) + Text
    G_BA = GeneratorResNet((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(Config.DEVICE)
    
    D_A = Discriminator((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(Config.DEVICE)
    D_B = Discriminator((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).to(Config.DEVICE)
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(Config.DEVICE)
    text_encoder.eval() 
    
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=Config.LR, betas=(Config.B1, Config.B2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=Config.LR, betas=(Config.B1, Config.B2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=Config.LR, betas=(Config.B1, Config.B2))

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = CycleVehicleDataset(Config.EXCEL_PATH, Config.DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    for epoch in range(Config.EPOCHS):
        for i, (imgs_A, input_ids, attention_mask) in enumerate(dataloader):
            
            real_A = imgs_A.to(Config.DEVICE)
            # In a true CycleGAN we have unpaired real_B. 
            # Because this is a guided overlay task, we treat real_A as our source,
            # and rely heavily on the text-guidance and cycle-consistency to create B.
            # We don't have a dataset of "real modified cars", so we disable D_B's real vs fake 
            # and instead focus on D_A making sure the reconstruction looks right, 
            # and G_AB making structural changes guided by text.
            
            input_ids = input_ids.to(Config.DEVICE)
            attention_mask = attention_mask.to(Config.DEVICE)
            
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

            valid = torch.ones((real_A.size(0), 1, real_A.size(2)//16, real_A.size(3)//16)).to(Config.DEVICE)
            fake = torch.zeros((real_A.size(0), 1, real_A.size(2)//16, real_A.size(3)//16)).to(Config.DEVICE)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss (G_BA(A) should equal A)
            loss_id_A = criterion_identity(G_BA(real_A, text_embeddings), real_A)

            # GAN loss
            fake_B = G_AB(real_A, text_embeddings)
            # Evaluate fake_B using D_B
            loss_GAN_AB = criterion_GAN(D_B(fake_B, text_embeddings), valid) 
            
            # Reconstruction A -> B -> A
            recov_A = G_BA(fake_B, text_embeddings)
            loss_cycle_A = criterion_cycle(recov_A, real_A)

            # Total generator loss
            loss_G = loss_GAN_AB + Config.LAMBDA_CYC * loss_cycle_A + Config.LAMBDA_ID * loss_id_A

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A, text_embeddings), valid)
            # Fake loss (on reconstructed A)
            loss_fake = criterion_GAN(D_A(recov_A.detach(), text_embeddings), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            
            # We don't have real "Modified" cars, so D_B only learns to penalize unrealistic generated mods
            loss_fake_B = criterion_GAN(D_B(fake_B.detach(), text_embeddings), fake)
            loss_D_B = loss_fake_B * 0.5 
            
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            if i % 10 == 0:
                logger.info(f"[Epoch {epoch}/{Config.EPOCHS}] [Batch {i}/{len(dataloader)}] "
                           f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                           f"[Cycle A: {loss_cycle_A.item():.4f}]")

        if epoch % 5 == 0:
            comparison = torch.cat((real_A.data[:4], fake_B.data[:4], recov_A.data[:4]), -2)
            save_image(comparison, f"{Config.OUTPUT_DIR}/epoch_{epoch}_sample.png", nrow=4, normalize=True)
            
            torch.save(G_AB.state_dict(), f"{Config.MODEL_SAVE_DIR}/G_AB_epoch_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"{Config.MODEL_SAVE_DIR}/G_BA_epoch_{epoch}.pth")
            torch.save(D_A.state_dict(), f"{Config.MODEL_SAVE_DIR}/D_A_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_cycle_gan()
