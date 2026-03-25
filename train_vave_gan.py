import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import json
import logging
import random
import time
from transformers import CLIPTextModel, CLIPTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
# --- 1. CONFIGURATION ---
class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 4 # Keep small for basic GPUs
    EPOCHS = 100
    LR = 0.0002
    B1 = 0.5
    B2 = 0.999
    LATENT_DIM = 100
    TEXT_EMBED_DIM = 512 # CLIP output dimension
    DATA_DIR = "images/mg" # Assuming images are collected here
    JSON_DATA_PATH = "AIML Dummy Ideas Data.xlsx"
    OUTPUT_DIR = "gan_output"
    MODEL_SAVE_DIR = "saved_models"
    # Force GPU usage if available, else CPU
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
        logger.warning("CUDA/GPU is NOT available! Training will run on CPU and be very slow.")

# --- 2. DATASET CONTROLLER ---
class VehicleIdeaDataset(Dataset):
    """
    Loads Vehicle Images and their associated Engineering Modification Idea texts straight from Excel.
    """
    def __init__(self, excel_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load Idea Data
        self.data = []
        try:
            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
                
                # The images are named 1.jpg, 2.jpg... up to 213.jpg
                # We can map them via dataframe index + 1 (assuming row 0 -> 1.jpg)
                for index, row in df.iterrows():
                    # Attempt to find the idea column
                    idea_text = str(row.get("Cost Reduction Idea Proposal") or row.get("Cost Reduction Idea") or "")
                    
                    if not idea_text or "nan" == idea_text.lower() or len(idea_text) < 5:
                        continue
                        
                    # Calculate image ID (index + 1)
                    img_id = index + 1
                    img_path = os.path.join(img_dir, f"{img_id}.jpg")
                    
                    if os.path.exists(img_path):
                        self.data.append({"img_path": img_path, "text": idea_text})
            
            logger.info(f"Loaded {len(self.data)} valid image-text pairs for training.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")

        # CLIP Tokenizer for Text Processing
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process Image
        img = Image.open(item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        # Process Text
        tokens = self.tokenizer(
            item["text"], 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return img, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), item["text"]

# --- 3. GENERATOR ARCHITECTURE (cGAN with AdaIN/Concat) ---
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        # Process text embedding
        self.text_fc = nn.Sequential(
            nn.Linear(config.TEXT_EMBED_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Generator starting point (noise + text)
        self.init_size = config.IMAGE_SIZE // 4 # e.g., 64 if config is 256
        self.l1 = nn.Sequential(nn.Linear(config.LATENT_DIM + 256, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, text_embedding):
        # Conditionally project Text
        text_f = self.text_fc(text_embedding)
        
        # Combine noise vector + Text Feature
        gen_input = torch.cat((text_f, noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- 4. DISCRIMINATOR ARCHITECTURE ---
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.text_fc = nn.Sequential(
            nn.Linear(config.TEXT_EMBED_DIM, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        ds_size = config.IMAGE_SIZE // 2 ** 4
        # Fully connected matching Image features + Text Features
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2 + 128, 1), 
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        
        # Project text
        text_f = self.text_fc(text_embedding)
        
        # Concat img + text features
        validity = self.adv_layer(torch.cat((out, text_f), -1))
        return validity

# --- 5. TRAINING LOOP ---
def train_gan():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    
    logger.info(f"Starting GAN Training on Device: {Config.DEVICE}")

    # 1. Models
    generator = Generator(Config).to(Config.DEVICE)
    discriminator = Discriminator(Config).to(Config.DEVICE)
    
    # 2. Pre-trained Text Encoder (CLIP) - Frozen (not trained)
    logger.info("Loading CLIP Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(Config.DEVICE)
    text_encoder.eval() # Freeze weights
    
    # 3. Optimizers & Loss
    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Config.LR, betas=(Config.B1, Config.B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.LR, betas=(Config.B1, Config.B2))

    # 4. Data
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = VehicleIdeaDataset(Config.JSON_DATA_PATH, Config.DATA_DIR, transform=transform)
    if len(dataset) < 4:
        logger.error("Not enough images in the dataset to train a GAN. Need at least 4 valid image-text pairs.")
        return
        
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 5. Training Epochs
    for epoch in range(Config.EPOCHS):
        for i, (imgs, input_ids, attention_mask, raw_texts) in enumerate(dataloader):
            
            valid = torch.ones(Config.BATCH_SIZE, 1, requires_grad=False).to(Config.DEVICE)
            fake = torch.zeros(Config.BATCH_SIZE, 1, requires_grad=False).to(Config.DEVICE)

            # Move inputs to device
            real_imgs = imgs.to(Config.DEVICE)
            input_ids = input_ids.to(Config.DEVICE)
            attention_mask = attention_mask.to(Config.DEVICE)
            
            # Extract Text Embeddings using CLIP
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = text_outputs.pooler_output # Shape: [Batch, 512]

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(Config.BATCH_SIZE, Config.LATENT_DIM).to(Config.DEVICE)

            # Generate a batch of images conditioned on text
            gen_imgs = generator(z, text_embeddings)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, text_embeddings)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            validity_real = discriminator(real_imgs, text_embeddings)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_imgs.detach(), text_embeddings)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Log Progress
            if i % 10 == 0:
                logger.info(
                    f"[Epoch {epoch}/{Config.EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        # Save Sample Images Every 10 Epochs
        if epoch % 10 == 0:
            save_image(gen_imgs.data[:4], f"{Config.OUTPUT_DIR}/epoch_{epoch}_sample.png", nrow=2, normalize=True)
            logger.info(f"Saved sample image: {Config.OUTPUT_DIR}/epoch_{epoch}_sample.png")
            
            # Save Model Checkpoints
            torch.save(generator.state_dict(), f"{Config.MODEL_SAVE_DIR}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{Config.MODEL_SAVE_DIR}/discriminator_epoch_{epoch}.pth")

    # Save Final Model
    torch.save(generator.state_dict(), f"{Config.MODEL_SAVE_DIR}/generator_final.pth")
    logger.info("Training Complete! Models saved to /saved_models")


if __name__ == "__main__":
    # Note: Requires PyTorch, Torchvision, and Transformers (HuggingFace) to be installed.
    # pip install torch torchvision transformers pillow pandas numpy
    try:
        train_gan()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
