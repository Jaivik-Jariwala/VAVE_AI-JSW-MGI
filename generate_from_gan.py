"""
VAVE GAN Inference Script
=========================
Loads the trained GAN generator model and generates vehicle proposal images from text ideas.

Usage:
    python generate_from_gan.py --idea "Replace dual LED lamps with single LED" --output my_proposal.png
    python generate_from_gan.py --idea "Replace plastic caps with adhesive stickers" --n_samples 4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION (MUST MATCH train_vave_gan.py) ---
IMAGE_SIZE = 256
LATENT_DIM = 100
TEXT_EMBED_DIM = 512
MODEL_PATH = "saved_models/generator_final.pth"
OUTPUT_DIR = "gan_generated"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GENERATOR ARCHITECTURE (copy from train_vave_gan.py) ---
class Config:
    IMAGE_SIZE = IMAGE_SIZE
    LATENT_DIM = LATENT_DIM
    TEXT_EMBED_DIM = TEXT_EMBED_DIM

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.text_fc = nn.Sequential(
            nn.Linear(config.TEXT_EMBED_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.init_size = config.IMAGE_SIZE // 4
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
        text_f = self.text_fc(text_embedding)
        gen_input = torch.cat((text_f, noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# --- INFERENCE FUNCTION ---
def generate_images(idea_text: str, n_samples: int = 4, output_path: str = None, 
                    model_path: str = MODEL_PATH):
    """
    Generate vehicle proposal images from a text idea using the trained GAN.
    
    Args:
        idea_text: The modification idea text from the Excel doc.
        n_samples: How many variations to generate.
        output_path: Custom path to save the resulting image grid.
        model_path: Path to the saved generator weights (.pth file).
    
    Returns:
        output_path: Path where the image was saved.
    """
    logger.info(f"Loading Generator from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        sys.exit(1)
    
    # Load Generator
    config = Config()
    generator = Generator(config)
    generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
    generator.to(DEVICE)
    generator.eval()
    logger.info("Generator loaded successfully.")
    
    # Load CLIP Text Encoder
    logger.info("Loading CLIP Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    text_encoder.eval()
    
    # Encode the idea text
    logger.info(f"Encoding idea: '{idea_text[:80]}...'")
    tokens = tokenizer(idea_text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        text_outputs = text_encoder(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE)
        )
        text_embedding = text_outputs.pooler_output  # [1, 512]
        
        # Repeat for batch of n_samples
        text_embedding = text_embedding.repeat(n_samples, 1) # [n_samples, 512]
        
        # Generate
        noise = torch.randn(n_samples, LATENT_DIM).to(DEVICE)
        gen_imgs = generator(noise, text_embedding) # [n_samples, 3, 256, 256]
    
    # Save the output
    if output_path is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in idea_text[:40])
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    
    nrow = min(n_samples, 4)
    save_image(gen_imgs.data, output_path, nrow=nrow, normalize=True)
    logger.info(f"Generated {n_samples} image(s) saved to: {output_path}")
    
    return output_path


# --- BATCH GENERATION from Excel ---
def generate_from_excel(excel_path: str, num_ideas: int = 5):
    """
    Read the first N ideas from the Excel and generate GAN images for each.
    """
    try:
        import pandas as pd
        df = pd.read_excel(excel_path)
        count = 0
        for index, row in df.iterrows():
            idea_text = str(row.get("Cost Reduction Idea Proposal") or "")
            if len(idea_text) < 5 or idea_text.lower() == "nan":
                continue
            
            logger.info(f"=== Generating for Row {index+1}: {idea_text[:60]}... ===")
            output_path = os.path.join(OUTPUT_DIR, f"row_{index+1}_proposal.png")
            generate_images(idea_text=idea_text, n_samples=1, output_path=output_path)
            count += 1
            if count >= num_ideas:
                break
        
        logger.info(f"\nDone! Generated {count} proposal images in '{OUTPUT_DIR}/' folder.")
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vehicle modification proposals using trained GAN")
    parser.add_argument("--idea", type=str, default=None, help="Engineering modification idea text")
    parser.add_argument("--excel", type=str, default=None, help="Path to Excel file for batch generation")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of variations to generate")
    parser.add_argument("--num_ideas", type=int, default=5, help="Number of Excel rows to process (batch mode)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to generator weights")
    
    args = parser.parse_args()
    
    logger.info(f"Running on Device: {DEVICE}")
    
    if args.excel:
        generate_from_excel(excel_path=args.excel, num_ideas=args.num_ideas)
    elif args.idea:
        generate_images(idea_text=args.idea, n_samples=args.n_samples, output_path=args.output, model_path=args.model)
    else:
        # Default demo: Try from Excel
        logger.info("No --idea or --excel provided. Running demo from default Excel...")
        generate_from_excel(excel_path="AIML Dummy Ideas Data.xlsx", num_ideas=3)
