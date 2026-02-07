import logging
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_owlvit():
    logger.info("Testing OWL-ViT Loading...")
    try:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        logger.info("Model Loaded Successfully.")
        
        # Create dummy image
        img = Image.new('RGB', (500, 500), color = 'white')
        
        texts = [["a photo of a car"]]
        inputs = processor(text=texts, images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logger.info("Inference Successful.")
        
    except Exception as e:
        logger.error(f"OWL-ViT Test Failed: {e}")

if __name__ == "__main__":
    test_owlvit()
