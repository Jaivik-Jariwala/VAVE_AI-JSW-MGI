"""
Image Caption Indexer for VAVE AI
Generates captions for all images in static/images/mg using BLIP model
and saves them to static/image_captions.json for semantic search.
"""
import os
import json
import logging
from pathlib import Path
from PIL import Image
import torch

logger = logging.getLogger(__name__)

def generate_index(force_regenerate=False):
    """
    Generate image caption index for semantic search.
    
    Args:
        force_regenerate: If True, regenerate index even if it exists
    
    Returns:
        dict: Dictionary mapping filenames to captions, or None if failed
    """
    # Get base directory (parent of utils)
    base_dir = Path(__file__).parent.parent.resolve()
    mg_dir = base_dir / "static" / "images" / "mg"
    index_path = base_dir / "static" / "image_captions.json"
    
    # Check if index already exists
    if index_path.exists() and not force_regenerate:
        logger.info(f"Image caption index already exists at {index_path}. Skipping generation.")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
                logger.info(f"Loaded existing index with {len(index)} captions.")
                return index
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}. Regenerating...")
    
    # Check if mg directory exists
    if not mg_dir.exists():
        logger.error(f"MG images directory not found: {mg_dir}")
        return None
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load BLIP model
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model_name = "Salesforce/blip-image-captioning-base"
        logger.info(f"Loading BLIP model: {model_name}")
        
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        
        logger.info("BLIP model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load BLIP model: {e}")
        return None
    
    # Get all image files
    image_files = sorted([f for f in mg_dir.glob("*.jpg")] + 
                        [f for f in mg_dir.glob("*.jpeg")] + 
                        [f for f in mg_dir.glob("*.png")])
    
    if not image_files:
        logger.warning(f"No image files found in {mg_dir}")
        return None
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Generate captions
    captions = {}
    processed = 0
    failed = 0
    
    for img_path in image_files:
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            
            # Generate caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50, num_beams=3)
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Store caption with filename (relative to mg_dir)
            filename = img_path.name
            captions[filename] = caption
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(image_files)} images...")
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")
            failed += 1
            continue
    
    logger.info(f"Caption generation complete: {processed} successful, {failed} failed")
    
    # Save to JSON
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(captions)} captions to {index_path}")
        return captions
    except Exception as e:
        logger.error(f"Failed to save index: {e}")
        return None


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    force = '--force' in sys.argv
    
    logger.info("Starting image caption index generation...")
    result = generate_index(force_regenerate=force)
    
    if result:
        logger.info(f"Success! Generated {len(result)} captions.")
    else:
        logger.error("Failed to generate index.")
        sys.exit(1)

