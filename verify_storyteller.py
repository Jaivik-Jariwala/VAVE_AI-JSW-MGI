import os
import sys
import logging
from pathlib import Path
from vlm_engine import VLMEngine, VisualStoryteller

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_storyteller():
    """
    Verifies that the VisualStoryteller class correctly composites images.
    """
    print("\n--- Verifying Visual Storyteller ---")
    
    # 1. Setup paths
    base_dir = Path(__file__).parent.resolve()
    gen_dir = base_dir / "static" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Use existing images for testing
    # Find some jpgs in static/images/mg or use placeholders
    img_dir = base_dir / "static" / "images" / "mg"
    images = list(img_dir.glob("*.jpg"))
    
    if not images:
        print("No test images found in static/images/mg. Checking defaults...")
        img_dir = base_dir / "static" / "defaults"
        images = list(img_dir.glob("*.jpg"))
        
    if not images:
        print("No images found at all. Cannot verify.")
        return

    current = str(images[0])
    competitor = str(images[1]) if len(images) > 1 else current
    proposal = str(images[2]) if len(images) > 2 else current
    
    print(f"Test Inputs:\nCurrent: {current}\nCompetitor: {competitor}\nProposal: {proposal}")
    
    # 3. Instantiate Storyteller
    storyteller = VisualStoryteller(base_dir, gen_dir)
    
    # 4. Generate Composite
    # Simulate a bounding box: [ymin, xmin, ymax, xmax] -> Center box [0.3, 0.3, 0.7, 0.7]
    roi_box = [0.3, 0.3, 0.7, 0.7]
    
    output_path = storyteller.generate_composite(
        current_ref=current,
        competitor_ref=competitor,
        proposal_ref=proposal,
        component_name="Brake Caliper Assembly",
        roi_box=roi_box
    )
    
    if output_path:
        print(f"\nSUCCESS: Generated Dashboard Image:\n{output_path}")
        # Resolve to absolute path to verify existence
        abs_path = base_dir / output_path
        if abs_path.exists():
             print(f"File verified on disk: {abs_path}")
        else:
             print("File path returned but file not found on disk!")
    else:
        print("\nFAILURE: Storyteller returned None")

if __name__ == "__main__":
    verify_storyteller()
