
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLM_Verifier")

# Ensure we can import from the project root
sys.path.append(os.getcwd())

try:
    import cv2
    import numpy as np
    from PIL import Image
    print("OpenCV and PIL imported successfully.")
except ImportError as e:
    print(f"CRITICAL: Missing dependency {e}. Please install opencv-python pillow.")
    sys.exit(1)

from vlm_engine import VLMEngine

def test_saliency():
    print("\n--- Testing OpenCV Saliency Fallback ---")
    
    # Initialize Engine (Mock dependencies as None since we only test specific methods)
    vlm = VLMEngine(None, None, None)
    
    # Use a real image from the list we found
    test_img_rel = "static/images/mg/10.jpg" 
    test_img_path = os.path.abspath(test_img_rel)
    
    if not os.path.exists(test_img_path):
        print(f"Test image not found at {test_img_path}. Skipping.")
        return

    print(f"Testing on {test_img_path}")
    
    # Call Saliency directly
    bbox = vlm._get_salient_bbox(test_img_rel)
    
    if bbox:
        print(f"SUCCESS: Saliency detected bbox: {bbox}")
        
        # Visualize it
        img = cv2.imread(test_img_path)
        h, w = img.shape[:2]
        ymin, xmin, ymax, xmax = bbox
        
        # Convert to pixels
        pt1 = (int(xmin * w), int(ymin * h))
        pt2 = (int(xmax * w), int(ymax * h))
        
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        
        out_path = "vlm_saliency_test_output.jpg"
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")
    else:
        print("FAILURE: Saliency returned None.")

def test_overlay():
    print("\n--- Testing Hybrid Overlay ---")
    vlm = VLMEngine(None, None, None)
    
    # We need two images. Let's use 10.jpg as base and 11.jpg as competitor
    base_rel = "static/images/mg/10.jpg"
    comp_rel = "static/images/mg/11.jpg"
    
    if not os.path.exists(base_rel) or not os.path.exists(comp_rel):
        print("Missing test images for overlay.")
        return
        
    # Simulate a "Brake" prompt (though these images might be random cars)
    # We just want to see if the pipeline runs without crashing
    idea_text = "Optimize brake caliper material"
    
    # We extracted "Brake" as visual prompt
    ctx = {
        "visual_prompt": "Brake",
        "mg_vehicle_image": base_rel # Needed for target inference
    }
    
    # This should trigger the full chain: 
    # 1. Detect "Brake" in competitor (via HF or Saliency)
    # 2. Detect "Brake" location in Base (via Inference)
    # 3. Merge
    
    res_path = vlm._compose_competitor_overlay(base_rel, comp_rel, idea_text, ctx)
    
    if res_path:
        print(f"SUCCESS: Overlay generated at {res_path}")
    else:
        print("FAILURE: Overlay returned None.")

if __name__ == "__main__":
    test_saliency()
    test_overlay()
