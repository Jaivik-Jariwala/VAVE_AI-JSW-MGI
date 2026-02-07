import cv2
import numpy as np
import os
import sys

# Synthesize dummy images for testing (since we may not have real ones readily available/mapped)
def create_dummy_car(width=640, height=480):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (200, 200, 200) # Grey background
    
    # Draw a "car" (blue rectangle)
    cv2.rectangle(img, (100, 200), (540, 400), (255, 0, 0), -1)
    # Wheel
    cv2.circle(img, (200, 400), 40, (0, 0, 0), -1)
    cv2.circle(img, (440, 400), 40, (0, 0, 0), -1)
    return img

def create_dummy_part(width=300, height=300):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50) # Dark background
    
    # Draw a "component" (Red Brake Caliper)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.putText(img, "BREMBO", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

if __name__ == "__main__":
    os.makedirs("test_images", exist_ok=True)
    car_path = "test_images/car.jpg"
    part_path = "test_images/competitor_part.jpg"
    
    cv2.imwrite(car_path, create_dummy_car())
    cv2.imwrite(part_path, create_dummy_part())
    
    print("Created dummy images.")
    
    # Import the engine (need to set path or copy function)
    # We will test the _smart_merge_images_cv function specifically
    sys.path.append(os.getcwd())
    
    try:
        from vlm_engine import VLMEngine
        # Mock dependencies
        engine = VLMEngine(None, None, None)
        
        # Test Params
        # Comp bbox: Center middle where we drew the red box
        # 300x300 image. Box is 100,100 to 200,200
        comp_bbox = [100/300, 100/300, 200/300, 200/300] # ymin, xmin, ymax, xmax
        
        # Target: Wheel area
        # 640x480. Wheel at 200,400. Let's aim there.
        # Center x = 200/640 = 0.31
        # Center y = 400/480 = 0.83
        # Size 80x80 -> w=0.12, h=0.16
        current_bbox = {"center_x": 0.31, "center_y": 0.83, "width_ratio": 0.15, "height_ratio": 0.15}
        
        print("Testing Smart Merge...")
        result = engine._smart_merge_images_cv(car_path, part_path, current_bbox, comp_bbox)
        
        if result:
            print(f"SUCCESS: Generated {result}")
            # Verify file exists
            if os.path.exists(result):
                print("Output file confirmed.")
            else:
                print("Output file missing?!")
        else:
            print("FAILURE: Merge returned None")
            
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
