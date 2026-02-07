import os
import re

# Adjust path to your actual folder
IMG_DIR = "static/images/mg"

print(f"--- DIAGNOSTIC: Checking {IMG_DIR} ---")
if not os.path.exists(IMG_DIR):
    print("❌ Directory not found!")
else:
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Found {len(files)} images.")
    print("Filenames:", files[:10]) # Show first 10

    # Test the Matching Logic
    test_queries = ["seat", "bumper", "chassis", "light", "panel"]
    
    print("\n--- SIMULATING SEARCH ---")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        matches = []
        for f in files:
            if q in f.lower():
                matches.append(f)
        
        if matches:
            print(f"✅ Match Found: {matches}")
        else:
            print(f"⚠️  No filename match. System would use RANDOM fallback.")