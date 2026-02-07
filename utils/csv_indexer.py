import pandas as pd
import json
import os

def generate_ground_truth_index():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "AIML Dummy Ideas Data.xlsx") # Or your CSV path
    json_path = os.path.join(base_dir, "static", "image_captions.json")
    
    print(f"Reading data from: {csv_path}")
    
    try:
        # 2. Read the specific sheet
        # Adjust 'AIML Dummy ideas' if your sheet name is different
        try:
            df = pd.read_excel(csv_path, sheet_name='AIML Dummy ideas')
        except:
            # Fallback to reading it as CSV if excel fails or if user uses the converted csv
            csv_path = os.path.join(base_dir, "AIML Dummy Ideas Data.xlsx - AIML Dummy ideas.csv")
            df = pd.read_csv(csv_path)

        # 3. Create the Mapping
        # Formula: Row Index 0 (Excel Row 2) -> "2.jpg"
        image_map = {}
        
        # Identify the correct column for description
        # We prefer "Cost Reduction Idea Proposal" or "Cost Reduction Idea"
        col_name = None
        possible_cols = ["Cost Reduction Idea Proposal", "Cost Reduction Idea", "Idea", "Proposal"]
        for c in possible_cols:
            if c in df.columns:
                col_name = c
                break
        
        if not col_name:
            print(f"Error: Could not find idea column. Available: {df.columns}")
            return

        print(f"Indexing column: '{col_name}'...")

        for index, row in df.iterrows():
            # Excel Row 2 is Index 0.
            # So Image Name = Index + 2
            img_name = f"{index + 2}.jpg"
            
            # The 'caption' is the actual engineering text
            description = str(row[col_name]).strip()
            
            # Clean up the text
            if description and description.lower() != 'nan':
                image_map[img_name] = description

        # 4. Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(image_map, f, indent=2)
            
        print(f"Success! Indexed {len(image_map)} images to {json_path}")
        print(f"Sample: '2.jpg' -> '{image_map.get('2.jpg', 'Not Found')}'[:50]...")

    except Exception as e:
        print(f"Failed to index: {e}")

if __name__ == "__main__":
    generate_ground_truth_index()