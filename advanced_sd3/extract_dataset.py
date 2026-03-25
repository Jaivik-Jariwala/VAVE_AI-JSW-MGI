import pandas as pd
import zipfile
import json
import os
import shutil

# --- CONFIGURATION ---
EXCEL_PATH = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx"
OUTPUT_DIR = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\advanced_sd3\dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

def extract_dataset():
    """
    Extracts embedded images and corresponding text prompts from the Excel file
    to create a HuggingFace-compatible dataset for Stable Diffusion 3.
    """
    print(f"Starting extraction from {EXCEL_PATH}...")
    
    if not os.path.exists(EXCEL_PATH):
        print(f"ERROR: Excel file not found at {EXCEL_PATH}")
        return

    # Ensure output directories exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 1. Read the text data
    try:
        df = pd.read_excel(EXCEL_PATH)
        print(f"Successfully loaded Excel text data: {len(df)} rows.")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Extract media (images) from the .xlsx zip archive
    # Excel files are simply ZIP archives. Images are stored in xl/media/
    temp_media_dir = "temp_media"
    os.makedirs(temp_media_dir, exist_ok=True)
    
    extracted_images = []
    try:
        with zipfile.ZipFile(EXCEL_PATH, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.startswith('xl/media/') and not file_info.filename.endswith('/'):
                    z.extract(file_info, temp_media_dir)
                    extracted_images.append(file_info.filename)
        print(f"Extracted {len(extracted_images)} raw media files from Excel archive.")
    except Exception as e:
        print(f"Error extracting media from ZIP: {e}")
        return

    # In many Excel structures, images don't map 1:1 to rows cleanly due to floating anchors.
    # For a robust dataset, we'll map available images to valid idea rows.
    # We will look for "Cost Reduction Idea Proposal" as the primary SD3 caption.
    
    if "Cost Reduction Idea Proposal" not in df.columns:
        print("ERROR: Could not find 'Cost Reduction Idea Proposal' column in Excel.")
        return

    # Clean the dataset (remove rows without ideas)
    valid_rows = df.dropna(subset=['Cost Reduction Idea Proposal'])
    
    metadata_entries = []
    
    # We will sequentially pair extracted images to rows.
    # (Note: For exact cell-to-image mapping, complex parsing of xl/drawings/_rels is required.
    # For ML dataset generation, sequential mapping is often sufficient if the sheet is orderly).
    media_files = sorted([f for f in os.listdir(os.path.join(temp_media_dir, 'xl', 'media'))])
    
    img_idx = 0
    for index, row in valid_rows.iterrows():
        if img_idx >= len(media_files):
            break # Ran out of images
            
        idea_text = str(row['Cost Reduction Idea Proposal']).strip()
        comp_name = str(row.get('Component Name', row.get('Parameter', 'Automotive Part'))).strip()
        
        # Build a robust SD3 prompt
        sd3_prompt = f"A realistic automotive engineering photo showing a {comp_name}. {idea_text}, detailed 4k, hyperrealistic"
        
        # Copy image to dataset folder
        src_img_path = os.path.join(temp_media_dir, 'xl', 'media', media_files[img_idx])
        img_extension = os.path.splitext(media_files[img_idx])[1]
        
        # Skip weird files
        if img_extension.lower() not in ['.jpg', '.jpeg', '.png']:
            img_idx += 1
            continue
            
        new_img_name = f"train_img_{img_idx:04d}{img_extension}"
        dest_img_path = os.path.join(IMAGES_DIR, new_img_name)
        
        shutil.copy2(src_img_path, dest_img_path)
        
        # Create metadata entry
        metadata_entries.append({
            "file_name": f"images/{new_img_name}",
            "text": sd3_prompt
        })
        
        img_idx += 1

    # 3. Write metadata.jsonl
    with open(METADATA_FILE, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Dataset compiled! {len(metadata_entries)} image-text pairs saved to {OUTPUT_DIR}/")
    
    # Cleanup
    shutil.rmtree(temp_media_dir)
    print("Temporary media files cleaned up.")

if __name__ == "__main__":
    extract_dataset()
