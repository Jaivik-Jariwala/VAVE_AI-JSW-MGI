import os
import json
import time
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- CONFIGURATION ---
API_KEY = "AIzaSyBiFxUfL8EwecD7g8or-kesN9zlB0eysJk"  # <--- PUT YOUR KEY HERE
DATASET_DIR = "MG_Hector_Full_Engineering_Dataset"
OUTPUT_FILE = "MG_Hector_Master_Engineering_DB.json"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3.1-flash-lite')

def get_engineering_context(part_name):
    """
    Fetches real technical context about the part to prevent AI hallucinations.
    """
    try:
        # Search query focused on specs
        query = f"{part_name} specifications material function MG Hector"
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        snippets = [x.get_text() for x in soup.find_all('a', class_='result__snippet', limit=2)]
        return " | ".join(snippets)
    except:
        return "Standard automotive component context applied."

def analyze_image(image_path, part_name, context):
    """
    Uses Gemini Vision to label the image.
    """
    file = genai.upload_file(image_path)
    
    prompt = f"""
    You are an expert Automotive Engineer analyzing the MG Hector (Indian Spec).
    
    PART NAME: {part_name}
    WEB CONTEXT: {context}
    
    Analyze this image and return a JSON object with these EXACT keys:
    {{
        "component_name": "Precise technical name",
        "part_category": "System (e.g., Powertrain, Chassis)",
        "visual_identification": "Describe what is visible (e.g., 'Disassembled view showing piston rings')",
        "material_type": "Inferred material (e.g., Cast Iron, Polypropylene)",
        "engineering_function": "What this part does in the vehicle",
        "variant_notes": "Mention if this looks like Petrol (1.5T) or Diesel (2.0L) specific, or Hybrid (48V)"
    }}
    Output strictly valid JSON. No markdown formatting.
    """
    
    try:
        result = model.generate_content([file, prompt])
        # Strip markdown codes if Gemini adds them
        clean_json = result.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    database = []
    
    for root, dirs, files in os.walk(DATASET_DIR):
        if not files: continue
        
        # Folder name is the part name (e.g., "MG_Hector_turbocharger")
        part_name = os.path.basename(root).replace("_", " ")
        category_group = os.path.dirname(root).split(os.sep)[-1]
        
        print(f"\n🔧 Analyzing: {part_name} ({category_group})")
        
        # 1. Get Context (Once per part type to save time)
        context = get_engineering_context(part_name)
        
        # 2. Process Images
        for file in files:
            if not file.endswith('jpg'): continue
            
            img_path = os.path.join(root, file)
            data = analyze_image(img_path, part_name, context)
            
            if data:
                # Append file metadata
                data['image_id'] = file
                data['local_path'] = img_path
                data['bom_category'] = category_group
                database.append(data)
                print(f"   [OK] Processed {file}")
                
                # Rate limit protection
                time.sleep(2)
                
    # Save Final DB
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(database, f, indent=4)
    print(f"\nSUCCESS. Database saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()