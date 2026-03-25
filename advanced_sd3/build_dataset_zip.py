import pandas as pd
import zipfile
import json
import os

excel_path = r'd:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
zip_path = r'd:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\advanced_sd3\dataset.zip'
output_dir = 'masactrl_dataset'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

# 1. Extract zip images
zip_images = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for fileinfo in z.infolist():
        if fileinfo.filename.endswith(('.jpg', '.jpeg', '.png')):
            zip_images.append(fileinfo.filename)
            z.extract(fileinfo, output_dir)
zip_images = sorted(zip_images)

# 2. Read Excel valid rows
df = pd.read_excel(excel_path)
records = []
for row_idx, row in df.iterrows():
    prompt = str(row.get('Cost Reduction Idea Proposal', ''))
    if prompt.strip() == 'nan' or prompt.strip() == '':
        continue
    idea_id = str(row.get('Idea Id', f'Idea_{row_idx}'))
    records.append({
        'idea_id': idea_id,
        'prompt': prompt,
        'row_idx': row_idx
    })

# print counts
print(f"Found {len(zip_images)} images in zip.")
print(f"Found {len(records)} valid text prompts in Excel.")

# Map together sequentially (assuming order matches)
mapping = []
limit = min(len(zip_images), len(records))
for i in range(limit):
    img_path = zip_images[i]
    rec = records[i]
    # Keep flat path in 'images/'
    flat_img_name = os.path.basename(img_path)
    # the extract already put it where it belongs basically, but let's just save metadata
    mapping.append({
        'idea_id': rec['idea_id'],
        'image_path': img_path,  # this contains 'images/train_img_00...jpeg'
        'prompt': rec['prompt'],
        'row_idx': rec['row_idx']
    })

with open(os.path.join(output_dir, 'metadata.jsonl'), 'w') as f:
    for item in mapping:
        f.write(json.dumps(item) + '\n')

print(f"Created paired metadata for {len(mapping)} examples in {output_dir}/metadata.jsonl")
