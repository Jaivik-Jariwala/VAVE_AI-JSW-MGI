import zipfile
import os
import shutil

excel_path = r'd:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
output_dir = 'excel_extracted_images'

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(excel_path, 'r') as z:
    for fileinfo in z.infolist():
        if fileinfo.filename.startswith('xl/media/') and getattr(fileinfo, 'file_size', 0) > 0:
            z.extract(fileinfo, output_dir)
            
print("Extracted images to:", output_dir)
