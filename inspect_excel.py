
import pandas as pd
import os

files = [
    r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Hector Plus 1.5 Gas CVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx",
    r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Seltos 1.5 Petrol HTX IVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx"
]

output_log = "excel_structure.txt"
with open(output_log, "w", encoding="utf-8") as log:
    for f in files:
        log.write(f"\n{'='*50}\n")
        log.write(f"File: {os.path.basename(f)}\n")
        if not os.path.exists(f):
            log.write("FILE NOT FOUND\n")
            continue
            
        try:
            df = pd.read_excel(f, nrows=5)
            log.write("Columns List:\n")
            for i, c in enumerate(df.columns):
                log.write(f"  {i}: {c}\n")
            
            log.write("\nChecking Image Column Content (Rows 1-5):\n")
            # We know the real header is at row 1 (0-indexed in excel, so skip 1 row)
            df_data = pd.read_excel(f, header=1, nrows=5)
            log.write(str(df_data[['Top View', 'Front View']].head().to_dict()) + "\n")
            
            # Check for actual images using OpenPyXL
            try:
                from openpyxl import load_workbook
                wb = load_workbook(f)
                ws = wb.active
                log.write(f"\nOpenPyXL Image Check: {len(ws._images)} embedded images found.\n")
            except Exception as e:
                log.write(f"\nOpenPyXL check failed: {e}\n")
                
        except Exception as e:
            log.write(f"Error reading excel: {e}\n")

print(f"Inspection complete. Written to {output_log}")
