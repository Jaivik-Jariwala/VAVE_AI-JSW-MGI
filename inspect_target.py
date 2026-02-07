
import pandas as pd
import os

target_file = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\AIML Dummy Ideas Data.xlsx"

print(f"Inspecting: {target_file}")
if os.path.exists(target_file):
    try:
        df = pd.read_excel(target_file, nrows=0) # Read only headers
        print("Columns Found:")
        for col in df.columns:
            print(f"- {col}")
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print("File not found.")
