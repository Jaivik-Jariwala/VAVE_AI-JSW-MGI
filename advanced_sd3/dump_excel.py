import pandas as pd
import json
import os

excel_path = r'd:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'

df = pd.read_excel(excel_path)
columns = df.columns.tolist()

# Get first 3 rows of relevant columns if they exist
sample_data = df.head(3).to_dict(orient='records')

out = {
    'columns': columns,
    'sample_data': sample_data
}

with open('excel_metadata.json', 'w') as f:
    json.dump(out, f, indent=4)
