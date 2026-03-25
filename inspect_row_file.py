import pandas as pd
import json

file_path = '20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Fill NAs
df['Group ID'] = df['Group ID'].fillna('')
df['Idea Id'] = df['Idea Id'].fillna('')

filtered = df[(df['Group ID'].astype(str).str.contains('1|4')) | (df['Idea Id'].astype(str).str.contains('DA|AI'))]
if not filtered.empty:
    row = filtered.iloc[0].to_dict()
else:
    row = df.iloc[0].to_dict()
    
cleaned_row = {k: str(v) for k, v in row.items() if pd.notna(v) and str(v).strip() != 'nan'}

with open('row.json', 'w') as f:
    json.dump(cleaned_row, f, indent=2)
print("Saved to row.json")
