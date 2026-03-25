import pandas as pd
import json

file_path = '20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name=0) # Read the first sheet
    print(f"Sheet Name: {pd.ExcelFile(file_path).sheet_names[0]}")
    
    # Filter for Group ID in [1, 4] or Idea Id containing DA or AI
    group_col = 'Group ID'
    idea_col = 'Idea Id'
    
    # Fill NAs
    df[group_col] = df[group_col].fillna('')
    df[idea_col] = df[idea_col].fillna('')
    
    filtered = df[(df[group_col].astype(str).str.contains('1|4')) | (df[idea_col].astype(str).str.contains('DA|AI'))]
    if not filtered.empty:
        row = filtered.iloc[0].to_dict()
    else:
        row = df.iloc[0].to_dict()
        
    cleaned_row = {k: str(v) for k, v in row.items() if pd.notna(v) and str(v).strip() != 'nan'}
    print(json.dumps(cleaned_row, indent=2))
except Exception as e:
    print(e)
