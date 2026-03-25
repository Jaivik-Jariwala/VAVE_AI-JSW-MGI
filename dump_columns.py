import pandas as pd
file_path = '20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
df = pd.read_excel(file_path)
with open('excel_columns.txt', 'w') as f:
    for c in df.columns:
        f.write(f"{c}\n")

print("Columns dumped to excel_columns.txt")
