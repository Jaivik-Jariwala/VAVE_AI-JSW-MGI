import pandas as pd
import sys

file_path = '20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
df = pd.read_excel(file_path)
print("Shape:", df.shape)
print("Columns:")
for c in df.columns:
    print(f"- {c}")

print("\nSample Data for relevant columns:")
# let's try to match something roughly like group id and idea id
cols_to_print = [c for c in df.columns if 'idea id' in str(c).lower() or 'group' in str(c).lower() or 'description' in str(c).lower() or 'title' in str(c).lower()]
if not cols_to_print:
    cols_to_print = list(df.columns)[:5]

print(df[cols_to_print].head())
