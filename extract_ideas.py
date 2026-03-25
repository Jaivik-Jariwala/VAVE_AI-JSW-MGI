import pandas as pd
import json

file_path = '20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx'
df = pd.read_excel(file_path)

# Let's find columns
idea_id_col = next((c for c in df.columns if 'idea id' in str(c).lower()), None)
group_id_col = next((c for c in df.columns if 'group' in str(c).lower()), None)
desc_col = next((c for c in df.columns if 'description' in str(c).lower() or 'idea' in str(c).lower() and 'id' not in str(c).lower()), None)
title_col = next((c for c in df.columns if 'title' in str(c).lower() or 'parameter' in str(c).lower()), None)

print(f"Using columns: idea_id={idea_id_col}, group={group_id_col}, desc={desc_col}, title={title_col}")

# Filter for relevant groups: 1, 4, and Idea IDs containing DA or AI
# Or just sample some of them to formulate the AI prompt examples.
relevant_ideas = []

def get_val(row, col):
    if col and col in row:
        return str(row[col])
    return ""

for _, row in df.iterrows():
    group = get_val(row, group_id_col)
    idea_id = get_val(row, idea_id_col)
    
    if group in ['1', '4', 1, 4] or 'DA' in idea_id or 'AI' in idea_id:
        relevant_ideas.append({
            "Idea ID": idea_id,
            "Group": group,
            "Title": get_val(row, title_col),
            "Description": get_val(row, desc_col),
            "Saving Value(INR)": get_val(row, 'Saving Value(INR)'),
            "Weight Saving(Kg)": get_val(row, 'Weight Saving(Kg)')
        })

print(f"Found {len(relevant_ideas)} relevant ideas.")

with open('extracted_ideas.json', 'w') as f:
    json.dump(relevant_ideas[:20], f, indent=4) # Save up to 20 for review

print(json.dumps(relevant_ideas[:5], indent=2))
