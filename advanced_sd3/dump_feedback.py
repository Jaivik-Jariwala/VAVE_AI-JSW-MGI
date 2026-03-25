import pandas as pd
import sys
import os

files = [
    r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\Human_Align_correction_feedback\Door_Assembly.xlsx",
    r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\Human_Align_correction_feedback\exterior of vehicle and body panel.xlsx",
    r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI - Copy\VAVE_AI-JSW-MGI - Copy\Human_Align_correction_feedback\Nishant sir - Feedback R1 - interior and seats.xlsx"
]

output_file = "feedback_dump.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for f in files:
        out.write(f"\n--- {os.path.basename(f)} ---\n")
        try:
            df = pd.read_excel(f)
            feedback_cols = [c for c in df.columns if 'Feedback' in str(c) or 'feedback' in str(c).lower()]
            if not feedback_cols:
                out.write("No feedback column found.\n")
                continue
                
            feedback_col = feedback_cols[-1] # take the last one, usually the most specific
            idea_col = "Cost Reduction Idea" if "Cost Reduction Idea" in df.columns else "Cost reduction idea"
            if idea_col not in df.columns:
                idea_cols = [c for c in df.columns if 'idea' in str(c).lower()]
                if idea_cols: idea_col = idea_cols[0]
            
            for _, row in df.dropna(subset=[feedback_col]).iterrows():
                feedback = str(row[feedback_col]).strip()
                if feedback and feedback.lower() not in ['nan', 'none']:
                    idea = str(row[idea_col]).strip() if idea_col in df.columns else "N/A"
                    out.write(f"Idea: {idea}\n")
                    out.write(f"Feedback: {feedback}\n\n")
        except Exception as e:
            out.write(f"Error reading file: {e}\n")

print(f"Feedback dumped to {output_file}")
