import pandas as pd
import os

file_path = "cost_reduction_ideas_20260203_105557.xlsx"

try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Columns:", df.columns.tolist())
    
    # Check for image columns
    img_cols = ["Current Scenario Image", "Competitor Image", "Proposal Scenario Image"]
    
    print("\n--- Image Path Analysis ---")
    for idx, row in df.head(10).iterrows():
        idea = row.get("Cost Reduction Idea", "No Idea")
        print(f"\nIdea {idx+1}: {idea[:50]}...")
        for col in img_cols:
            if col in df.columns:
                val = str(row[col])
                print(f"  {col}: {val}")
                if "ai_gen" in val:
                    print("    -> Type: AI Fallback Generation")
                elif "comp_overlay" in val:
                    print("    -> Type: Competitor Overlay")
                elif "impl_overlay" in val:
                    print("    -> Type: Engineering Annotation Overlay")
                elif "static" in val:
                    print("    -> Type: Existing Static Image")
                else:
                    print("    -> Type: Unknown/Missing")
except Exception as e:
    print(f"Error reading Excel: {e}")
