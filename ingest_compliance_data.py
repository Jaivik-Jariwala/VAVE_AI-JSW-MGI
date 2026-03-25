import pandas as pd
import json
import os
from pathlib import Path

def ingest_compliance_data(excel_path="compliance_data_template.xlsx", output_path="static/knowledge/component_knowledge.json"):
    """
    Reads the detailed engineering Excel and converts it into a structured Knowledge Base JSON.
    This JSON is then loaded by agent.py to "ground" the AI in real engineering data.
    """
    
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # Clean header names (strip spaces, lowercase for internal keys)
    df.columns = [str(c).strip() for c in df.columns]

    knowledge_base = {}

    for _, row in df.iterrows():
        # Skip empty rows or example rows if marked
        if pd.isna(row.get("Part Name")) or str(row.get("Part Name")).startswith("Example"):
            continue

        # 1. Create a Key for the Part (System + Part Name)
        # e.g. "braking_brake_rotor_front"
        part_key = f"{str(row.get('Sub-System', 'general')).lower().replace(' ', '_')}_{str(row.get('Part Name', 'unknown')).lower().replace(' ', '_')}"
        
        # 2. Build the Engineering Data Dict
        entry = {
            "meta": {
                "system": row.get("System Group"),
                "sub_system": row.get("Sub-System"),
                "part_name": row.get("Part Name"),
                "part_number": row.get("Part Number"),
                "vehicle": row.get("Vehicle Model")
            },
            "current_spec": {
                "material": row.get("MG Material Grade"),
                "process": row.get("MG Manufacturing Process"),
                "surface_finish": row.get("MG Surface Treatment"),
                "weight_kg": row.get("MG Weight (kg)"),
                "cost_inr": row.get("MG Cost (INR)"),
                "supplier": row.get("MG Supplier Name"),
                "sourcing": row.get("MG Sourcing")
            },
            "benchmark_spec": {
                "model": row.get("Competitor Model"),
                "material": row.get("Competitor Material"),
                "process": row.get("Competitor Process"),
                "weight_kg": row.get("Competitor Weight (kg)"),
                "cost_inr": row.get("Competitor Cost (INR)"),
                "feature_diff": row.get("Competitor Feature Diff")
            },
            "compliance": {
                "safety_critical": str(row.get("Safety Critical Item?", "NO")).upper().strip() == "YES",
                "standard": row.get("Regulatory Standard (AIS/FMVSS)"),
                "homologation_required": str(row.get("Homologation Impact?", "NO")).upper().strip() == "YES",
                "nvh_critical": str(row.get("NVH Critical?", "NO")).upper().strip() == "YES",
                "durability_target": row.get("Durability Target (cycles)")
            },
            "constraints": {
                "max_temp_c": row.get("Max Operating Temp (C)"),
                "min_yield_mpa": row.get("Min Yield Strength (MPa)"),
                "corrosion_hours": row.get("Corrosion Requirement (Hours)"),
                "tolerance_mm": row.get("Dimensional Tolerance (mm)"),
                "non_negotiables": row.get("Design Non-Negotiables"),
                "material_blockers": row.get("Material Constraints")
            },
            "vave_potential": {
                "approved_alternatives": row.get("Approved Alt Materials"),
                "save_weight_est": row.get("Potential Weight Save (Est)"),
                "save_cost_est": row.get("Potential Cost Save (Est)")
            }
        }
        
        # Add to KB
        knowledge_base[part_key] = entry
        
        # Also add a simple lookup by "Part Name" alone for easier fuzzy matching
        simple_key = str(row.get("Part Name", "")).lower().strip()
        if simple_key and simple_key not in knowledge_base:
             knowledge_base[simple_key] = entry

    # Save JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, "w") as f:
        json.dump(knowledge_base, f, indent=2)
    
    print(f"Success! Ingested {len(knowledge_base)} items into {output_path}")

if __name__ == "__main__":
    ingest_compliance_data()
