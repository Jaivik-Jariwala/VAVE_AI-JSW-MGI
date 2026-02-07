
import json
import os
from teardown_processor import TeardownProcessor

def ingest():
    mg_file = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Hector Plus 1.5 Gas CVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx"
    comp_file = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Seltos 1.5 Petrol HTX IVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx"
    
    proc = TeardownProcessor()
    
    # 1. Process
    mg = proc.process_file(mg_file, "Hector_Plus")
    comp = proc.process_file(comp_file, "Kia_Seltos")
    
    # 2. Compare
    comparisons = proc.compare_vehicles(mg, comp)
    
    # 3. Save Cache
    output_path = "static/teardown/brake_assembly_data.json"
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "system": "Brake Assembly",
                "mg_vehicle": "Hector Plus",
                "comp_vehicle": "Kia Seltos"
            },
            "comparisons": comparisons
        }, f, indent=2)
        
    print(f"Ingestion Complete. Saved {len(comparisons)} items to {output_path}")

if __name__ == "__main__":
    ingest()
