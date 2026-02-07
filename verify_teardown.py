
import os
import json
from teardown_processor import TeardownProcessor

def test_teardown_parsing():
    mg_file = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Hector Plus 1.5 Gas CVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx"
    comp_file = r"d:\Internship Workspaces\JSW Morris Garages India Workspace\Deployment Project\VAVE_AI-JSW-MGI (5)\VAVE_AI-JSW-MGI\BRAKE ASSEMBLY_Seltos 1.5 Petrol HTX IVT_Scenario_Costing_Consolidated_W Image.xlsx.xlsx"
    
    processor = TeardownProcessor()
    
    print("--- Processing MG Hector ---")
    mg_parts = processor.process_file(mg_file, "Hector_Plus")
    print(f"Extracted {len(mg_parts)} MG parts.")
    if mg_parts:
        print("Sample MG Part:", json.dumps(mg_parts[5], indent=2))

    print("\n--- Processing Kia Seltos ---")
    comp_parts = processor.process_file(comp_file, "Kia_Seltos")
    print(f"Extracted {len(comp_parts)} Competitor parts.")
    
    print("\n--- Generating Comparison ---")
    pairs = processor.compare_vehicles(mg_parts, comp_parts)
    print(f"Generated {len(pairs)} comparison pairs.")
    
    # Show matched pairs with images
    matched = [p for p in pairs if p['competitor_data'] and p['mg_data']['images'] and p['competitor_data']['images']]
    print(f"Fully Matched Pairs (Data + Images): {len(matched)}")
    
    if matched:
        print("Best Match Sample:")
        print(json.dumps(matched[0], indent=2))
    else:
        print("No matches with images found. Showing any match:")
        matches_any = [p for p in pairs if p['competitor_data']]
        if matches_any:
            print(json.dumps(matches_any[0], indent=2))

if __name__ == "__main__":
    test_teardown_parsing()
