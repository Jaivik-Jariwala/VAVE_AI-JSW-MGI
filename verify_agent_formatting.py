import sys
import os
from agent import VAVEAgent

# Mock VAVEAgent to test strictly the formatting
class MockAgent(VAVEAgent):
    def __init__(self):
        # Skip init that requires DB/Models
        self.component_knowledge = {}
        pass

def test_formatting():
    agent = MockAgent()
    
    # Create dummy ideas
    ideas = [
        {
            "Cost Reduction Idea": "Optimize Brake Caliper Material to Aluminum Alloy",
            "Idea Id": "ID-101",
            "Feasibility Score": 85,
            "Status": "Auto-Approved",
            "Saving Value (INR)": 5500000, # 55 Lakhs
            "Weight Saving (Kg)": 2.5,
            "Way Forward": "Perform CAE analysis. Check fatigue life. Validate with supplier.",
            "Material": "Al-6061",
            "Process": "Forging",
            "CAPEX": "0.5 Cr",
            "Competitor Name": "Tesla Model Y",
            "Homologation Theory": "Requires ECE R13H certification re-validation.",
            "Origin": "AI Innovation"
        },
        {
            "Cost Reduction Idea": "Remove hood insulation pad",
            "Idea Id": "ID-102",
            "Feasibility Score": 40,
            "Status": "Needs Review",
            "Saving Value (INR)": 80000,
            "Origin": "Existing DB"
        }
    ]
    
    # Normalize (simulating what happens in run)
    # We need to manually normalize since we skipped full init? 
    # Actually _format_final_response expects NORMALIZED data?
    # Let's check agent.py... _format_final_response takes 'all_data'.
    # And 'all_data' comes from 'final_results'.
    # 'final_results' comes from _normalize_data (lines 1194).
    # So yes, we should normalize first.
    
    normalized = agent._normalize_data(ideas, "Mixed")
    
    # Override 'Origin' because _normalize_data sets it to "Mixed" for all
    normalized[0]["Origin"] = "AI Innovation"
    normalized[1]["Origin"] = "Existing DB"

    print("--- Testing Corporate Card Formatting ---")
    response = agent._format_final_response(normalized, "Optimize Brakes")
    print(response)

if __name__ == "__main__":
    test_formatting()
