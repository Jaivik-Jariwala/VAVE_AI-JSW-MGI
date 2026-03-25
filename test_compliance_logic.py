import os
import sys
import logging
from agent import VAVEAgent

# Mock logger
logging.basicConfig(level=logging.INFO)

def mock_vector_db(query, top_k=5):
    return [], []

def test_compliance():
    print("--- Testing VAVE Agent Compliance Logic ---")
    
    # 1. Initialize Agent (with dummy DB path)
    agent = VAVEAgent(db_path="dummy.db", vector_db_func=mock_vector_db)
    
    # 2. Verify Knowledge Base Loading
    if not agent.component_knowledge:
        print("FAIL: Component Knowledge Base not loaded!")
        return
    print(f"SUCCESS: Loaded {len(agent.component_knowledge)} rules from KB.")
    
    # 3. Test Context Resolution
    # We built the template with "Front Brake Disc" -> so "brake rotor" or "brake disc" should trigger it
    print("\n--- Testing Context Injection ---")
    context = agent._resolve_engineering_context("Reduce cost of Front Brake Disc")
    
    if "DETAILED OEM SPECIFICATIONS" in context:
        print("SUCCESS: Context contains detailed OEM specs.")
        if "Max Temp" in context:
            print("SUCCESS: Context contains engineering constraints (Max Temp).")
    else:
        print("FAIL: Detailed specs NOT found in context.")
        print("Context snippet:", context[:500])

    # 4. Test Validation Logic
    print("\n--- Testing Validation Rejection ---")
    
    # Scenario: Safe idea
    safe_ideas = [{
        "cost_reduction_idea": "Optimize machining process for the brake disc to reduce cycle time.",
        "feasibility_score": 80,
        "cost_saving_score": 80, 
        "weight_reduction_score": 0,
        "homologation_feasibility_score": 80,
        "homologation_theory": "No impact."
    }]
    
    # Scenario: Dangerous idea (Change material of safety critical part without validation)
    dangerous_ideas = [{
        "cost_reduction_idea": "Change brake disc material to cheaper Aluminum alloy for weight save.",
        "feasibility_score": 90, # High score initially
        "cost_saving_score": 90,
        "weight_reduction_score": 90,
        "homologation_feasibility_score": 50,
        "homologation_theory": "Needs check."
    }]
    
    # We must ensure _last_component_key is set correctly for validation to work
    # _resolve_engineering_context(query) sets this state.
    agent._resolve_engineering_context("Front Brake Disc") 
    
    # Test Safe
    result_safe = agent._validate_and_filter_ideas(safe_ideas, "brake rotor", "Front Brake Disc")
    print(f"Safe Idea Status: {result_safe[0].get('validation_status') if result_safe else 'None'}")
    
    # Test Dangerous
    # "Aluminum" might be in the 'blockers' list if we added it in the Excel template test data
    # In `create_compliance_template.py` expected data was: "No Aluminum allowed due to heat"
    # So "Aluminum" should trigger a blocker rejection.
    result_danger = agent._validate_and_filter_ideas(dangerous_ideas, "brake rotor", "Front Brake Disc")
    
    if result_danger:
        status = result_danger[0].get('validation_status')
        notes = result_danger[0].get('validation_notes')
        print(f"Dangerous Idea Status: {status}")
        print(f"Dangerous Idea Notes: {notes}")
        
        if "Rejected" in status or "Safety Critical" in notes:
            print("SUCCESS: Dangerous idea was correctly flagged/rejected.")
        else:
            print("FAIL: Dangerous idea was NOT flagged.")
    else:
        print("SUCCESS: Dangerous idea was filtered out completely.")

if __name__ == "__main__":
    test_compliance()
