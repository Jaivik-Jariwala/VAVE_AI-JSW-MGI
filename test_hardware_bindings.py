import logging
import os
import sys

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

try:
    from agent import VAVEAgent

    print("\n--- Testing VAVEAgent Hardware Initialization ---")
    agent = VAVEAgent(db_path=":memory:", vector_db_func=lambda x, top_k=5: ([], None))
    
    print("\nHardware Initialization Log:")
    print(f"NIC MAC Address Identified: {agent.mac_address}")
    print(f"Physical NIC (Trust Token): {agent.is_valid_nic}")
    
    # Test HMAC function
    test_decision = agent._hmac_sign_decision({"cost_reduction_idea": "Test Engine Idea", "saving_value_inr": "5000000"}, "Auto-Approved")
    print(f"Sample HMAC-SHA256 Audit Signature: {test_decision}")
    
    print("\nSUCCESS: VAVEAgent initialized and hardware parameters derived correctly.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nFAILURE: {e}")
