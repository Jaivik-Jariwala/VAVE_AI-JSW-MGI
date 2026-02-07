import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

print("Importing modules...")
try:
    import llm_cache
    print("llm_cache imported.")
    
    import agent
    print("agent imported.")
    
    import vlm_engine
    print("vlm_engine imported.")
    
    print("SUCCESS: All modules imported correctly with new caching logic.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
