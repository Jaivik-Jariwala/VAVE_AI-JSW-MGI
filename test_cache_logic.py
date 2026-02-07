import llm_cache
import time
import os

def test_cache_basic():
    print("Testing Basic Cache Logic...")
    
    prompt = "Test Prompt " + str(time.time())
    sys_prompt = "System Prompt"
    model = "test-model"
    response_content = "Cached Response Content"
    
    # 1. Check miss
    print(f"Checking miss for: {prompt}")
    res = llm_cache.get_cached_response(prompt, sys_prompt, model)
    assert res is None, "Should be None on first try"
    
    # 2. Write
    print("Writing to cache...")
    llm_cache.cache_response(prompt, sys_prompt, model, response_content)
    
    # 3. Check hit
    print("Checking hit...")
    res = llm_cache.get_cached_response(prompt, sys_prompt, model)
    assert res == response_content, f"Expected {response_content}, got {res}"
    
    print("Basic Cache Test PASSED")

if __name__ == "__main__":
    test_cache_basic()
