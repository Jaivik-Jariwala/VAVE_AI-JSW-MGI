import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: API Key not found in .env")
    exit()

genai.configure(api_key=api_key)

print(f"--- Checking Available Models for Key: {api_key[:5]}... ---")
try:
    models = genai.list_models()
    found = False
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
            found = True
    if not found:
        print("❌ No text-generation models found. Check your API Key permissions.")
except Exception as e:
    print(f"❌ Error listing models: {e}")