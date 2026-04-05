"""
test_llm_connection.py — Test LLM API Connection (OpenAI & Anthropic)
====================================================================
Validates LLM API credentials and makes a sample call to test connectivity.
Auto-detects API type (OpenAI or Anthropic) based on API_BASE_URL.
"""

import os
import json
import sys
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Get configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()

print("\n" + "=" * 70)
print("  LLM API Connection Test")
print("=" * 70)

# Check 1: Environment variables
print("\n[1] CHECKING ENVIRONMENT VARIABLES")
print("-" * 70)

if not API_BASE_URL:
    print("❌ API_BASE_URL not set")
    sys.exit(1)
else:
    print(f"✓ API_BASE_URL: {API_BASE_URL}")

if not MODEL_NAME:
    print("❌ MODEL_NAME not set")
    sys.exit(1)
else:
    print(f"✓ MODEL_NAME: {MODEL_NAME}")

api_key_to_use = HF_TOKEN or API_KEY
if not api_key_to_use:
    print("❌ Neither HF_TOKEN nor API_KEY is set")
    sys.exit(1)
else:
    print(f"✓ API Key found: {api_key_to_use[:10]}...{api_key_to_use[-4:]}")

# Check 2: Initialize client
print("\n[2] INITIALIZING CLIENT")
print("-" * 70)

# Detect which API to use
IS_ANTHROPIC = "anthropic" in API_BASE_URL.lower()

try:
    if IS_ANTHROPIC:
        client = Anthropic(api_key=api_key_to_use)
        print(f"✓ Anthropic client initialized successfully")
    else:
        client = OpenAI(api_key=api_key_to_use, base_url=API_BASE_URL)
        print(f"✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    print("\n[TRACEBACK]")
    print("-" * 70)
    traceback.print_exc()
    print("-" * 70)
    sys.exit(1)

# Check 3: Make a test API call
print("\n[3] TESTING API CALL")
print("-" * 70)

test_prompt = """Respond with ONLY a JSON object (no explanation):
{"goal": "idle", "target_slot": null}"""

try:
    print(f"Sending test request to {MODEL_NAME}...")
    
    if IS_ANTHROPIC:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": test_prompt,
                }
            ],
            timeout=10,
        )
        result = response.content[0].text
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": test_prompt,
                }
            ],
            temperature=0.7,
            max_tokens=100,
            timeout=10,
        )
        result = response.choices[0].message.content
    
    print(f"✓ API call successful!")
    print(f"\nResponse from {MODEL_NAME}:")
    print(f"  {result}")
    
    # Try to parse as JSON
    try:
        parsed = json.loads(result)
        print(f"\n✓ Response is valid JSON:")
        print(f"  {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError as je:
        print(f"\n⚠ Response is not valid JSON (may need post-processing)")
        print(f"JSON Parse Error: {je}")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    print(f"\n[TRACEBACK]")
    print("-" * 70)
    traceback.print_exc()
    print("-" * 70)
    print(f"\n[DIAGNOSTIC TIPS]")
    print(f"Common issues:")
    print(f"  1. Invalid API key - check HF_TOKEN is correct")
    print(f"  2. Wrong API endpoint - verify API_BASE_URL format")
    print(f"  3. Model not found - check MODEL_NAME is accessible to your account")
    print(f"  4. Network/timeout - check internet connection and try again")
    print(f"  5. Rate limiting - wait a moment and retry")
    sys.exit(1)

# Check 4: Summary
print("\n" + "=" * 70)
print("  ALL TESTS PASSED ✓")
print("=" * 70)
print(f"\nConfiguration verified:")
print(f"  API Endpoint: {API_BASE_URL}")
print(f"  Model: {MODEL_NAME}")
print(f"  Status: Ready for inference.py")
print("\nYou can now run:")
print("  python inference.py\n")
