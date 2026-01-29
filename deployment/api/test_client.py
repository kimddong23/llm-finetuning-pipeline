"""
Simple test client for the Code Generation API.
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed:")
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_generation(prompt="def fibonacci(n):", max_tokens=100):
    """Test code generation endpoint."""
    print(f"\nTesting /generate endpoint...")
    print(f"Prompt: {prompt}")

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    start = time.time()
    response = requests.post(f"{API_URL}/generate", json=payload)
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generation successful:")
        print(f"   Generation time: {data['generation_time']:.2f}s (API reported)")
        print(f"   Total time: {elapsed:.2f}s (including network)")
        print(f"   Tokens generated: {data['tokens_generated']}")
        print(f"\n   Generated code:")
        print(f"   {'-' * 60}")
        for i, completion in enumerate(data['completions'], 1):
            print(f"   Completion {i}:")
            print(f"   {prompt}{completion}")
            print(f"   {'-' * 60}")
        return True
    else:
        print(f"❌ Generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_multiple_completions():
    """Test generating multiple completions."""
    print(f"\nTesting multiple completions...")
    prompt = "def is_prime(n):"

    payload = {
        "prompt": prompt,
        "max_tokens": 80,
        "temperature": 0.7,  # Higher temperature for diversity
        "num_return_sequences": 3
    }

    response = requests.post(f"{API_URL}/generate", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {len(data['completions'])} completions:")
        for i, completion in enumerate(data['completions'], 1):
            print(f"\n   Completion {i}:")
            print(f"   {prompt}{completion[:100]}...")
        return True
    else:
        print(f"❌ Multiple completions failed: {response.status_code}")
        return False

def main():
    print("="*60)
    print("CODE GENERATION API TEST CLIENT")
    print("="*60)
    print(f"API URL: {API_URL}")
    print()

    # Test sequence
    tests = [
        ("Health Check", test_health),
        ("Basic Generation", lambda: test_generation("def fibonacci(n):", 100)),
        ("Palindrome Function", lambda: test_generation("def is_palindrome(s):", 80)),
        ("Multiple Completions", test_multiple_completions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error: Is the server running at {API_URL}?")
            results.append((test_name, False))
            break
        except Exception as e:
            print(f"❌ Test error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
