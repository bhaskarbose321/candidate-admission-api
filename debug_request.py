import requests
import json

# Test different request variations to identify the issue
url = "http://localhost:8000/quantize"
headers = {"Content-Type": "application/json"}

# Test 1: Basic freeze request
print("Test 1: Basic freeze request")
request1 = {
    "phase": "freeze",
    "freezeId": "test-1",
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "allowedUnsupportedReasons": [],
    "candidates": [
        {
            "name": "int8",
            "files": {"model.safetensors": "test"},
            "loadable": True,
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123"
        }
    ]
}

try:
    response = requests.post(url, headers=headers, json=request1)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Missing field
print("Test 2: Missing phase field")
request2 = {
    "freezeId": "test-2",
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "candidates": []
}

try:
    response = requests.post(url, headers=headers, json=request2)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Invalid phase
print("Test 3: Invalid phase")
request3 = {
    "phase": "invalid",
    "freezeId": "test-3",
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "candidates": []
}

try:
    response = requests.post(url, headers=headers, json=request3)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
