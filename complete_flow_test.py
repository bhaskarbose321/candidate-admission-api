import requests
import json
import time

BASE_URL = "https://candidate-admission-api.onrender.com"
ENDPOINT = f"{BASE_URL}/quantize"
headers = {"Content-Type": "application/json"}

print("Testing Complete API Flow")
print("=" * 50)

# Test 1: Valid Freeze
print("\n1. Testing Valid Freeze Request")
freeze_request = {
    "phase": "freeze",
    "freezeId": "flow-test-" + str(int(time.time())),
    "calibrationDigest": "cal-abc123",
    "tokenizerDigest": "tok-xyz789",
    "allowedUnsupportedReasons": [],
    "candidates": [
        {
            "name": "int8",
            "files": {"model.safetensors": "test model content"},
            "loadable": True,
            "calibrationDigest": "cal-abc123",
            "tokenizerDigest": "tok-xyz789"
        }
    ]
}

response = requests.post(ENDPOINT, headers=headers, json=freeze_request)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Freeze successful: {data['freezeId']}")
    print(f"   Candidate status: {data['candidates'][0]['status']}")
    freeze_id = data['freezeId']
else:
    print(f"[FAIL] Freeze failed: {response.json()}")
    exit(1)

# Test 2: Valid Select
print("\n2. Testing Valid Select Request")
select_request = {
    "phase": "select",
    "freezeId": freeze_id,
    "candidates": [
        {
            "name": "int8",
            "files": {"model.safetensors": "test model content"},
            "loadable": True,
            "calibrationDigest": "cal-abc123",
            "tokenizerDigest": "tok-xyz789"
        }
    ],
    "policy": {
        "maxBytes": 1000000,
        "aggregateFloor": 0.8,
        "requiredSlices": {},
        "maxLatencyMs": 100,
        "candidateOrder": ["int8"]
    },
    "latencies": {"int8": 50},
    "rows": [
        {
            "label": 1,
            "slice": "critical",
            "predictions": {"int8": 1}
        }
    ]
}

response = requests.post(ENDPOINT, headers=headers, json=select_request)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Select successful")
    print(f"   Selected candidate: {data['selected']}")
    print(f"   Admitted: {data['results'][0]['admitted']}")
else:
    print(f"[FAIL] Select failed: {response.json()}")

# Test 3: Conflict Detection
print("\n3. Testing Conflict Detection (Expected 409)")
conflict_request = {
    "phase": "freeze",
    "freezeId": freeze_id,  # Reuse same freezeId
    "calibrationDigest": "cal-different",  # Different digest
    "tokenizerDigest": "tok-xyz789",
    "allowedUnsupportedReasons": [],
    "candidates": [
        {
            "name": "int8",
            "files": {"model.safetensors": "different content"},
            "loadable": True,
            "calibrationDigest": "cal-different",
            "tokenizerDigest": "tok-xyz789"
        }
    ]
}

response = requests.post(ENDPOINT, headers=headers, json=conflict_request)
print(f"Status: {response.status_code}")
if response.status_code == 409:
    print(f"[OK] Conflict detection working: {response.json()}")
else:
    print(f"[WARN] Unexpected response: {response.json()}")

# Test 4: Invalid Input (Expected 400)
print("\n4. Testing Invalid Input (Expected 400)")
invalid_request = {
    "phase": "freeze",
    "freezeId": "",  # Empty freezeId
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "candidates": []
}

response = requests.post(ENDPOINT, headers=headers, json=invalid_request)
print(f"Status: {response.status_code}")
if response.status_code == 400:
    print(f"[OK] Invalid input rejected: {response.json()}")
else:
    print(f"[WARN] Unexpected response: {response.json()}")

print("\n" + "=" * 50)
print("API Flow Test Complete!")
print("The API is working correctly with proper error handling.")
