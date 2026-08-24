import requests
import json

# Test empty files case
url = "https://candidate-admission-api.onrender.com/quantize"
headers = {"Content-Type": "application/json"}

request_data = {
    "phase": "freeze",
    "freezeId": "test-empty-files",
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "allowedUnsupportedReasons": [],
    "candidates": [
        {
            "name": "int8",
            "files": {},  # Empty files
            "loadable": True,
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123"
        }
    ]
}

print("Testing empty files case...")
response = requests.post(url, headers=headers, json=request_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    data = response.json()
    if data["candidates"][0]["status"] == "invalid":
        print("✅ Empty files correctly marked as invalid candidate")
    else:
        print(f"❌ Expected status 'invalid', got '{data['candidates'][0]['status']}'")
else:
    print(f"❌ Expected 200, got {response.status_code}")
