import requests
import json

# Test the endpoint with the sample request
url = "http://localhost:8000/quantize"
headers = {"Content-Type": "application/json"}

request_data = {
    "phase": "freeze",
    "freezeId": "test-freeze-1",
    "calibrationDigest": "cal123",
    "tokenizerDigest": "tok123",
    "allowedUnsupportedReasons": [],
    "candidates": [
        {
            "name": "int8",
            "files": {
                "model.safetensors": "test content"
            },
            "loadable": True,
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123"
        }
    ]
}

try:
    response = requests.post(url, headers=headers, json=request_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
