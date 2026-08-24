import requests
import json

# Replace this with your actual request that's failing
your_request = {
    # Paste your failing request here
}

url = "http://localhost:8000/quantize"  # or your deployed URL
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, headers=headers, json=your_request)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 400:
        print("\nTroubleshooting tips:")
        print("1. Check that all required fields are present")
        print("2. Ensure Content-Type header is 'application/json'")
        print("3. Verify data types match expectations")
        print("4. Check that freezeId is non-empty and <= 128 characters")
        print("5. Ensure candidates array is non-empty")
        
except Exception as e:
    print(f"Error: {e}")
