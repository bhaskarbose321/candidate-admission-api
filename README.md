# Candidate Admission API

A production-ready FastAPI service implementing a stateful two-phase candidate-admission API. This service provides quantization candidate management with freeze and select phases, supporting deterministic operations, state persistence, and comprehensive validation.

## API Endpoint

`POST /quantize`

The API accepts JSON request bodies and returns JSON responses. The endpoint supports two phases:

- **freeze phase**: Validate and store candidate information
- **select phase**: Select the best candidate based on policy constraints

## Freeze Phase

### Request Example

```json
{
  "phase": "freeze",
  "freezeId": "example-freeze-123",
  "calibrationDigest": "cal-abc123",
  "tokenizerDigest": "tok-xyz789",
  "allowedUnsupportedReasons": ["legacy_format"],
  "candidates": [
    {
      "name": "int8",
      "files": {
        "model.safetensors": "model file content"
      },
      "loadable": true,
      "calibrationDigest": "cal-abc123",
      "tokenizerDigest": "tok-xyz789"
    }
  ]
}
```

### Response Example

```json
{
  "freezeId": "example-freeze-123",
  "candidates": [
    {
      "name": "int8",
      "status": "frozen",
      "inventory": [
        {
          "name": "model.safetensors",
          "bytes": 17,
          "sha256": "a1b2c3d4e5f6..."
        }
      ],
      "totalBytes": 17,
      "packageDigest": "f6e5d4c3b2a1...",
      "reasonCodes": []
    }
  ]
}
```

## Select Phase

### Request Example

```json
{
  "phase": "select",
  "freezeId": "example-freeze-123",
  "candidates": [
    {
      "name": "int8",
      "files": {
        "model.safetensors": "model file content"
      },
      "loadable": true,
      "calibrationDigest": "cal-abc123",
      "tokenizerDigest": "tok-xyz789"
    }
  ],
  "policy": {
    "maxBytes": 1000000,
    "aggregateFloor": 0.8,
    "requiredSlices": {
      "critical": 0.75
    },
    "maxLatencyMs": 100,
    "candidateOrder": ["int8", "int4"]
  },
  "latencies": {
    "int8": 60
  },
  "rows": [
    {
      "label": 1,
      "slice": "critical",
      "predictions": {
        "int8": 1
      }
    }
  ]
}
```

### Response Example

```json
{
  "freezeId": "example-freeze-123",
  "selected": "int8",
  "results": [
    {
      "name": "int8",
      "aggregate": 1.0,
      "slices": {
        "critical": 1.0
      },
      "totalBytes": 17,
      "latencyMs": 60,
      "admitted": true,
      "reasonCodes": []
    }
  ],
  "packageManifest": {
    "name": "int8",
    "inventory": [
      {
        "name": "model.safetensors",
        "bytes": 17,
        "sha256": "a1b2c3d4e5f6..."
      }
    ],
    "totalBytes": 17,
    "packageDigest": "f6e5d4c3b2a1..."
  }
}
```

## Local Setup

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bhaskarbose321/candidate-admission-api.git
cd candidate-admission-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Service

Start the server locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### Testing

Run the comprehensive test suite:
```bash
pytest test_main.py -v
```

The test suite includes 40+ test cases covering:
- Valid freeze operations
- UTF-8 byte length and SHA-256 calculations
- File inventory calculations
- Candidate status determination
- Input validation
- State persistence and conflict detection
- Selection logic and winner determination
- Accuracy calculations
- Policy validation
- Edge cases and error handling

## API Testing with cURL

### Freeze Request

```bash
curl -X POST http://localhost:8000/quantize \
  -H "Content-Type: application/json" \
  -d '{
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
        "loadable": true,
        "calibrationDigest": "cal123",
        "tokenizerDigest": "tok123"
      }
    ]
  }'
```

### Select Request

```bash
curl -X POST http://localhost:8000/quantize \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "select",
    "freezeId": "test-freeze-1",
    "candidates": [
      {
        "name": "int8",
        "files": {
          "model.safetensors": "test content"
        },
        "loadable": true,
        "calibrationDigest": "cal123",
        "tokenizerDigest": "tok123"
      }
    ],
    "policy": {
      "maxBytes": 1000000,
      "aggregateFloor": 0.8,
      "requiredSlices": {},
      "maxLatencyMs": 100,
      "candidateOrder": ["int8"]
    },
    "latencies": {
      "int8": 50
    },
    "rows": [
      {
        "label": 1,
        "slice": "critical",
        "predictions": {
          "int8": 1
        }
      }
    ]
  }'
```

## Render Deployment

### Prerequisites

- A Render account
- The repository pushed to GitHub

### Deployment Steps

1. Push the repository to GitHub (if not already done)
2. Log in to [Render](https://render.com)
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Name**: candidate-admission-api (or your preferred name)
   - **Region**: Select your preferred region
   - **Branch**: main
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

### Alternative: Render Blueprint

You can also deploy using the included `render.yaml` blueprint:

1. Go to Render Dashboard
2. Click "New +" and select "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect and use the `render.yaml` configuration

### Environment Variables

The service uses the `PORT` environment variable which Render automatically provides. The default is set to 8000 in the configuration.

### Endpoint URL After Deployment

After deployment, your API will be available at:

```
https://<your-service-name>.onrender.com/quantize
```

Replace `<your-service-name>` with the actual name you chose during deployment.

For example, if you named your service `candidate-admission-api`, the endpoint would be:

```
https://candidate-admission-api.onrender.com/quantize
```

## Key Features

- **Stateful Operations**: Maintains state between requests using in-memory storage
- **Deterministic**: Produces consistent results for identical inputs
- **UTF-8 Byte Ordering**: Uses explicit UTF-8 byte ordering for sorting and comparisons
- **SHA-256 Hashing**: Calculates exact SHA-256 hashes using UTF-8 byte encoding
- **Input Validation**: Comprehensive validation for all request fields
- **Conflict Detection**: Detects and prevents conflicting freezeId reuse
- **Lineage Verification**: Recomputes manifests instead of trusting client-submitted values
- **Accuracy Calculations**: Rounds to 12 decimal places for precision
- **Winner Selection**: Multi-criteria selection (bytes, latency, candidate order)

## Error Responses

The API returns appropriate HTTP status codes and JSON error responses:

- **400 Bad Request**: Invalid input structure
```json
{"error": "INVALID_INPUT"}
```

- **409 Conflict**: Conflicting freezeId reuse
```json
{"error": "FREEZE_ID_CONFLICT"}
```

## Architecture

- **main.py**: FastAPI application and endpoint handlers
- **models.py**: Pydantic data models for requests/responses
- **storage.py**: In-memory state storage with freezeId persistence
- **utils.py**: Utility functions (UTF-8 sorting, SHA-256, JSON serialization)
- **test_main.py**: Comprehensive test suite

## Implementation Notes

- Uses in-memory storage for state persistence (survives within the running service)
- Does not use external databases (as per requirements)
- Does not implement authentication (as per requirements)
- Follows exact specification for deterministic behavior
- All sorting uses UTF-8 byte ordering
- All hashing uses UTF-8 byte encoding
- Never trusts client-submitted derived values

## License

This project is provided as-is for educational and production use.
