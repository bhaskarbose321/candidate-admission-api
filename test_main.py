import pytest
from fastapi.testclient import TestClient
from main import app, storage
from utils import utf8_byte_sort, utf8_byte_length, sha256_hex, compact_json

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_storage():
    """Clear storage before each test."""
    storage.clear()
    yield


class TestUtilities:
    """Test utility functions."""
    
    def test_utf8_byte_sort(self):
        """Test UTF-8 byte sorting."""
        strings = ["int8", "int4", "float16"]
        sorted_strings = utf8_byte_sort(strings)
        assert sorted_strings == ["float16", "int4", "int8"]
    
    def test_utf8_byte_length(self):
        """Test UTF-8 byte length calculation."""
        assert utf8_byte_length("hello") == 5
        assert utf8_byte_length("héllo") == 6  # é takes 2 bytes in UTF-8
    
    def test_sha256_hex(self):
        """Test SHA-256 calculation."""
        data = b"test"
        result = sha256_hex(data)
        assert len(result) == 64
        assert result == result.lower()
    
    def test_compact_json(self):
        """Test compact JSON serialization."""
        obj = {"name": "test", "value": 123}
        result = compact_json(obj)
        assert " " not in result
        assert "\n" not in result


class TestFreezePhase:
    """Test freeze phase functionality."""
    
    def test_valid_freeze(self):
        """Test valid freeze request."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-1",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test content"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["freezeId"] == "test-freeze-1"
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["name"] == "int8"
        assert data["candidates"][0]["status"] == "frozen"
    
    def test_utf8_byte_length_calculation(self):
        """Test UTF-8 byte length in file inventory."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-2",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "héllo world"},  # Contains UTF-8 character
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        # "héllo world" = 11 characters, but é takes 2 bytes, so 12 bytes total
        assert data["candidates"][0]["inventory"][0]["bytes"] == 12
    
    def test_utf8_sha256_calculation(self):
        """Test SHA-256 calculation uses UTF-8 bytes."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-3",
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
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        sha256 = data["candidates"][0]["inventory"][0]["sha256"]
        assert len(sha256) == 64
        assert sha256.islower()
    
    def test_filename_utf8_sorting(self):
        """Test filenames are sorted by UTF-8 byte ordering."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-4",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {
                        "z.model": "content1",
                        "a.model": "content2",
                        "m.model": "content3"
                    },
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        inventory = data["candidates"][0]["inventory"]
        filenames = [item["name"] for item in inventory]
        assert filenames == ["a.model", "m.model", "z.model"]
    
    def test_candidate_utf8_sorting(self):
        """Test candidates are sorted by UTF-8 byte ordering."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-5",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "content1"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int4",
                    "files": {"model.safetensors": "content2"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "float16",
                    "files": {"model.safetensors": "content3"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        candidate_names = [c["name"] for c in data["candidates"]]
        assert candidate_names == ["float16", "int4", "int8"]
    
    def test_package_digest_calculation(self):
        """Test packageDigest calculation."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-6",
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
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["packageDigest"] is not None
        assert len(data["candidates"][0]["packageDigest"]) == 64
    
    def test_supported_frozen_candidate(self):
        """Test candidate becomes frozen when supported."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-7",
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
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "frozen"
        assert data["candidates"][0]["reasonCodes"] == []
    
    def test_allowed_unsupported_reason(self):
        """Test candidate with allowed unsupported reason becomes unsupported."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-8",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": ["legacy_format"],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123",
                    "unsupportedReason": "legacy_format"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "unsupported"
        assert data["candidates"][0]["reasonCodes"] == []
    
    def test_unallowed_unsupported_reason(self):
        """Test candidate with unallowed unsupported reason becomes invalid."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-9",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": ["other_reason"],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123",
                    "unsupportedReason": "legacy_format"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "invalid"
        assert "UNALLOWED_UNSUPPORTED_REASON" in data["candidates"][0]["reasonCodes"]
    
    def test_loadable_false(self):
        """Test candidate with loadable=false becomes invalid."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-10",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": False,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "invalid"
        assert "NOT_LOADABLE" in data["candidates"][0]["reasonCodes"]
    
    def test_calibration_mismatch(self):
        """Test candidate with calibration mismatch becomes invalid."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-11",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal456",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "invalid"
        assert "CALIBRATION_MISMATCH" in data["candidates"][0]["reasonCodes"]
    
    def test_tokenizer_mismatch(self):
        """Test candidate with tokenizer mismatch becomes invalid."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-12",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok456"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["candidates"][0]["status"] == "invalid"
        assert "TOKENIZER_MISMATCH" in data["candidates"][0]["reasonCodes"]
    
    def test_invalid_candidate_files(self):
        """Test candidate with empty files returns 400 as per specification requirement."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-13",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {},  # Empty files - must be non-empty per specification
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_invalid_freeze_id(self):
        """Test invalid freezeId returns 400."""
        request = {
            "phase": "freeze",
            "freezeId": "",  # Empty freezeId
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
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_duplicate_candidate_names(self):
        """Test duplicate candidate names returns 400."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-14",
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
                },
                {
                    "name": "int8",  # Duplicate name
                    "files": {"model.safetensors": "test2"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_duplicate_allowed_reasons(self):
        """Test duplicate allowed unsupported reasons returns 400."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-15",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": ["reason1", "reason1"],  # Duplicate
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
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_empty_candidate_list(self):
        """Test empty candidate list returns 400."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-16",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": []  # Empty candidates
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_identical_freeze_replay(self):
        """Test identical freeze replay returns stored response."""
        request = {
            "phase": "freeze",
            "freezeId": "test-freeze-17",
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
        
        # First request
        response1 = client.post("/quantize", json=request)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Identical replay
        response2 = client.post("/quantize", json=request)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should return identical response
        assert data1 == data2
    
    def test_conflicting_freeze_id_reuse(self):
        """Test conflicting freezeId reuse returns 409."""
        request1 = {
            "phase": "freeze",
            "freezeId": "test-freeze-18",
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
        
        request2 = {
            "phase": "freeze",
            "freezeId": "test-freeze-18",  # Same freezeId
            "calibrationDigest": "cal456",  # Different calibration
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test2"},
                    "loadable": True,
                    "calibrationDigest": "cal456",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        
        # First request
        response1 = client.post("/quantize", json=request1)
        assert response1.status_code == 200
        
        # Conflicting reuse
        response2 = client.post("/quantize", json=request2)
        assert response2.status_code == 409
        assert response2.json() == {"error": "FREEZE_ID_CONFLICT"}


class TestSelectPhase:
    """Test select phase functionality."""
    
    def test_unknown_phase(self):
        """Test unknown phase returns 400."""
        request = {
            "phase": "unknown",
            "freezeId": "test-select-1",
            "candidates": [],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": []
            },
            "latencies": {},
            "rows": []
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_missing_phase(self):
        """Test missing phase returns 400."""
        request = {
            "freezeId": "test-select-2",
            "candidates": [],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": []
            },
            "latencies": {},
            "rows": []
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_invalid_select_policy(self):
        """Test invalid select policy returns 400."""
        request = {
            "phase": "select",
            "freezeId": "test-select-3",
            "candidates": [],
            "policy": {
                "maxBytes": -1,  # Invalid negative value
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": []
            },
            "latencies": {},
            "rows": []
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_candidate_set_mismatch(self):
        """Test candidate set mismatch returns 400."""
        request = {
            "phase": "select",
            "freezeId": "test-select-4",
            "candidates": [{"name": "int8"}],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int4"]  # Different from candidates
            },
            "latencies": {},
            "rows": []
        }
        response = client.post("/quantize", json=request)
        assert response.status_code == 400
        assert response.json() == {"error": "INVALID_INPUT"}
    
    def test_invalid_lineage(self):
        """Test invalid lineage candidate not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-5",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with different files (invalid lineage)
        select_request = {
            "phase": "select",
            "freezeId": "test-select-5",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "different"},  # Different content
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": []
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_LINEAGE" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_invalid_predictions(self):
        """Test invalid predictions not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-6",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with invalid predictions
        select_request = {
            "phase": "select",
            "freezeId": "test-select-6",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 2}  # Invalid prediction (not 0 or 1)
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_PREDICTIONS" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_missing_predictions(self):
        """Test missing predictions not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-7",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with missing predictions
        select_request = {
            "phase": "select",
            "freezeId": "test-select-7",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {}  # Missing predictions
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_PREDICTIONS" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_aggregate_floor_failure(self):
        """Test aggregate floor failure not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-8",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with low accuracy
        select_request = {
            "phase": "select",
            "freezeId": "test-select-8",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.9,  # High floor
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int8"]
            },
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 0}  # Wrong prediction
                },
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 0}  # Wrong prediction
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "AGGREGATE_FLOOR" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_missing_required_slice(self):
        """Test missing required slice not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-9",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with missing required slice
        select_request = {
            "phase": "select",
            "freezeId": "test-select-9",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {"critical": 0.75},  # Required slice
                "maxLatencyMs": 100,
                "candidateOrder": ["int8"]
            },
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "other",  # Different slice
                    "predictions": {"int8": 1}
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "MISSING_SLICE:critical" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_slice_floor_failure(self):
        """Test slice floor failure not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-10",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with low slice accuracy
        select_request = {
            "phase": "select",
            "freezeId": "test-select-10",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {"critical": 0.9},  # High slice floor
                "maxLatencyMs": 100,
                "candidateOrder": ["int8"]
            },
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 0}  # Wrong prediction
                },
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 0}  # Wrong prediction
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "SLICE_FLOOR:critical" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_size_limit(self):
        """Test size limit not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-11",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "large content here"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        client.post("/quantize", json=freeze_request)
        
        # Select with low size limit
        select_request = {
            "phase": "select",
            "freezeId": "test-select-11",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "large content here"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 5,  # Very low limit
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
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "SIZE_LIMIT" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_latency_limit(self):
        """Test latency limit not admitted."""
        # First freeze a candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-12",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with high latency
        select_request = {
            "phase": "select",
            "freezeId": "test-select-12",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 10,  # Very low latency limit
                "candidateOrder": ["int8"]
            },
            "latencies": {"int8": 100},  # High latency
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 1}
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert "LATENCY_LIMIT" in data["results"][0]["reasonCodes"]
        assert data["results"][0]["admitted"] == False
    
    def test_successful_selection(self):
        """Test successful selection with admitted candidate."""
        # First freeze candidates
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-13",
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
        client.post("/quantize", json=freeze_request)
        
        # Select successfully
        select_request = {
            "phase": "select",
            "freezeId": "test-select-13",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 1}
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "int8"
        assert data["results"][0]["admitted"] == True
        assert data["packageManifest"] is not None
    
    def test_smaller_candidate_rejected_while_larger_wins(self):
        """Test that smaller candidate rejected while larger wins due to floors."""
        # First freeze candidates
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-14",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "small"},  # Smaller
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "larger content"},  # Larger
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        client.post("/quantize", json=freeze_request)
        
        # Select where int4 fails floor but int8 passes
        select_request = {
            "phase": "select",
            "freezeId": "test-select-14",
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "small"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "larger content"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.9,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int4", "int8"]
            },
            "latencies": {"int4": 40, "int8": 60},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int4": 0, "int8": 1}  # int4 wrong, int8 right
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "int8"  # Larger wins because smaller fails floor
        assert data["results"][0]["admitted"] == False  # int4 not admitted
        assert data["results"][1]["admitted"] == True  # int8 admitted
    
    def test_tie_breaking_by_bytes(self):
        """Test tie-breaking by totalBytes."""
        # First freeze candidates
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-15",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "small"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "larger content"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        client.post("/quantize", json=freeze_request)
        
        # Select where both are admitted, smaller should win
        select_request = {
            "phase": "select",
            "freezeId": "test-select-15",
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "small"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "larger content"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int8", "int4"]  # int8 preferred in order
            },
            "latencies": {"int4": 60, "int8": 60},  # Same latency
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int4": 1, "int8": 1}  # Both correct
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "int4"  # Smaller wins despite order preference
    
    def test_tie_breaking_by_latency(self):
        """Test tie-breaking by latency."""
        # First freeze candidates with same size
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-16",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        client.post("/quantize", json=freeze_request)
        
        # Select where lower latency wins
        select_request = {
            "phase": "select",
            "freezeId": "test-select-16",
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int8", "int4"]  # int8 preferred in order
            },
            "latencies": {"int4": 40, "int8": 60},  # int4 lower latency
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int4": 1, "int8": 1}  # Both correct
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "int4"  # Lower latency wins
    
    def test_tie_breaking_by_candidate_order(self):
        """Test tie-breaking by candidate order when bytes and latency are equal."""
        # First freeze candidates with same size
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-17",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ]
        }
        client.post("/quantize", json=freeze_request)
        
        # Select where candidate order decides
        select_request = {
            "phase": "select",
            "freezeId": "test-select-17",
            "candidates": [
                {
                    "name": "int4",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                },
                {
                    "name": "int8",
                    "files": {"model.safetensors": "same size"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.8,
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int8", "int4"]  # int8 preferred
            },
            "latencies": {"int4": 50, "int8": 50},  # Same latency
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int4": 1, "int8": 1}  # Both correct
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "int8"  # Candidate order decides
    
    def test_no_admitted_candidate(self):
        """Test no admitted candidate returns null selected."""
        # First freeze candidates
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-18",
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
        client.post("/quantize", json=freeze_request)
        
        # Select where candidate fails floor
        select_request = {
            "phase": "select",
            "freezeId": "test-select-18",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
                    "calibrationDigest": "cal123",
                    "tokenizerDigest": "tok123"
                }
            ],
            "policy": {
                "maxBytes": 1000000,
                "aggregateFloor": 0.9,  # High floor
                "requiredSlices": {},
                "maxLatencyMs": 100,
                "candidateOrder": ["int8"]
            },
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 0}  # Wrong prediction
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] is None
        assert data["packageManifest"] is None
    
    def test_reason_code_sorting_deduplication(self):
        """Test reason codes are sorted and deduplicated."""
        # First freeze candidate with multiple issues
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-19",
            "calibrationDigest": "cal123",
            "tokenizerDigest": "tok123",
            "allowedUnsupportedReasons": [],
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": False,  # NOT_LOADABLE
                    "calibrationDigest": "cal456",  # CALIBRATION_MISMATCH
                    "tokenizerDigest": "tok789"  # TOKENIZER_MISMATCH
                }
            ]
        }
        response = client.post("/quantize", json=freeze_request)
        assert response.status_code == 200
        data = response.json()
        reason_codes = data["candidates"][0]["reasonCodes"]
        
        # Check deduplication
        assert len(reason_codes) == len(set(reason_codes))
        
        # Check UTF-8 byte sorting
        assert reason_codes == utf8_byte_sort(reason_codes)
    
    def test_recomputation_of_manifest(self):
        """Test manifest is recomputed, not trusted from client."""
        # First freeze candidate
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-20",
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
        client.post("/quantize", json=freeze_request)
        
        # Select with different files but try to spoof manifest
        select_request = {
            "phase": "select",
            "freezeId": "test-select-20",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "different content"},
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 1}
                }
            ]
        }
        response = client.post("/quantize", json=select_request)
        assert response.status_code == 200
        data = response.json()
        # Should detect lineage mismatch due to manifest recomputation
        assert "INVALID_LINEAGE" in data["results"][0]["reasonCodes"]
    
    def test_replay_determinism(self):
        """Test replay produces deterministic results."""
        # First freeze
        freeze_request = {
            "phase": "freeze",
            "freezeId": "test-select-21",
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
        client.post("/quantize", json=freeze_request)
        
        # First select
        select_request = {
            "phase": "select",
            "freezeId": "test-select-21",
            "candidates": [
                {
                    "name": "int8",
                    "files": {"model.safetensors": "test"},
                    "loadable": True,
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
            "latencies": {"int8": 50},
            "rows": [
                {
                    "label": 1,
                    "slice": "critical",
                    "predictions": {"int8": 1}
                }
            ]
        }
        response1 = client.post("/quantize", json=select_request)
        data1 = response1.json()
        
        # Identical replay
        response2 = client.post("/quantize", json=select_request)
        data2 = response2.json()
        
        # Should be identical
        assert data1 == data2
