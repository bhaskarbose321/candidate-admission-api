from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Set
import os

from models import (
    FreezeRequest, FreezeResponse, FreezeCandidateResult, InventoryEntry,
    SelectRequest, SelectResponse, SliceResult, ErrorResponse
)
from storage import storage
from utils import (
    utf8_byte_sort, utf8_byte_length, sha256_hex, compact_json,
    canonical_json, sort_dict_by_keys, round_to_decimal_places
)

app = FastAPI()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/quantize")
async def quantize(request: Request):
    """Handle both freeze and select phases."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"}
        )
    
    # Ensure body is a dictionary
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"}
        )
    
    phase = body.get("phase")
    
    if phase == "freeze":
        return await handle_freeze(body)
    elif phase == "select":
        return await handle_select(body)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"}
        )


async def handle_freeze(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle freeze phase."""
    # Validate input structure
    validation_error = validate_freeze_input(body)
    if validation_error:
        return JSONResponse(
            status_code=400,
            content={"error": validation_error}
        )
    
    freeze_id = body["freezeId"]
    
    # Check for conflicting reuse
    if storage.freeze_id_exists(freeze_id):
        if not storage.is_same_request(freeze_id, body):
            return JSONResponse(
                status_code=409,
                content={"error": "FREEZE_ID_CONFLICT"}
            )
        else:
            # Return stored response for identical replay
            return storage.get_freeze_response(freeze_id)
    
    # Process candidates
    candidates_result = []
    for candidate in body["candidates"]:
        candidate_result = process_freeze_candidate(
            candidate, 
            body["calibrationDigest"],
            body["tokenizerDigest"],
            body["allowedUnsupportedReasons"]
        )
        candidates_result.append(candidate_result)
    
    # Sort candidates by name using UTF-8 byte ordering
    candidates_result.sort(key=lambda c: utf8_byte_sort([c["name"]])[0] if c["name"] else "")
    
    response = {
        "freezeId": freeze_id,
        "candidates": candidates_result
    }
    
    # Store the response
    storage.store_freeze(freeze_id, body, response)
    
    return response


def validate_freeze_input(body: Dict[str, Any]) -> Optional[str]:
    """Validate freeze request input."""
    # Ensure body is not None
    if body is None:
        return "INVALID_INPUT"
    
    # Check required fields
    required_fields = ["phase", "freezeId", "calibrationDigest", "tokenizerDigest", "candidates"]
    for field in required_fields:
        if field not in body:
            return "INVALID_INPUT"
    
    # Validate freezeId
    freeze_id = body["freezeId"]
    if not isinstance(freeze_id, str) or not freeze_id or len(freeze_id) > 128:
        return "INVALID_INPUT"
    
    # Validate calibrationDigest
    if not isinstance(body["calibrationDigest"], str) or not body["calibrationDigest"]:
        return "INVALID_INPUT"
    
    # Validate tokenizerDigest
    if not isinstance(body["tokenizerDigest"], str) or not body["tokenizerDigest"]:
        return "INVALID_INPUT"
    
    # Validate allowedUnsupportedReasons
    allowed_reasons = body.get("allowedUnsupportedReasons", [])
    if not isinstance(allowed_reasons, list):
        return "INVALID_INPUT"
    for reason in allowed_reasons:
        if not isinstance(reason, str) or not reason:
            return "INVALID_INPUT"
    # Check uniqueness
    if len(set(allowed_reasons)) != len(allowed_reasons):
        return "INVALID_INPUT"
    
    # Validate candidates
    candidates = body["candidates"]
    if not isinstance(candidates, list) or not candidates:
        return "INVALID_INPUT"
    
    # Check candidate name uniqueness
    candidate_names = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return "INVALID_INPUT"
        if "name" not in candidate or not isinstance(candidate["name"], str) or not candidate["name"]:
            return "INVALID_INPUT"
        candidate_names.append(candidate["name"])
    
    if len(set(candidate_names)) != len(candidate_names):
        return "INVALID_INPUT"
    
    # Validate each candidate structure
    for candidate in candidates:
        required_candidate_fields = ["name", "files", "loadable", "calibrationDigest", "tokenizerDigest"]
        for field in required_candidate_fields:
            if field not in candidate:
                return "INVALID_INPUT"
        
        # Validate files
        files = candidate["files"]
        if not isinstance(files, dict):
            return "INVALID_INPUT"
        
        # If files is empty, allow it but will result in invalid candidate (per specification)
        if files:
            # Check filename uniqueness
            filenames = list(files.keys())
            if len(set(filenames)) != len(filenames):
                return "INVALID_INPUT"
            
            # Validate file contents are strings
            for filename, content in files.items():
                if not isinstance(content, str):
                    return "INVALID_INPUT"
        
        # Validate loadable
        if not isinstance(candidate["loadable"], bool):
            return "INVALID_INPUT"
        
        # Validate digests
        if not isinstance(candidate["calibrationDigest"], str) or not candidate["calibrationDigest"]:
            return "INVALID_INPUT"
        if not isinstance(candidate["tokenizerDigest"], str) or not candidate["tokenizerDigest"]:
            return "INVALID_INPUT"
        
        # unsupportedReason is optional
        if "unsupportedReason" in candidate:
            if not isinstance(candidate["unsupportedReason"], str):
                return "INVALID_INPUT"
    
    return None


def process_freeze_candidate(
    candidate: Dict[str, Any],
    request_calibration_digest: str,
    request_tokenizer_digest: str,
    allowed_unsupported_reasons: List[str]
) -> Dict[str, Any]:
    """Process a single freeze candidate."""
    name = candidate["name"]
    files = candidate["files"]
    loadable = candidate["loadable"]
    candidate_calibration_digest = candidate["calibrationDigest"]
    candidate_tokenizer_digest = candidate["tokenizerDigest"]
    unsupported_reason = candidate.get("unsupportedReason")
    
    reason_codes = []
    
    # Calculate inventory
    inventory_result = calculate_inventory(files)
    
    if inventory_result["inventory"] == []:
        # Invalid files
        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": []
        }
    
    inventory = inventory_result["inventory"]
    total_bytes = inventory_result["totalBytes"]
    package_digest = inventory_result["packageDigest"]
    
    # Check if unsupported
    if unsupported_reason is not None:
        if unsupported_reason in allowed_unsupported_reasons:
            status = "unsupported"
        else:
            status = "invalid"
            reason_codes.append("UNALLOWED_UNSUPPORTED_REASON")
    else:
        # Check if can be frozen
        candidate_reason_codes = []
        
        if not loadable:
            candidate_reason_codes.append("NOT_LOADABLE")
        
        if candidate_calibration_digest != request_calibration_digest:
            candidate_reason_codes.append("CALIBRATION_MISMATCH")
        
        if candidate_tokenizer_digest != request_tokenizer_digest:
            candidate_reason_codes.append("TOKENIZER_MISMATCH")
        
        if candidate_reason_codes:
            status = "invalid"
            reason_codes.extend(candidate_reason_codes)
        else:
            status = "frozen"
    
    # Sort and deduplicate reason codes by UTF-8 byte ordering
    reason_codes = utf8_byte_sort(list(set(reason_codes)))
    
    return {
        "name": name,
        "status": status,
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": reason_codes
    }


def calculate_inventory(files: Dict[str, str]) -> Dict[str, Any]:
    """Calculate file inventory with UTF-8 byte lengths and SHA-256 hashes."""
    inventory = []
    
    # If files is empty, return empty inventory (invalid candidate)
    if not files:
        return {
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None
        }
    
    try:
        for filename in utf8_byte_sort(list(files.keys())):
            content = files[filename]
            utf8_bytes = content.encode('utf-8')
            byte_length = len(utf8_bytes)
            sha256_hash = sha256_hex(utf8_bytes)
            
            inventory.append({
                "name": filename,
                "bytes": byte_length,
                "sha256": sha256_hash
            })
        
        # Sort inventory by filename using UTF-8 byte ordering
        inventory.sort(key=lambda x: utf8_byte_sort([x["name"]])[0])
        
        total_bytes = sum(item["bytes"] for item in inventory)
        
        # Calculate packageDigest
        # Inventory object key order must be exactly: name, bytes, sha256
        canonical_inventory = []
        for item in inventory:
            canonical_item = sort_dict_by_keys(item, ["name", "bytes", "sha256"])
            canonical_inventory.append(canonical_item)
        
        inventory_json = compact_json(canonical_inventory)
        package_digest = sha256_hex(inventory_json.encode('utf-8'))
        
        return {
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest
        }
    except Exception:
        return {
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None
        }


async def handle_select(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle select phase."""
    # Validate input structure
    validation_error = validate_select_input(body)
    if validation_error:
        return JSONResponse(
            status_code=400,
            content={"error": validation_error}
        )
    
    freeze_id = body["freezeId"]
    
    # Check if freezeId exists
    if not storage.freeze_id_exists(freeze_id):
        # Return response with NOT_FROZEN for all candidates
        results = []
        for candidate in body["candidates"]:
            results.append({
                "name": candidate.get("name", ""),
                "aggregate": None,
                "slices": {},
                "totalBytes": None,
                "latencyMs": None,
                "admitted": False,
                "reasonCodes": ["NOT_FROZEN"]
            })
        
        # Sort results by candidateOrder then UTF-8 byte order
        results = sort_select_results(results, body["policy"]["candidateOrder"])
        
        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None
        }
    
    # Get stored freeze response
    freeze_response = storage.get_freeze_response(freeze_id)
    stored_candidates = freeze_response["candidates"]
    
    # Process selection
    results = []
    for candidate in body["candidates"]:
        result = process_select_candidate(
            candidate,
            stored_candidates,
            body["policy"],
            body["latencies"],
            body["rows"]
        )
        results.append(result)
    
    # Sort results by candidateOrder then UTF-8 byte order
    results = sort_select_results(results, body["policy"]["candidateOrder"])
    
    # Select winner among admitted candidates
    selected = select_winner(results, body["policy"]["candidateOrder"])
    
    # Get package manifest for winner
    package_manifest = None
    if selected:
        for stored_candidate in stored_candidates:
            if stored_candidate["name"] == selected:
                package_manifest = {
                    "name": stored_candidate["name"],
                    "inventory": stored_candidate["inventory"],
                    "totalBytes": stored_candidate["totalBytes"],
                    "packageDigest": stored_candidate["packageDigest"]
                }
                break
    
    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": package_manifest
    }


def validate_select_input(body: Dict[str, Any]) -> Optional[str]:
    """Validate select request input."""
    # Ensure body is not None
    if body is None:
        return "INVALID_INPUT"
    
    # Check required fields
    required_fields = ["phase", "freezeId", "candidates", "policy", "rows"]
    for field in required_fields:
        if field not in body:
            return "INVALID_INPUT"
    
    # Validate phase
    if body["phase"] != "select":
        return "INVALID_INPUT"
    
    # Validate freezeId
    if not isinstance(body["freezeId"], str) or not body["freezeId"]:
        return "INVALID_INPUT"
    
    # Validate candidates
    candidates = body["candidates"]
    if not isinstance(candidates, list) or not candidates:
        return "INVALID_INPUT"
    
    # Validate policy
    policy = body["policy"]
    if not isinstance(policy, dict):
        return "INVALID_INPUT"
    
    # Validate maxBytes
    if "maxBytes" not in policy:
        return "INVALID_INPUT"
    max_bytes = policy["maxBytes"]
    if not isinstance(max_bytes, int) or max_bytes < 0:
        return "INVALID_INPUT"
    
    # Validate aggregateFloor
    if "aggregateFloor" not in policy:
        return "INVALID_INPUT"
    aggregate_floor = policy["aggregateFloor"]
    if not isinstance(aggregate_floor, (int, float)) or not (0 <= aggregate_floor <= 1):
        return "INVALID_INPUT"
    
    # Validate requiredSlices
    required_slices = policy.get("requiredSlices", {})
    if not isinstance(required_slices, dict):
        return "INVALID_INPUT"
    for slice_name, floor in required_slices.items():
        if not isinstance(floor, (int, float)) or not (0 <= floor <= 1):
            return "INVALID_INPUT"
    
    # Validate maxLatencyMs
    if "maxLatencyMs" not in policy:
        return "INVALID_INPUT"
    max_latency = policy["maxLatencyMs"]
    if not isinstance(max_latency, (int, float)) or max_latency < 0:
        return "INVALID_INPUT"
    
    # Validate candidateOrder
    candidate_order = policy.get("candidateOrder", [])
    if not isinstance(candidate_order, list):
        return "INVALID_INPUT"
    # Check uniqueness
    if len(set(candidate_order)) != len(candidate_order):
        return "INVALID_INPUT"
    
    # Validate candidate names match
    submitted_names = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return "INVALID_INPUT"
        if "name" not in candidate or not isinstance(candidate["name"], str):
            return "INVALID_INPUT"
        submitted_names.add(candidate["name"])
    
    order_names = set(candidate_order)
    if submitted_names != order_names:
        return "INVALID_INPUT"
    
    # Validate latencies
    latencies = body.get("latencies", {})
    if not isinstance(latencies, dict):
        return "INVALID_INPUT"
    
    # Validate rows
    rows = body["rows"]
    if not isinstance(rows, list):
        return "INVALID_INPUT"
    for row in rows:
        if not isinstance(row, dict):
            return "INVALID_INPUT"
        if "label" not in row or not isinstance(row["label"], int):
            return "INVALID_INPUT"
        if "slice" not in row or not isinstance(row["slice"], str):
            return "INVALID_INPUT"
        if "predictions" not in row or not isinstance(row["predictions"], dict):
            return "INVALID_INPUT"
    
    return None


def process_select_candidate(
    candidate: Dict[str, Any],
    stored_candidates: List[Dict[str, Any]],
    policy: Dict[str, Any],
    latencies: Dict[str, float],
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Process a single select candidate."""
    name = candidate["name"]
    reason_codes = []
    
    # Find matching stored candidate
    stored_candidate = None
    for sc in stored_candidates:
        if sc["name"] == name:
            stored_candidate = sc
            break
    
    if stored_candidate is None:
        return {
            "name": name,
            "aggregate": None,
            "slices": {},
            "totalBytes": None,
            "latencyMs": None,
            "admitted": False,
            "reasonCodes": ["NOT_FROZEN"]
        }
    
    # Check if candidate is frozen
    if stored_candidate["status"] != "frozen":
        reason_codes.append("NOT_FROZEN")
        return {
            "name": name,
            "aggregate": None,
            "slices": {},
            "totalBytes": None,
            "latencyMs": None,
            "admitted": False,
            "reasonCodes": reason_codes
        }
    
    # Verify lineage - candidates must match exactly
    if not verify_lineage(candidate, stored_candidate):
        reason_codes.append("INVALID_LINEAGE")
        return {
            "name": name,
            "aggregate": None,
            "slices": {},
            "totalBytes": None,
            "latencyMs": None,
            "admitted": False,
            "reasonCodes": reason_codes
        }
    
    # Recompute manifest
    manifest_result = calculate_inventory(candidate["files"])
    
    # Validate manifest
    total_bytes = None
    if manifest_result["inventory"] == []:
        reason_codes.append("INVALID_MANIFEST")
    else:
        # Verify against stored manifest
        if (manifest_result["totalBytes"] != stored_candidate["totalBytes"] or
            manifest_result["packageDigest"] != stored_candidate["packageDigest"]):
            reason_codes.append("INVALID_MANIFEST")
            total_bytes = None
        else:
            total_bytes = manifest_result["totalBytes"]
    
    # Check size limit
    if total_bytes is not None and total_bytes > policy["maxBytes"]:
        reason_codes.append("SIZE_LIMIT")
    
    # Get latency
    latency_ms = latencies.get(name)
    if latency_ms is None:
        latency_ms = None
        reason_codes.append("LATENCY_LIMIT")
    else:
        if latency_ms > policy["maxLatencyMs"]:
            reason_codes.append("LATENCY_LIMIT")
    
    # Validate predictions and calculate accuracy
    prediction_result = validate_predictions_and_calculate_accuracy(
        name, rows, policy["aggregateFloor"], policy["requiredSlices"]
    )
    
    if prediction_result["invalid"]:
        reason_codes.append("INVALID_PREDICTIONS")
        aggregate = None
        slices = {}
    else:
        aggregate = prediction_result["aggregate"]
        slices = prediction_result["slices"]
        reason_codes.extend(prediction_result["reason_codes"])
    
    # Determine admission
    admitted = (
        "NOT_FROZEN" not in reason_codes and
        "INVALID_LINEAGE" not in reason_codes and
        "INVALID_PREDICTIONS" not in reason_codes and
        "INVALID_MANIFEST" not in reason_codes and
        "SIZE_LIMIT" not in reason_codes and
        "LATENCY_LIMIT" not in reason_codes and
        "AGGREGATE_FLOOR" not in reason_codes and
        not any(code.startswith("MISSING_SLICE:") for code in reason_codes) and
        not any(code.startswith("SLICE_FLOOR:") for code in reason_codes)
    )
    
    # Sort and deduplicate reason codes
    reason_codes = utf8_byte_sort(list(set(reason_codes)))
    
    return {
        "name": name,
        "aggregate": aggregate,
        "slices": slices,
        "totalBytes": total_bytes,
        "latencyMs": latency_ms,
        "admitted": admitted,
        "reasonCodes": reason_codes
    }


def verify_lineage(candidate: Dict[str, Any], stored_candidate: Dict[str, Any]) -> bool:
    """Verify that the candidate matches the stored candidate exactly."""
    # Recompute manifest for the candidate
    manifest_result = calculate_inventory(candidate["files"])
    
    # Compare manifests - this is the key lineage check
    # The stored candidate should have the same totalBytes and packageDigest
    if (manifest_result["totalBytes"] != stored_candidate["totalBytes"] or
        manifest_result["packageDigest"] != stored_candidate["packageDigest"]):
        return False
    
    return True


def validate_predictions_and_calculate_accuracy(
    candidate_name: str,
    rows: List[Dict[str, Any]],
    aggregate_floor: float,
    required_slices: Dict[str, float]
) -> Dict[str, Any]:
    """Validate predictions and calculate accuracy."""
    reason_codes = []
    invalid = False
    
    # Check predictions for each row
    valid_predictions = True
    for row in rows:
        predictions = row["predictions"]
        if candidate_name not in predictions:
            valid_predictions = False
            break
        prediction = predictions[candidate_name]
        if prediction not in [0, 1]:
            valid_predictions = False
            break
    
    if not valid_predictions:
        return {
            "invalid": True,
            "aggregate": None,
            "slices": {},
            "reason_codes": []
        }
    
    # Calculate accuracy
    total_rows = len(rows)
    correct_predictions = 0
    
    # Track slice data
    slice_data = {}
    for row in rows:
        slice_name = row["slice"]
        if slice_name not in slice_data:
            slice_data[slice_name] = {"total": 0, "correct": 0}
        
        slice_data[slice_name]["total"] += 1
        
        prediction = row["predictions"][candidate_name]
        if prediction == row["label"]:
            correct_predictions += 1
            slice_data[slice_name]["correct"] += 1
    
    # Calculate aggregate accuracy
    aggregate = round_to_decimal_places(correct_predictions / total_rows, 12) if total_rows > 0 else 0.0
    
    # Check aggregate floor
    if aggregate < aggregate_floor:
        reason_codes.append("AGGREGATE_FLOOR")
    
    # Calculate slice accuracies
    slices = {}
    for slice_name, floor in required_slices.items():
        if slice_name not in slice_data:
            reason_codes.append(f"MISSING_SLICE:{slice_name}")
            slices[slice_name] = None
        else:
            slice_correct = slice_data[slice_name]["correct"]
            slice_total = slice_data[slice_name]["total"]
            slice_accuracy = round_to_decimal_places(slice_correct / slice_total, 12) if slice_total > 0 else 0.0
            slices[slice_name] = slice_accuracy
            
            if slice_accuracy < floor:
                reason_codes.append(f"SLICE_FLOOR:{slice_name}")
    
    return {
        "invalid": False,
        "aggregate": aggregate,
        "slices": slices,
        "reason_codes": reason_codes
    }


def sort_select_results(results: List[Dict[str, Any]], candidate_order: List[str]) -> List[Dict[str, Any]]:
    """Sort select results by candidateOrder, then UTF-8 byte order."""
    def sort_key(result):
        name = result["name"]
        if name in candidate_order:
            return (0, candidate_order.index(name))
        else:
            return (1, utf8_byte_sort([name])[0])
    
    return sorted(results, key=sort_key)


def select_winner(results: List[Dict[str, Any]], candidate_order: List[str]) -> Optional[str]:
    """Select winner among admitted candidates."""
    admitted = [r for r in results if r["admitted"]]
    
    if not admitted:
        return None
    
    # Sort by: 1. smaller totalBytes, 2. lower latencyMs, 3. candidate order
    def winner_key(result):
        # Handle None values in sorting
        total_bytes = result["totalBytes"] if result["totalBytes"] is not None else float('inf')
        latency_ms = result["latencyMs"] if result["latencyMs"] is not None else float('inf')
        
        # Candidate order preference
        name = result["name"]
        if name in candidate_order:
            order_index = candidate_order.index(name)
        else:
            order_index = len(candidate_order)
        
        return (total_bytes, latency_ms, order_index)
    
    winner = min(admitted, key=winner_key)
    return winner["name"]


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
