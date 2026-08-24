import hashlib
import json
from typing import List, Dict, Any


def utf8_byte_sort(strings: List[str]) -> List[str]:
    """Sort strings by UTF-8 byte ordering."""
    return sorted(strings, key=lambda s: s.encode('utf-8'))


def utf8_byte_length(text: str) -> int:
    """Calculate UTF-8 byte length of a string."""
    return len(text.encode('utf-8'))


def sha256_hex(data: bytes) -> str:
    """Calculate SHA-256 hash and return lowercase hexadecimal string."""
    return hashlib.sha256(data).hexdigest()


def compact_json(obj: Any) -> str:
    """Serialize object to compact JSON (no spaces, no pretty printing)."""
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def canonical_json(obj: Any) -> str:
    """Serialize object to canonical JSON with sorted keys."""
    return json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False)


def sort_dict_by_keys(obj: Dict[str, Any], key_order: List[str]) -> Dict[str, Any]:
    """Sort dictionary keys according to specified order, then alphabetically."""
    ordered = {}
    remaining_keys = set(obj.keys())
    
    # Add keys in specified order
    for key in key_order:
        if key in remaining_keys:
            ordered[key] = obj[key]
            remaining_keys.remove(key)
    
    # Add remaining keys in UTF-8 byte order
    for key in utf8_byte_sort(list(remaining_keys)):
        ordered[key] = obj[key]
    
    return ordered


def round_to_decimal_places(value: float, places: int) -> float:
    """Round a float to specified decimal places."""
    if value is None:
        return None
    return round(value, places)
