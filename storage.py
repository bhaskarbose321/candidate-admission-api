from typing import Dict, Any, Optional
import hashlib
import json


class StateStorage:
    """In-memory storage for freeze responses and request identity."""
    
    def __init__(self):
        self._freeze_responses: Dict[str, Dict[str, Any]] = {}
        self._freeze_request_identities: Dict[str, str] = {}
    
    def _compute_request_identity(self, request: Dict[str, Any]) -> str:
        """Compute a hash that uniquely identifies a freeze request."""
        # Use canonical JSON to ensure deterministic identity
        canonical = json.dumps(request, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def store_freeze(self, freeze_id: str, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Store freeze response and request identity."""
        request_identity = self._compute_request_identity(request)
        self._freeze_responses[freeze_id] = response
        self._freeze_request_identities[freeze_id] = request_identity
    
    def get_freeze_response(self, freeze_id: str) -> Optional[Dict[str, Any]]:
        """Get stored freeze response by freezeId."""
        return self._freeze_responses.get(freeze_id)
    
    def freeze_id_exists(self, freeze_id: str) -> bool:
        """Check if freezeId exists."""
        return freeze_id in self._freeze_responses
    
    def is_same_request(self, freeze_id: str, request: Dict[str, Any]) -> bool:
        """Check if the request is identical to the stored one."""
        if freeze_id not in self._freeze_request_identities:
            return False
        current_identity = self._compute_request_identity(request)
        stored_identity = self._freeze_request_identities[freeze_id]
        return current_identity == stored_identity
    
    def clear(self) -> None:
        """Clear all stored state (useful for testing)."""
        self._freeze_responses.clear()
        self._freeze_request_identities.clear()


# Global storage instance
storage = StateStorage()
