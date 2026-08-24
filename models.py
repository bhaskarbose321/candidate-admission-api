from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class CandidateFile(BaseModel):
    files: Dict[str, str] = Field(..., description="Filename to file content mapping")


class FreezeCandidate(BaseModel):
    name: str = Field(..., description="Candidate name")
    files: Dict[str, str] = Field(..., description="Filename to file content mapping")
    loadable: bool = Field(..., description="Whether the candidate is loadable")
    calibrationDigest: str = Field(..., description="Calibration digest")
    tokenizerDigest: str = Field(..., description="Tokenizer digest")
    unsupportedReason: Optional[str] = Field(None, description="Unsupported reason if any")


class FreezeRequest(BaseModel):
    phase: str = Field(..., description="Must be 'freeze'")
    freezeId: str = Field(..., description="Unique freeze identifier")
    calibrationDigest: str = Field(..., description="Calibration digest")
    tokenizerDigest: str = Field(..., description="Tokenizer digest")
    allowedUnsupportedReasons: List[str] = Field(default_factory=list, description="Allowed unsupported reasons")
    candidates: List[FreezeCandidate] = Field(..., description="List of candidates")


class InventoryEntry(BaseModel):
    name: str
    bytes: int
    sha256: str


class FreezeCandidateResult(BaseModel):
    name: str
    status: str  # "frozen", "unsupported", "invalid"
    inventory: List[InventoryEntry]
    totalBytes: Optional[int]
    packageDigest: Optional[str]
    reasonCodes: List[str]


class FreezeResponse(BaseModel):
    freezeId: str
    candidates: List[FreezeCandidateResult]


class SelectPolicy(BaseModel):
    maxBytes: int = Field(..., description="Maximum allowed bytes")
    aggregateFloor: float = Field(..., description="Aggregate accuracy floor")
    requiredSlices: Dict[str, float] = Field(default_factory=dict, description="Required slice floors")
    maxLatencyMs: float = Field(..., description="Maximum allowed latency in ms")
    candidateOrder: List[str] = Field(default_factory=list, description="Candidate order preference")


class RowPrediction(BaseModel):
    label: int = Field(..., description="True label")
    slice: str = Field(..., description="Slice name")
    predictions: Dict[str, int] = Field(..., description="Candidate predictions")


class SelectRequest(BaseModel):
    phase: str = Field(..., description="Must be 'select'")
    freezeId: str = Field(..., description="Freeze identifier to select from")
    candidates: List[Dict[str, Any]] = Field(..., description="Candidates to select from")
    policy: SelectPolicy = Field(..., description="Selection policy")
    latencies: Dict[str, float] = Field(default_factory=dict, description="Candidate latencies")
    rows: List[RowPrediction] = Field(..., description="Prediction rows")


class SliceResult(BaseModel):
    name: str
    aggregate: Optional[float]
    slices: Dict[str, Optional[float]]
    totalBytes: Optional[int]
    latencyMs: Optional[float]
    admitted: bool
    reasonCodes: List[str]


class SelectResponse(BaseModel):
    freezeId: str
    selected: Optional[str]
    results: List[SliceResult]
    packageManifest: Optional[Dict[str, Any]]


class ErrorResponse(BaseModel):
    error: str
