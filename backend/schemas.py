"""
Pydantic request/response models for all API endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel


# ── Validation ──────────────────────────────────────────────

class FileInfoResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]  # first 10 rows as list of dicts


class MatchColumnsRequest(BaseModel):
    session_id: str
    model_name: str


class MatchColumnsResponse(BaseModel):
    matched: Dict[str, str]  # input_col -> trained_col
    unmatched: List[str]
    trained_columns: List[str]


class EditCellRequest(BaseModel):
    row: int
    column: str
    value: Any


class ApplyCorrectionRequest(BaseModel):
    row: int
    column: str
    suggested: str


class CorrectionItem(BaseModel):
    row_index: int
    row: int  # 1-based display row
    column: str
    original: str
    suggested: str
    has_correction: bool
    reason: str
    confidence: float = 0.0


class ValidationResultsResponse(BaseModel):
    data: List[Dict[str, Any]]
    cell_validity: Dict[str, bool]  # "rowIdx_colName" -> bool
    cell_confidence: Dict[str, float] = {}
    modified_cells: List[str]
    corrections: List[CorrectionItem]
    quality: float
    valid_cells: int
    invalid_cells: int
    total_cells: int


# ── Training ────────────────────────────────────────────────

class TrainUploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]


class ColumnMetrics(BaseModel):
    unique_valid: Optional[int] = None
    total_samples: Optional[int] = None
    used_split: bool = False
    test_size: Optional[int] = None
    train_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    test_confusion_matrix: Optional[List[List[int]]] = None


# ── Models ──────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    size: str
    columns: str
