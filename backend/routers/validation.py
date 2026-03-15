"""
Validation endpoints - upload, validate, correct, export.
"""

import io
import json
import os

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas import (
    ApplyCorrectionRequest,
    CorrectionItem,
    EditCellRequest,
    FileInfoResponse,
    MatchColumnsRequest,
    MatchColumnsResponse,
    ValidationResultsResponse,
)
from backend.state import store
from ml.validator import UnifiedMLValidator

router = APIRouter()


def _safe_cast_value(value, target_dtype):
    """Cast a value to the target dtype, handling type mismatches gracefully."""
    try:
        if pd.api.types.is_float_dtype(target_dtype):
            return float(value)
        elif pd.api.types.is_integer_dtype(target_dtype):
            return int(float(value))
        elif pd.api.types.is_bool_dtype(target_dtype):
            return bool(value)
        else:
            return value
    except (ValueError, TypeError):
        return value


def _get_session(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


# ── Upload ──────────────────────────────────────────────────

@router.post("/upload", response_model=FileInfoResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are accepted")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 10MB.")

    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    session_id = store.create()
    session = store.get(session_id)
    session.filename = file.filename
    session.df = df
    session.original_df = df.copy()

    preview = df.head(10).fillna("").to_dict(orient="records")

    return FileInfoResponse(
        session_id=session_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        preview=preview,
    )


# ── Column Matching ─────────────────────────────────────────

@router.post("/match-columns", response_model=MatchColumnsResponse)
async def match_columns(req: MatchColumnsRequest):
    session = _get_session(req.session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No file uploaded in this session")

    model_path = f"models/{req.model_name}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")

    validator = UnifiedMLValidator(model_path)
    if not validator.is_trained:
        raise HTTPException(status_code=400, detail="Model failed to load")

    trained_cols = validator.get_trained_columns()
    matched = validator._match_columns(session.df.columns.tolist())
    unmatched = [c for c in session.df.columns if c not in matched]

    session.column_mappings = matched

    return MatchColumnsResponse(
        matched=matched,
        unmatched=unmatched,
        trained_columns=trained_cols,
    )


# ── Run Validation (SSE) ────────────────────────────────────

@router.get("/{session_id}/run")
async def run_validation(session_id: str, model_name: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    model_path = f"models/{model_name}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    def generate():
        validator = UnifiedMLValidator(model_path)
        if not validator.is_trained:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Model failed to load'})}\n\n"
            return

        df = session.df
        column_mapping = validator._match_columns(df.columns.tolist())
        session.column_mappings = column_mapping

        cell_validity = {}
        all_corrections = []
        total_columns = len(column_mapping)

        for col_idx, (input_col, trained_col) in enumerate(column_mapping.items()):
            progress = col_idx / total_columns if total_columns else 0
            yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'message': f'Validating column: {input_col}...'})}\n\n"

            # Validate batch
            results = validator.validate_batch(
                df[input_col].astype(str).tolist(), trained_col
            )

            for idx, (is_valid, confidence) in enumerate(results):
                key = f"{idx}_{input_col}"
                cell_validity[key] = is_valid
                session.cell_confidence[key] = round(float(confidence), 3)

            # Get corrections for invalid cells
            for idx, row in df.iterrows():
                if not results[idx][0]:
                    original = str(row[input_col])
                    corrected = validator.correct(original, trained_col)
                    reason = validator.explain_invalidity(original, trained_col)

                    suggested = corrected if (corrected and corrected != original) else original
                    has_correction = corrected is not None and corrected != original
                    conf = session.cell_confidence.get(f"{idx}_{input_col}", 0.0)

                    all_corrections.append(
                        CorrectionItem(
                            row_index=int(idx),
                            row=int(idx) + 1,
                            column=input_col,
                            original=original,
                            suggested=suggested,
                            has_correction=has_correction,
                            reason=reason or "Unknown",
                            confidence=conf,
                        ).model_dump()
                    )

        # Store results in session
        session.cell_validity = cell_validity
        session.corrections = all_corrections
        session.modified_cells = set()

        yield f"data: {json.dumps({'type': 'progress', 'progress': 1.0, 'message': 'Validation complete!'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Results ──────────────────────────────────────────────────

@router.get("/{session_id}/results", response_model=ValidationResultsResponse)
async def get_results(session_id: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    df = session.df
    total_cells = len(df) * len(session.column_mappings) if session.column_mappings else 0
    valid_cells = sum(1 for v in session.cell_validity.values() if v)
    invalid_cells = total_cells - valid_cells
    quality = (valid_cells / total_cells * 100) if total_cells > 0 else 0

    return ValidationResultsResponse(
        data=df.fillna("").to_dict(orient="records"),
        cell_validity=session.cell_validity,
        cell_confidence=session.cell_confidence,
        modified_cells=list(session.modified_cells),
        corrections=session.corrections,
        quality=quality,
        valid_cells=valid_cells,
        invalid_cells=invalid_cells,
        total_cells=total_cells,
    )


# ── Edit Cell ────────────────────────────────────────────────

@router.post("/{session_id}/edit-cell")
async def edit_cell(session_id: str, req: EditCellRequest):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    if req.row < 0 or req.row >= len(session.df):
        raise HTTPException(status_code=400, detail="Row index out of range")
    if req.column not in session.df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found")

    target_dtype = session.df[req.column].dtype
    casted = _safe_cast_value(req.value, target_dtype)
    session.df.at[req.row, req.column] = casted

    key = f"{req.row}_{req.column}"
    session.modified_cells.add(key)

    return {"ok": True, "key": key}


# ── Apply Single Correction ─────────────────────────────────

@router.post("/{session_id}/apply-correction")
async def apply_correction(session_id: str, req: ApplyCorrectionRequest):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    if req.row < 0 or req.row >= len(session.df):
        raise HTTPException(status_code=400, detail="Row index out of range")
    if req.column not in session.df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found")

    target_dtype = session.df[req.column].dtype
    casted = _safe_cast_value(req.suggested, target_dtype)
    session.df.at[req.row, req.column] = casted

    key = f"{req.row}_{req.column}"
    session.modified_cells.add(key)

    return {"ok": True, "key": key}


# ── Apply All Corrections ───────────────────────────────────

@router.post("/{session_id}/apply-all")
async def apply_all_corrections(session_id: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    applied = 0
    for c in session.corrections:
        correction = c if isinstance(c, dict) else c.model_dump()
        if not correction["has_correction"]:
            continue
        row_idx = correction["row_index"]
        col = correction["column"]
        key = f"{row_idx}_{col}"
        if key in session.modified_cells:
            continue

        target_dtype = session.df[col].dtype
        casted = _safe_cast_value(correction["suggested"], target_dtype)
        session.df.at[row_idx, col] = casted
        session.modified_cells.add(key)
        applied += 1

    return {"ok": True, "applied": applied}


# ── Export ───────────────────────────────────────────────────

@router.get("/{session_id}/export")
async def export_csv(session_id: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    export_df = session.df.drop(columns=["is_valid", "confidence"], errors="ignore")
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    filename = f"validated_{session.filename}" if session.filename else "validated_data.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Export Excel ─────────────────────────────────────────────

@router.get("/{session_id}/export-xlsx")
async def export_xlsx(session_id: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    export_df = session.df.drop(columns=["is_valid", "confidence"], errors="ignore")

    buffer = io.BytesIO()
    export_df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    base = session.filename.rsplit(".", 1)[0] if session.filename else "validated_data"
    filename = f"validated_{base}.xlsx"

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Summary Report Export ────────────────────────────────────

@router.get("/{session_id}/export-report")
async def export_summary_report(session_id: str):
    session = _get_session(session_id)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No data in session")

    df = session.df
    cell_validity = session.cell_validity
    corrections = session.corrections
    column_mappings = session.column_mappings or {}

    total_rows = len(df)
    total_cells = total_rows * len(column_mappings)
    valid_cells = sum(1 for v in cell_validity.values() if v)
    invalid_cells = total_cells - valid_cells
    quality = (valid_cells / total_cells * 100) if total_cells > 0 else 0

    lines = []

    # Section 1: Overview
    lines.append("Section,Metric,Value")
    lines.append(f"Overview,Total Rows,{total_rows}")
    lines.append(f"Overview,Total Cells Validated,{total_cells}")
    lines.append(f"Overview,Valid Cells,{valid_cells}")
    lines.append(f"Overview,Invalid Cells,{invalid_cells}")
    lines.append(f"Overview,Data Quality,{quality:.1f}%")

    # Blank separator
    lines.append("")

    # Section 2: Per-column breakdown
    lines.append("Column,Total Cells,Invalid Count,Invalid %,Corrections Available,Top Error Reason")
    for input_col in column_mappings:
        col_keys = [k for k in cell_validity if k.endswith(f"_{input_col}")]
        col_total = len(col_keys)
        col_invalid = sum(1 for k in col_keys if not cell_validity[k])
        col_invalid_pct = (col_invalid / col_total * 100) if col_total > 0 else 0

        col_corrections = [c for c in corrections if c["column"] == input_col]
        corrections_available = sum(1 for c in col_corrections if c["has_correction"])

        reasons = [c["reason"] for c in col_corrections if c.get("reason")]
        top_reason = max(set(reasons), key=reasons.count) if reasons else ""
        # Escape commas in the reason for CSV safety
        top_reason_escaped = f'"{top_reason}"' if "," in top_reason else top_reason

        lines.append(
            f"{input_col},{col_total},{col_invalid},{col_invalid_pct:.1f}%,"
            f"{corrections_available},{top_reason_escaped}"
        )

    csv_bytes = "\n".join(lines).encode("utf-8")
    filename = f"validation_report_{session.filename}" if session.filename else "validation_report.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
