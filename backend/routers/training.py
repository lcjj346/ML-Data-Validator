"""
Training endpoints - upload training data, run training, list models.
"""

import io
import json
import os
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas import ModelInfo, TrainUploadResponse
from backend.state import store
from ml.validator import UnifiedMLValidator

router = APIRouter()
models_router = APIRouter()


def _get_session(session_id: str):
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


# ── Upload Training Data ────────────────────────────────────

@router.post("/upload", response_model=TrainUploadResponse)
async def upload_training_csv(file: UploadFile = File(...)):
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
    session.training_df = df
    session.training_filename = file.filename

    preview = df.head(10).fillna("").to_dict(orient="records")

    return TrainUploadResponse(
        session_id=session_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        preview=preview,
    )


# ── Run Training (SSE) ──────────────────────────────────────

@router.get("/{session_id}/run")
async def run_training(
    session_id: str,
    model_name: str,
    training_mode: str = "new",
    exclude_columns: str = "",
    use_reference_lists: bool = True,
):
    session = _get_session(session_id)
    if session.training_df is None:
        raise HTTPException(status_code=400, detail="No training data uploaded")

    exclude_list = [c.strip() for c in exclude_columns.split(",") if c.strip()]

    def generate():
        train_df = session.training_df

        if training_mode == "new":
            yield f"data: {json.dumps({'type': 'progress', 'progress': 0.05, 'message': 'Initializing new model...'})}\n\n"
            validator = UnifiedMLValidator()
            yield f"data: {json.dumps({'type': 'progress', 'progress': 0.10, 'message': 'Training...'})}\n\n"
            metrics = validator.train(train_df, model_name, exclude_list)
        else:
            model_path = f"models/{model_name}.pkl"
            if not os.path.exists(model_path):
                yield f"data: {json.dumps({'type': 'error', 'message': f'Model {model_name} not found'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'progress', 'progress': 0.10, 'message': 'Loading existing model...'})}\n\n"
            validator = UnifiedMLValidator(model_path)
            if not validator.is_trained:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to load model'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'progress', 'progress': 0.30, 'message': 'Adding new data...'})}\n\n"
            metrics = validator.add_training_data(train_df, retrain=True)

        # Load reference lists if enabled
        if use_reference_lists and os.path.exists("reference_lists"):
            yield f"data: {json.dumps({'type': 'progress', 'progress': 0.70, 'message': 'Loading reference lists...'})}\n\n"
            validator.load_reference_lists_from_dir("reference_lists")

        yield f"data: {json.dumps({'type': 'progress', 'progress': 0.80, 'message': 'Saving model...'})}\n\n"
        os.makedirs("models", exist_ok=True)
        validator.save(f"models/{model_name}.pkl")

        # Serialize metrics (convert numpy types, confusion matrices etc.)
        serializable_metrics = {}
        for col_name, col_metrics in metrics.items():
            clean = {}
            for k, v in col_metrics.items():
                if hasattr(v, "tolist"):
                    clean[k] = v.tolist()
                elif isinstance(v, float):
                    clean[k] = round(v, 4)
                else:
                    clean[k] = v
            serializable_metrics[col_name] = clean

        session.training_metrics = serializable_metrics

        yield f"data: {json.dumps({'type': 'progress', 'progress': 1.0, 'message': 'Done!'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'metrics': serializable_metrics})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Delete Model ─────────────────────────────────────────────

@models_router.delete("/{model_name}")
async def delete_model(model_name: str):
    if model_name == "base_model":
        raise HTTPException(status_code=403, detail="The base model cannot be deleted.")

    model_path = os.path.join("models", f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    os.remove(model_path)
    return {"ok": True, "deleted": model_name}


# ── List Models ──────────────────────────────────────────────

@models_router.get("", response_model=List[ModelInfo])
async def list_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []

    result = []
    for f in os.listdir(models_dir):
        if not f.endswith(".pkl"):
            continue

        model_name = f.replace(".pkl", "")
        model_path = os.path.join(models_dir, f)

        try:
            temp = UnifiedMLValidator(model_path)
            columns = temp.get_trained_columns()
            columns_str = ", ".join(columns[:5])
            if len(columns) > 5:
                columns_str += f" (+{len(columns) - 5} more)"
        except Exception:
            columns_str = "Unknown"

        size_kb = os.path.getsize(model_path) / 1024
        result.append(ModelInfo(
            name=model_name,
            size=f"{size_kb:.1f} KB",
            columns=columns_str,
        ))

    return result
