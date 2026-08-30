"""Prediction endpoint -- Stage 6 (DATA_FLOW_GUIDE.md SS7,
BUILD_SESSIONS.md Session 4).

POST /api/prediction/predict -- run an already-trained model on a freshly
uploaded CSV. Reuses the exact fitted imputer/encoder/scaler captured at
training time (app.core.training_store); never refits. A column mismatch
against the training features is rejected before the model is called,
naming the mismatched columns (app.core.preprocessing.FittedPreprocessors).
"""
import io

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from app.core import training_store
from app.core.errors import DataValidationError, NotFoundError
from app.core.preprocessing import transform_new
from app.core.training_store import TrainedModelResult, TrainingRecord
from app.schemas.errors import ErrorResponse
from app.schemas.prediction import PredictionResponse

router = APIRouter(
    prefix="/api/prediction",
    tags=["prediction"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)


def _parse_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise DataValidationError(f"Could not parse file as CSV: {exc}") from exc
    if len(df.columns) == 0:
        raise DataValidationError("Could not parse file as CSV: no columns found.")
    return df


def _find_result(record: TrainingRecord, model_key: str) -> TrainedModelResult:
    for result in record.results:
        if result.model_key == model_key:
            return result
    raise NotFoundError(
        f"No model '{model_key}' was trained in training run '{record.training_id}'.",
        details={
            "training_id": record.training_id,
            "model_key": model_key,
            "available": [r.model_key for r in record.results],
        },
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    training_id: str = Form(...),
    model_key: str = Form(...),
    file: UploadFile = File(...),  # noqa: B008 — FastAPI's parameter-marker pattern requires the call in the default
) -> PredictionResponse:
    record = training_store.get_training(training_id)
    result = _find_result(record, model_key)

    raw = await file.read()
    df = _parse_csv(raw)
    X_new = transform_new(df, record.fitted)

    model_type = result.metadata["model_type"]
    try:
        predictions = result.model.predict(X_new)
        probabilities = result.model.predict_proba(X_new) if model_type == "classifier" else None
    except Exception as exc:
        # A model's own exception is caught at the route boundary and
        # returned as a clean message -- it never surfaces as a 500
        # (.claude/rules/backend.md "Errors").
        raise DataValidationError(
            f"Model '{model_key}' failed during prediction: {exc}",
            details={"model_key": model_key},
        ) from exc

    return PredictionResponse(
        training_id=training_id,
        model_key=model_key,
        model_type=model_type,
        n_samples=X_new.shape[0],
        predictions=predictions.tolist(),
        probabilities=probabilities.tolist() if probabilities is not None else None,
    )
