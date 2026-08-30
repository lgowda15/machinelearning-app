"""In-memory training-result store, mirroring app.core.storage's dataset
store (single-process demo app, no database -- see storage.py).

Holds the fitted model instance and the fitted preprocessors together per
training_id, since Session 4's Stage 6 prediction reuses the exact fitted
imputer/encoder/scaler from training and never refits (DATA_FLOW_GUIDE.md
SS7, .claude/rules/backend.md).
"""
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.errors import NotFoundError
from app.core.preprocessing import FittedPreprocessors
from app.schemas.data import DataType
from models.base_model import BaseModel


@dataclass
class TrainedModelResult:
    model_key: str
    model: BaseModel
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    visualization_data: dict[str, Any] | None
    plot_data: dict[str, Any] | None


@dataclass
class TrainingRecord:
    training_id: str
    data_id: str
    data_type: DataType
    target_column: str | None
    test_size: float
    fitted: FittedPreprocessors
    results: list[TrainedModelResult]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_STORE: dict[str, TrainingRecord] = {}


def save_training(
    data_id: str,
    data_type: DataType,
    target_column: str | None,
    test_size: float,
    fitted: FittedPreprocessors,
    results: list[TrainedModelResult],
) -> TrainingRecord:
    training_id = str(uuid.uuid4())
    record = TrainingRecord(
        training_id=training_id,
        data_id=data_id,
        data_type=data_type,
        target_column=target_column,
        test_size=test_size,
        fitted=fitted,
        results=results,
    )
    _STORE[training_id] = record
    return record


def get_training(training_id: str) -> TrainingRecord:
    record = _STORE.get(training_id)
    if record is None:
        raise NotFoundError(
            f"No training run found for training_id '{training_id}'.",
            details={"training_id": training_id},
        )
    return record


def clear() -> None:
    """Test-only: reset the store between tests."""
    _STORE.clear()
