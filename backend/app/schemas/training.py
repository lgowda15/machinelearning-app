"""Pydantic schemas for the training endpoints (.claude/rules/backend.md
-- authoritative contract). Training is synchronous (ARCHITECTURE.md SS6):
the client submits a train request and receives metrics in the response."""
from typing import Any

from pydantic import BaseModel, Field


class ModelTrainSpec(BaseModel):
    model_key: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    data_id: str
    models: list[ModelTrainSpec] = Field(min_length=1)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class TrainedModelResponse(BaseModel):
    model_key: str
    model_name: str
    model_type: str
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    training_time_seconds: float | None
    n_features: int
    feature_importance: dict[str, float] | None
    visualization_data: dict[str, Any] | None
    # Backend-generic per-sample data for screen 5's cluster-scatter /
    # predicted-vs-actual charts (app.core.metrics.compute_plot_data) --
    # deliberately separate from visualization_data, which stays
    # model-owned (DATA_FLOW_GUIDE.md SS5.3). None for classifier and
    # dimensionality_reducer, whose charts are fed entirely by `metrics`.
    plot_data: dict[str, Any] | None


class TrainResponse(BaseModel):
    training_id: str
    data_id: str
    test_size: float
    results: list[TrainedModelResponse]
