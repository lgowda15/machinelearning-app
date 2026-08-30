"""Pydantic schemas for the prediction endpoint (.claude/rules/backend.md
-- authoritative contract). Stage 6 (DATA_FLOW_GUIDE.md SS7): predict with
an already-trained model on freshly uploaded data, no refitting."""
from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    training_id: str
    model_key: str
    model_type: str
    n_samples: int
    # 1D for classifier/clusterer/regressor, 2D (n_samples, n_components)
    # for dimensionality_reducer (CLAUDE.md contract exceptions) -- Any
    # covers both shapes, same pattern as TrainedModelResponse.metrics.
    predictions: list[Any]
    # Only classifiers produce probabilities; None for every other
    # model_type (CLAUDE.md contract exceptions).
    probabilities: list[list[float]] | None = None
