"""Pydantic schemas for the results-comparison endpoint
(.claude/rules/backend.md -- authoritative contract).

Comparison is cross-run by design: two {training_id, model_key} pairs may
name the same training run or two different ones, so a user can compare
models trained separately without retraining them together.
"""
from typing import Any

from pydantic import BaseModel, Field


class ComparisonModelRef(BaseModel):
    training_id: str
    model_key: str


class ComparisonRequest(BaseModel):
    models: list[ComparisonModelRef] = Field(min_length=2)


class ComparisonEntry(BaseModel):
    training_id: str
    model_key: str
    model_name: str
    model_type: str
    metrics: dict[str, Any]


class ComparisonResponse(BaseModel):
    # Metric keys present on every requested model -- empty if the
    # requested models don't share a model_type. Descriptive only: this
    # is not a ranking or a recommendation (CLAUDE.md "Hard rules").
    common_metrics: list[str]
    models: list[ComparisonEntry]
