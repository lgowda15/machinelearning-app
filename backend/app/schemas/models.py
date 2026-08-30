"""Pydantic schemas for the model registry / compatibility endpoints
(.claude/rules/backend.md -- authoritative contract)."""
from typing import Any

from pydantic import BaseModel

from app.schemas.data import DataType


class ModelSummary(BaseModel):
    key: str
    model_name: str
    model_type: str
    default_hyperparameters: dict[str, Any]


class IncompatibleModelSummary(ModelSummary):
    reason: str


class RegistryResponse(BaseModel):
    models: list[ModelSummary]


class CompatibilityRequest(BaseModel):
    data_id: str


class CompatibilityResponse(BaseModel):
    data_type: DataType
    compatible: list[ModelSummary]
    incompatible: list[IncompatibleModelSummary]
