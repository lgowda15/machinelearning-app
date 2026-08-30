"""Pydantic schemas for the data upload / EDA endpoints.

These are the authoritative contract (.claude/rules/backend.md) -- the
frontend's TypeScript types mirror them, not the reverse.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field

DataType = Literal["classification", "regression", "clustering"]
ColumnKind = Literal["numeric", "categorical"]


class NumericDistribution(BaseModel):
    kind: Literal["numeric"] = "numeric"
    bin_edges: list[float]
    counts: list[int]


class CategoricalDistribution(BaseModel):
    kind: Literal["categorical"] = "categorical"
    categories: list[str]
    counts: list[int]
    other_count: int = 0  # samples collapsed beyond the top N categories shown


Distribution = Annotated[
    NumericDistribution | CategoricalDistribution, Field(discriminator="kind")
]


class ColumnSummary(BaseModel):
    name: str
    dtype: ColumnKind
    is_target: bool
    missing_count: int
    missing_pct: float
    unique_count: int
    distribution: Distribution

    # Numeric columns only.
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None

    # Categorical columns only.
    top_value: str | None = None
    top_value_freq: int | None = None


class ImbalanceInfo(BaseModel):
    is_imbalanced: bool
    class_counts: dict[str, int]
    message: str | None = None


class DataProfileResponse(BaseModel):
    data_id: str
    source: str
    n_rows: int
    n_columns: int
    data_type: DataType
    target_column: str | None = None
    columns: list[ColumnSummary]
    class_imbalance: ImbalanceInfo | None = None


class SampleDatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    data_type: DataType
    n_rows: int
    n_columns: int


class SampleListResponse(BaseModel):
    samples: list[SampleDatasetInfo]
