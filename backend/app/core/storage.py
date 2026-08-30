"""In-memory dataset store.

Single-process demo app, no database anywhere in ARCHITECTURE.md, so
datasets uploaded or loaded from a sample live in a module-level dict for
the life of the process. Session 3/4 (training, prediction) look datasets
up by the same data_id.

Not persisted across restarts -- that's a deliberate scope limit, not an
oversight.
"""
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import pandas as pd

from app.core.errors import NotFoundError

DataType = Literal["classification", "regression", "clustering"]


@dataclass
class DatasetRecord:
    data_id: str
    df: pd.DataFrame
    data_type: DataType
    target_column: str | None
    source: str  # original filename, or "sample:<id>"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_STORE: dict[str, DatasetRecord] = {}


def save_dataset(
    df: pd.DataFrame,
    data_type: DataType,
    target_column: str | None,
    source: str,
) -> DatasetRecord:
    data_id = str(uuid.uuid4())
    record = DatasetRecord(
        data_id=data_id,
        df=df,
        data_type=data_type,
        target_column=target_column,
        source=source,
    )
    _STORE[data_id] = record
    return record


def get_dataset(data_id: str) -> DatasetRecord:
    record = _STORE.get(data_id)
    if record is None:
        raise NotFoundError(
            f"No dataset found for data_id '{data_id}'.",
            details={"data_id": data_id},
        )
    return record


def clear() -> None:
    """Test-only: reset the store between tests."""
    _STORE.clear()
