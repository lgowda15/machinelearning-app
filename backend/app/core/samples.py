"""Sample datasets shipped with the app so the demo runs without a file
(BUILD_SESSIONS.md Session 2). Scikit-learn built-ins, one per Stage-1 data
type (DATA_FLOW_GUIDE.md SS2) so every path is demonstrable:

- iris        -- classification (3-class target)
- diabetes    -- regression (continuous target)
- blobs       -- clustering (synthetic, no target column)

Student groups never supply datasets; these are the only non-uploaded data
the app ever serves.
"""
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_blobs

from app.core.errors import NotFoundError
from app.schemas.data import DataType

RANDOM_STATE = 42


@dataclass
class SampleDataset:
    id: str
    name: str
    description: str
    data_type: DataType
    loader: Callable[[], tuple[pd.DataFrame, str | None]]


def _load_iris() -> tuple[pd.DataFrame, str | None]:
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(data.target_names)))
    return df, "species"


def _load_diabetes() -> tuple[pd.DataFrame, str | None]:
    data = load_diabetes(as_frame=True)
    return data.frame, "target"


def _load_blobs() -> tuple[pd.DataFrame, str | None]:
    X, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=RANDOM_STATE)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return df, None


_MANIFEST: dict[str, SampleDataset] = {
    "iris": SampleDataset(
        id="iris",
        name="Iris",
        description="Classic 3-class flower measurements; classification target 'species'.",
        data_type="classification",
        loader=_load_iris,
    ),
    "diabetes": SampleDataset(
        id="diabetes",
        name="Diabetes",
        description="Ten baseline health measurements predicting disease progression a year later.",
        data_type="regression",
        loader=_load_diabetes,
    ),
    "blobs": SampleDataset(
        id="blobs",
        name="Synthetic Blobs",
        description="Synthetic, unlabelled point clusters for the clustering path.",
        data_type="clustering",
        loader=_load_blobs,
    ),
}


def list_samples() -> list[SampleDataset]:
    return list(_MANIFEST.values())


def get_sample(sample_id: str) -> SampleDataset:
    sample = _MANIFEST.get(sample_id)
    if sample is None:
        raise NotFoundError(
            f"No sample dataset '{sample_id}'.",
            details={"sample_id": sample_id, "available": list(_MANIFEST.keys())},
        )
    return sample
