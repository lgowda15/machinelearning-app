"""Compatibility filter -- not recommendation (ARCHITECTURE.md SS5,
CLAUDE.md "Hard rules": no AutoML, no recommendations).

One rule, shared by POST /api/models/compatible and the train route, so an
incompatible model can never reach fit() -- it's rejected up front with a
reason, the same reason either endpoint would show.

Rule, keyed on model_type vs. the dataset's data_type:
- classifier requires a categorical target -> classification only.
- regressor requires a continuous target -> regression only.
- clusterer is fitted with y=None and is semantically for unlabelled data
  -> clustering only. A clusterer on labelled data is incompatible, not an
  error (BUILD_SESSIONS.md Session 3 "Done when").
- dimensionality_reducer is also fitted with y=None, but unlike a
  clusterer it only ever consumes X -- whether labels exist is irrelevant
  to it, so it is compatible with all three data types.
"""
from app.schemas.data import DataType

_COMPATIBLE_DATA_TYPES: dict[str, set[DataType]] = {
    "classifier": {"classification"},
    "regressor": {"regression"},
    "clusterer": {"clustering"},
    "dimensionality_reducer": {"classification", "regression", "clustering"},
}

_REASONS: dict[str, str] = {
    "classifier": "Classifier models require a categorical target column; this dataset is {data_type}.",
    "regressor": "Regressor models require a continuous target column; this dataset is {data_type}.",
    "clusterer": "Clusterer models expect unlabelled data; this dataset is {data_type}.",
}


def incompatibility_reason(model_type: str, data_type: DataType) -> str | None:
    """None if compatible; otherwise a one-sentence reason naming the mismatch."""
    compatible_types = _COMPATIBLE_DATA_TYPES.get(model_type)
    if compatible_types is None:
        raise ValueError(f"Unrecognised model_type: {model_type!r}")
    if data_type in compatible_types:
        return None
    return _REASONS[model_type].format(data_type=data_type)
