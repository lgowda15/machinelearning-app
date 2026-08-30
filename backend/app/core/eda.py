"""Stage 1 -- raw table validation, data-type inference, and EDA profiling
(DATA_FLOW_GUIDE.md SS2, CODING_STANDARDS.md SS4 upload rejects).

Nothing here cleans the data. This is what powers the EDA screen: summary
statistics, missing-value counts, and distribution charts computed
directly on the raw table, before Stage 2 preprocessing touches anything.
"""
import numpy as np
import pandas as pd

from app.core.errors import DataValidationError
from app.core.imbalance import detect_imbalance
from app.schemas.data import (
    CategoricalDistribution,
    ColumnSummary,
    DataProfileResponse,
    DataType,
    NumericDistribution,
)

MIN_ROWS = 50
MAX_COLUMNS = 100
NUMERIC_DISTRIBUTION_BINS = 10
CATEGORICAL_TOP_N = 20
MIN_CLASSES = 2
MAX_CLASSES = 10


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    """CODING_STANDARDS.md SS4 / BUILD_SESSIONS.md Session 2 upload rejects."""
    if len(df) < MIN_ROWS:
        raise DataValidationError(
            f"Dataset has {len(df)} rows; at least {MIN_ROWS} are required.",
            details={"n_rows": len(df), "minimum": MIN_ROWS},
        )
    if len(df.columns) > MAX_COLUMNS:
        raise DataValidationError(
            f"Dataset has {len(df.columns)} columns; at most {MAX_COLUMNS} are allowed.",
            details={"n_columns": len(df.columns), "maximum": MAX_COLUMNS},
        )
    if df.select_dtypes(include=np.number).shape[1] == 0:
        raise DataValidationError(
            "Dataset has no numeric columns.",
            details={"columns": list(df.columns)},
        )


def resolve_target_column(
    df: pd.DataFrame, target_column: str | None, has_target: bool
) -> str | None:
    """Named by the user, defaulted to the last column, or explicitly absent
    (clustering) -- DATA_FLOW_GUIDE.md SS2."""
    if not has_target:
        return None
    if target_column is not None:
        if target_column not in df.columns:
            raise DataValidationError(
                f"Target column '{target_column}' not found in the dataset.",
                details={"target_column": target_column, "columns": list(df.columns)},
            )
        return target_column
    return df.columns[-1]


def infer_data_type(df: pd.DataFrame, target_column: str | None) -> DataType:
    """DATA_FLOW_GUIDE.md SS2:
    - no target given -> clustering
    - target categorical, 2-10 unique values -> classification
    - target continuous, more than 10 unique values -> regression
    """
    if target_column is None:
        return "clustering"

    n_unique = df[target_column].nunique(dropna=True)
    if n_unique < MIN_CLASSES:
        raise DataValidationError(
            f"Target column '{target_column}' has {n_unique} unique value(s); "
            f"at least {MIN_CLASSES} are required to infer classification or regression.",
            details={"target_column": target_column, "unique_values": int(n_unique)},
        )
    if n_unique <= MAX_CLASSES:
        return "classification"
    return "regression"


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _numeric_distribution(series: pd.Series) -> NumericDistribution:
    clean = series.dropna()
    if clean.empty:
        return NumericDistribution(bin_edges=[], counts=[])
    counts, edges = np.histogram(clean.to_numpy(dtype=float), bins=NUMERIC_DISTRIBUTION_BINS)
    return NumericDistribution(bin_edges=edges.tolist(), counts=counts.tolist())


def _categorical_distribution(series: pd.Series) -> CategoricalDistribution:
    clean = series.dropna().astype(str)
    value_counts = clean.value_counts()
    top = value_counts.iloc[:CATEGORICAL_TOP_N]
    other_count = int(value_counts.iloc[CATEGORICAL_TOP_N:].sum())
    return CategoricalDistribution(
        categories=top.index.tolist(),
        counts=[int(c) for c in top.to_numpy()],
        other_count=other_count,
    )


def _profile_column(df: pd.DataFrame, column: str, target_column: str | None) -> ColumnSummary:
    series = df[column]
    n = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = (missing_count / n) if n else 0.0
    is_target = column == target_column

    if _is_numeric(series):
        clean = series.dropna()
        return ColumnSummary(
            name=column,
            dtype="numeric",
            is_target=is_target,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=int(series.nunique(dropna=True)),
            distribution=_numeric_distribution(series),
            mean=float(clean.mean()) if not clean.empty else None,
            std=float(clean.std()) if len(clean) > 1 else None,
            min=float(clean.min()) if not clean.empty else None,
            max=float(clean.max()) if not clean.empty else None,
            median=float(clean.median()) if not clean.empty else None,
        )

    clean = series.dropna().astype(str)
    top_value = None
    top_value_freq = None
    if not clean.empty:
        value_counts = clean.value_counts()
        top_value = str(value_counts.index[0])
        top_value_freq = int(value_counts.iloc[0])
    return ColumnSummary(
        name=column,
        dtype="categorical",
        is_target=is_target,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=int(series.nunique(dropna=True)),
        distribution=_categorical_distribution(series),
        top_value=top_value,
        top_value_freq=top_value_freq,
    )


def profile_dataset(
    data_id: str,
    df: pd.DataFrame,
    data_type: DataType,
    target_column: str | None,
    source: str,
) -> DataProfileResponse:
    columns = [_profile_column(df, column, target_column) for column in df.columns]

    class_imbalance = None
    if data_type == "classification" and target_column is not None:
        class_imbalance = detect_imbalance(df[target_column])

    return DataProfileResponse(
        data_id=data_id,
        source=source,
        n_rows=len(df),
        n_columns=len(df.columns),
        data_type=data_type,
        target_column=target_column,
        columns=columns,
        class_imbalance=class_imbalance,
    )
