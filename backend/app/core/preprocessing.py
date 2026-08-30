"""Stage 2 -- raw table to model-ready array (DATA_FLOW_GUIDE.md SS3,
.claude/rules/backend.md).

Fixed order: split -> impute -> encode -> scale -> cast float64. Every fit
is on the training split only; the test split (and, at Stage 6, brand-new
data) is only ever transformed. This is the one place "input format" gets
decided once, for every one of the twelve models -- models receive
identical clean float64 arrays and preprocess nothing themselves
(CODING_STANDARDS.md SS4).

Decisions not spelled out in the locked docs, confirmed with the
integration lead for Session 2:
- categorical encoding: one-hot for columns with <= ONE_HOT_MAX_CARDINALITY
  unique values (fit-time, train split only), label-encode above that.
- scaling: StandardScaler applies only to columns that were numeric before
  encoding. Encoded categorical columns (one-hot 0/1, or label codes) pass
  through unscaled.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.errors import DataValidationError
from app.schemas.data import DataType

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
ONE_HOT_MAX_CARDINALITY = 10


@dataclass
class _CategoricalEncoding:
    """One fitted encoding for a single categorical column."""

    column: str
    strategy: str  # "onehot" | "label"
    onehot: OneHotEncoder | None = None
    label_map: dict[str, int] = field(default_factory=dict)  # only for "label"
    unseen_code: int = -1  # code assigned to a category not seen at fit time

    def feature_names(self) -> list[str]:
        if self.strategy == "onehot":
            return [f"{self.column}={category}" for category in self.onehot.categories_[0]]
        return [self.column]

    def transform(self, series: pd.Series) -> np.ndarray:
        values = series.astype(str).to_numpy().reshape(-1, 1)
        if self.strategy == "onehot":
            return self.onehot.transform(values)
        codes = np.array(
            [self.label_map.get(v, self.unseen_code) for v in values.ravel()],
            dtype=np.float64,
        )
        return codes.reshape(-1, 1)


@dataclass
class FittedPreprocessors:
    """Everything needed to transform new data without refitting -- Stage 6
    reuses this instead of touching the training split again."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_imputer: SimpleImputer | None
    categorical_imputer: SimpleImputer | None
    encodings: dict[str, _CategoricalEncoding]
    scaler: StandardScaler | None
    feature_names: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the already-fitted pipeline to new data (no fitting)."""
        expected = set(self.numeric_columns) | set(self.categorical_columns)
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise DataValidationError(
                "Columns do not match the columns this model was trained on.",
                details={
                    "missing_columns": sorted(missing),
                    "unexpected_columns": sorted(extra),
                },
            )

        pieces = []

        if self.numeric_columns:
            numeric = self.numeric_imputer.transform(df[self.numeric_columns])
            numeric = self.scaler.transform(numeric)
            pieces.append(numeric)

        if self.categorical_columns:
            # The imputer was fit jointly on all categorical columns, so it
            # must be applied to all of them at once, not one at a time.
            imputed = self.categorical_imputer.transform(df[self.categorical_columns])
            categorical_df = pd.DataFrame(
                imputed, columns=self.categorical_columns, index=df.index,
            )
            for column in self.categorical_columns:
                pieces.append(self.encodings[column].transform(categorical_df[column]))

        if not pieces:
            raise DataValidationError("Dataset has no columns to build features from.")

        return np.hstack(pieces).astype(np.float64)


@dataclass
class PreprocessingResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray | None
    y_test: np.ndarray | None
    feature_names: list[str]
    fitted: FittedPreprocessors


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def _fit_categorical_encoding(column: str, train_series: pd.Series) -> _CategoricalEncoding:
    values = train_series.astype(str)
    categories = values.unique().tolist()

    if len(categories) <= ONE_HOT_MAX_CARDINALITY:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(values.to_numpy().reshape(-1, 1))
        return _CategoricalEncoding(column=column, strategy="onehot", onehot=encoder)

    label_map = {category: i for i, category in enumerate(sorted(categories))}
    return _CategoricalEncoding(
        column=column,
        strategy="label",
        label_map=label_map,
        unseen_code=len(label_map),
    )


def fit_transform_split(
    df: pd.DataFrame,
    target_column: str | None,
    data_type: DataType,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessingResult:
    """Split -> impute -> encode -> scale -> cast float64.

    target_column is None for clustering (fitted with y=None per the
    contract exceptions); every column is then treated as a feature.
    """
    if target_column is not None:
        X_df = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X_df = df.copy()
        y = None

    stratify = y if data_type == "classification" else None
    try:
        if y is not None:
            X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
                X_df, y, test_size=test_size, random_state=random_state, stratify=stratify,
            )
        else:
            X_train_df, X_test_df = train_test_split(
                X_df, test_size=test_size, random_state=random_state,
            )
            y_train_raw = y_test_raw = None
    except ValueError as exc:
        raise DataValidationError(
            f"Could not split the data at test_size={test_size}: {exc}",
            details={"test_size": test_size, "data_type": data_type},
        ) from exc

    numeric_columns, categorical_columns = _split_columns(X_df)

    numeric_imputer = None
    numeric_train = np.empty((len(X_train_df), 0))
    numeric_test = np.empty((len(X_test_df), 0))
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        numeric_train = numeric_imputer.fit_transform(X_train_df[numeric_columns])
        numeric_test = numeric_imputer.transform(X_test_df[numeric_columns])

    categorical_imputer = None
    train_categorical_df = X_train_df[categorical_columns].copy()
    test_categorical_df = X_test_df[categorical_columns].copy()
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        train_categorical_df[:] = categorical_imputer.fit_transform(train_categorical_df)
        test_categorical_df[:] = categorical_imputer.transform(test_categorical_df)

    encodings: dict[str, _CategoricalEncoding] = {
        column: _fit_categorical_encoding(column, train_categorical_df[column])
        for column in categorical_columns
    }

    scaler = None
    if numeric_columns:
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_train)
        numeric_test = scaler.transform(numeric_test)

    train_pieces = [numeric_train] if numeric_columns else []
    test_pieces = [numeric_test] if numeric_columns else []
    for column in categorical_columns:
        train_pieces.append(encodings[column].transform(train_categorical_df[column]))
        test_pieces.append(encodings[column].transform(test_categorical_df[column]))

    if not train_pieces:
        raise DataValidationError("Dataset has no columns to build features from.")

    X_train = np.hstack(train_pieces).astype(np.float64)
    X_test = np.hstack(test_pieces).astype(np.float64)

    feature_names = list(numeric_columns)
    for column in categorical_columns:
        feature_names.extend(encodings[column].feature_names())

    y_train = y_train_raw.to_numpy() if y_train_raw is not None else None
    y_test = y_test_raw.to_numpy() if y_test_raw is not None else None

    fitted = FittedPreprocessors(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        encodings=encodings,
        scaler=scaler,
        feature_names=feature_names,
    )

    return PreprocessingResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        fitted=fitted,
    )


def transform_new(df: pd.DataFrame, fitted: FittedPreprocessors) -> np.ndarray:
    """Stage 6: transform brand-new data with an already-trained model's
    fitted imputer/encoder/scaler. Never refits. Rejects a column mismatch
    before the model ever sees the data."""
    return fitted.transform(df)
