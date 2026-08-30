"""Validation helpers shared by the LDA/QDA project modules."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_X(
    X: NDArray[np.float64],
    *,
    expected_features: int | None = None,
) -> NDArray[np.float64]:
    """Validate the backend's preprocessed feature-matrix contract."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray.")
    if X.dtype != np.float64:
        raise TypeError(f"X must have dtype float64, got {X.dtype}.")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            f"Model trained on {expected_features} features, got {X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError(
            "X must contain only finite values; impute missing values upstream."
        )
    return X


def validate_y(y: NDArray[Any] | None, *, n_samples: int) -> NDArray[Any]:
    """Validate a classification target without encoding or changing its dtype."""
    if y is None:
        raise ValueError("LDA/QDA are supervised classifiers; y must not be None.")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}.")
    if y.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} rows, y has {y.shape[0]}.")
    if y.dtype.kind in {"c", "m", "M", "V"}:
        raise TypeError(f"Unsupported target dtype: {y.dtype}.")
    if _contains_missing_target(y):
        raise ValueError("y must not contain missing or non-finite values.")
    return y


def validate_training_data(
    X: NDArray[np.float64],
    y: NDArray[Any] | None,
) -> tuple[NDArray[np.float64], NDArray[Any], NDArray[Any], NDArray[np.intp]]:
    """Validate model training data and return class information."""
    X_valid = validate_X(X)
    y_valid = validate_y(y, n_samples=X_valid.shape[0])
    try:
        classes, counts = np.unique(y_valid, return_counts=True)
    except TypeError as exc:
        raise ValueError("y must contain mutually comparable class labels.") from exc
    if classes.size < 2:
        raise ValueError("LDA/QDA require at least two target classes.")
    if np.any(counts < 2):
        raise ValueError("Every target class must contain at least two samples.")
    return X_valid, y_valid, classes, counts


def require_fitted(is_fitted: bool, *, operation: str) -> None:
    """Raise a platform-friendly error when inference precedes fitting."""
    if not is_fitted:
        raise RuntimeError(f"Call fit() before {operation}().")


def to_python_scalar(value: Any) -> Any:
    """Convert a numpy scalar to a JSON-compatible Python scalar."""
    converted = value.item() if isinstance(value, np.generic) else value
    if isinstance(converted, bytes):
        return converted.decode("utf-8", errors="replace")
    return converted


def _contains_missing_target(y: NDArray[Any]) -> bool:
    if y.dtype.kind in {"f"}:
        return not np.isfinite(y).all()
    if y.dtype.kind in {"i", "u", "b", "S", "U"}:
        return False
    for value in y:
        if value is None:
            return True
        if isinstance(value, float | np.floating) and not np.isfinite(value):
            return True
    return False
