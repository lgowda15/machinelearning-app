"""Standards-compliant Linear Discriminant Analysis model."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.base_model import BaseModel

from ._validation import require_fitted, validate_training_data, validate_X


class LDAModel(BaseModel):
    """Linear Discriminant Analysis for preprocessed tabular classification data."""

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: str | float | None = None,
        priors: list[float] | None = None,
        n_components: int | None = None,
        store_covariance: bool = False,
        tol: float = 1e-4,
    ) -> None:
        _validate_hyperparameters(solver, shrinkage, n_components, tol)
        stored_priors = None if priors is None else list(priors)
        super().__init__(
            solver=solver,
            shrinkage=shrinkage,
            priors=stored_priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
        )
        model_priors = None if priors is None else np.asarray(priors, dtype=np.float64)
        self._model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            priors=model_priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
        )
        self.classes_: NDArray[Any] | None = None
        self._label_dtype: np.dtype[Any] | None = None
        self._train_time: float | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any] | None = None,
    ) -> "LDAModel":
        """Fit LDA without changing the backend-preprocessed inputs."""
        X_valid, y_valid, _, _ = validate_training_data(X, y)
        started = time.perf_counter()
        try:
            self._model.fit(X_valid, y_valid)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "LDA covariance estimation failed; inspect collinearity "
                "and sample size."
            ) from exc
        self._train_time = time.perf_counter() - started
        self.is_fitted = True
        self.n_features = X_valid.shape[1]
        self._label_dtype = y_valid.dtype
        self.classes_ = self._model.classes_.astype(self._label_dtype, copy=False)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[Any]:
        """Return labels with exactly the training target's numpy dtype."""
        require_fitted(self.is_fitted, operation="predict")
        X_valid = validate_X(X, expected_features=self.n_features)
        predictions = self._model.predict(X_valid)
        return predictions.astype(self._label_dtype, copy=False)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return normalized probabilities in ``classes_`` order."""
        require_fitted(self.is_fitted, operation="predict_proba")
        X_valid = validate_X(X, expected_features=self.n_features)
        return self._model.predict_proba(X_valid)

    def get_metadata(self) -> dict[str, Any]:
        """Return exactly the metadata keys required by the platform."""
        importance = None
        if self.is_fitted:
            coefficients = np.atleast_2d(self._model.coef_)
            scores = np.mean(np.abs(coefficients), axis=0)
            importance = {
                f"feature_{index}": float(score) for index, score in enumerate(scores)
            }
        return {
            "model_name": "Linear Discriminant Analysis",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": importance,
        }


def _validate_hyperparameters(
    solver: str,
    shrinkage: str | float | None,
    n_components: int | None,
    tol: float,
) -> None:
    if solver not in {"svd", "lsqr", "eigen"}:
        raise ValueError("solver must be 'svd', 'lsqr', or 'eigen'.")
    if solver == "svd" and shrinkage is not None:
        raise ValueError("shrinkage is unsupported when solver='svd'.")
    if isinstance(shrinkage, str) and shrinkage != "auto":
        raise ValueError("String shrinkage must be 'auto'.")
    if isinstance(shrinkage, int | float) and not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("Numeric shrinkage must be between 0 and 1.")
    if n_components is not None and n_components < 1:
        raise ValueError("n_components must be a positive integer or None.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
