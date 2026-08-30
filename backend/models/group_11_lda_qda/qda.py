"""Standards-compliant Quadratic Discriminant Analysis model."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from models.base_model import BaseModel

from ._validation import require_fitted, validate_training_data, validate_X


class QDAModel(BaseModel):
    """Quadratic Discriminant Analysis for preprocessed tabular data."""

    def __init__(
        self,
        priors: list[float] | None = None,
        reg_param: float = 0.0,
        store_covariance: bool = False,
        tol: float = 1e-4,
    ) -> None:
        if not 0.0 <= reg_param <= 1.0:
            raise ValueError("reg_param must be between 0 and 1.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        stored_priors = None if priors is None else list(priors)
        super().__init__(
            priors=stored_priors,
            reg_param=reg_param,
            store_covariance=store_covariance,
            tol=tol,
        )
        model_priors = None if priors is None else np.asarray(priors, dtype=np.float64)
        self._model = QuadraticDiscriminantAnalysis(
            priors=model_priors,
            reg_param=reg_param,
            store_covariance=store_covariance,
            tol=tol,
        )
        self.classes_: NDArray[Any] | None = None
        self._label_dtype: np.dtype[Any] | None = None
        self._train_time: float | None = None
        self._reg_param = reg_param

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any] | None = None,
    ) -> "QDAModel":
        """Fit QDA without changing the backend-preprocessed inputs."""
        X_valid, y_valid, _, counts = validate_training_data(X, y)
        if self._reg_param == 0.0 and np.min(counts) <= X_valid.shape[1]:
            raise ValueError(
                "Unregularized QDA requires more samples in every class than "
                "features; provide more data or use a positive reg_param."
            )
        started = time.perf_counter()
        try:
            self._model.fit(X_valid, y_valid)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "QDA covariance estimation failed; inspect per-class sample size."
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
        return {
            "model_name": "Quadratic Discriminant Analysis",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }
