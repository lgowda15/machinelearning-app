"""Throwaway regressor reference. Exercises the `regressor` branch of the
metrics switch (MSE, RMSE, R^2, MAE) end to end. Not a real submission --
do not pattern-match a group model against this file."""
import time

import numpy as np
from sklearn.linear_model import LinearRegression

from models.base_model import BaseModel


class RefLinearRegressionModel(BaseModel):
    """Linear Regression (conformance reference)."""

    def __init__(self, fit_intercept: bool = True, **kwargs):
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        # No random_state: ordinary least squares has no stochastic
        # component, so determinism holds without seeding anything.
        self._model = LinearRegression(fit_intercept=fit_intercept)
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RefLinearRegressionModel":
        if y is None:
            raise ValueError(
                "RefLinearRegressionModel is supervised; y must not be None."
            )
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")

        t0 = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, "
                f"got {X.shape[1]}."
            )
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return None

    def get_metadata(self) -> dict:
        return {
            "model_name": "Linear Regression",
            "model_type": "regressor",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None if not self.is_fitted else dict(zip(
                [f"feature_{i}" for i in range(self.n_features)],
                np.abs(self._model.coef_).tolist(),
            )),
        }
