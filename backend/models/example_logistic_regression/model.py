"""Reference implementation. Logistic Regression is not an allotted topic;
this folder exists so every group has a complete, passing example to
pattern-match against. Do not copy it as your submission."""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier (reference implementation)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 random_state: int = 42, **kwargs):
        super().__init__(C=C, max_iter=max_iter, **kwargs)
        self.random_state = random_state
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state,
        )
        self.classes_ = None
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogisticRegressionModel":
        if y is None:
            raise ValueError(
                "LogisticRegressionModel is supervised; y must not be None."
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
        self.classes_ = self._model.classes_
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
        return self._model.predict_proba(X)

    def get_metadata(self) -> dict:
        return {
            "model_name": "Logistic Regression",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None if not self.is_fitted else dict(zip(
                [f"feature_{i}" for i in range(self.n_features)],
                np.abs(self._model.coef_[0]).tolist(),
            )),
        }