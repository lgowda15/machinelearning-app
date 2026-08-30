"""K-Nearest Neighbors classifier — Group 5."""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from models.base_model import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier.
    Wraps sklearn KNeighborsClassifier. Receives preprocessed float64
    input from the backend (Section 4 of the coding standards).
    """

    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        **kwargs,
    ):
        super().__init__(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, **kwargs
        )
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
        )
        self.classes_ = None
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KNNModel":
        if y is None:
            raise ValueError("KNNModel is supervised; y must not be None.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")
        if X.shape[0] <= self.hyperparams["n_neighbors"]:
            raise ValueError(
                f"n_neighbors={self.hyperparams['n_neighbors']} requires more "
                f"than {self.hyperparams['n_neighbors']} training samples; got "
                f"{X.shape[0]}."
            )

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
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        return self._model.predict_proba(X)

    def get_metadata(self) -> dict:
        return {
            "model_name": "K-Nearest Neighbors",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,  # KNN has no native feature importance
        }
