"""Throwaway clusterer reference. Exercises the `clusterer` branch of the
metrics switch (silhouette, Davies-Bouldin, inertia) end to end. Not a real
submission -- do not pattern-match a group model against this file."""
import time

import numpy as np
from sklearn.cluster import KMeans

from models.base_model import BaseModel


class RefKMeansModel(BaseModel):
    """K-Means clusterer (conformance reference)."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.random_state = random_state
        self._model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init="auto",
        )
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RefKMeansModel":
        # Clusterers are fitted with y=None (CLAUDE.md contract exceptions);
        # y is accepted for interface uniformity and never used.
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")

        t0 = time.perf_counter()
        self._model.fit(X)
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
            "model_name": "K-Means",
            "model_type": "clusterer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }
