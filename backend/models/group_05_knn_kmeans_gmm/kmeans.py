"""K-Means clustering — Group 5."""

import time

import numpy as np
from sklearn.cluster import KMeans

from models.base_model import BaseModel


class KMeansModel(BaseModel):
    """K-Means clusterer.
    
    Wraps sklearn KMeans. Receives preprocessed float64 input from the
    backend (Section 4 of the coding standards). Unsupervised: y is
    ignored if supplied.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_init: int = 10,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_clusters=n_clusters, n_init=n_init, random_state=random_state, **kwargs
        )
        self._model = KMeans(
            n_clusters=n_clusters, n_init=n_init, random_state=random_state
        )
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KMeansModel":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] < self.hyperparams["n_clusters"]:
            raise ValueError(
                f"n_clusters={self.hyperparams['n_clusters']} requires at least "
                f"that many samples; got {X.shape[0]}."
            )

        t0 = time.perf_counter()
        self._model.fit(X)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
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
        # K-Means is a hard-assignment clusterer; no probability estimates.
        return None

    def get_metadata(self) -> dict:
        return {
            "model_name": "K-Means",
            "model_type": "clusterer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,  # not applicable to clustering
        }
