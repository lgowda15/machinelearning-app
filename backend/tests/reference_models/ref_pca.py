"""Throwaway dimensionality-reduction reference. Exercises the
`dimensionality_reducer` branch of the metrics switch (explained variance
ratio) end to end, including the 2D predict-output exception. Not a real
submission -- do not pattern-match a group model against this file."""
import time

import numpy as np
from sklearn.decomposition import PCA

from models.base_model import BaseModel


class RefPCAModel(BaseModel):
    """PCA (conformance reference)."""

    def __init__(self, n_components: int = 2, random_state: int = 42, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.random_state = random_state
        self._model = PCA(n_components=n_components, random_state=random_state)
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RefPCAModel":
        # Reducers are fitted with y=None (CLAUDE.md contract exceptions);
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
        """Return the transformed matrix, shape (n_samples, n_components).

        Contract exception (CODING_STANDARDS.md S7): dimensionality
        reducers return a 2D array here, not 1D. Code downstream that
        assumes 1D predict output is wrong, not this method.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, "
                f"got {X.shape[1]}."
            )
        return self._model.transform(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return None

    def get_metadata(self) -> dict:
        return {
            "model_name": "PCA",
            "model_type": "dimensionality_reducer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }

    def get_visualization_data(self) -> dict | None:
        if not self.is_fitted:
            return None
        return {
            "explained_variance_ratio": (
                self._model.explained_variance_ratio_.tolist()
            ),
        }
