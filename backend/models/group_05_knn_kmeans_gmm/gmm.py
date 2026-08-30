"""Gaussian Mixture Model clustering — Group 5."""

import time

import numpy as np
from sklearn.mixture import GaussianMixture

from models.base_model import BaseModel


class GMMModel(BaseModel):
    """Gaussian Mixture Model clusterer.
    Wraps sklearn GaussianMixture. Receives preprocessed float64 input
    from the backend (Section 4 of the coding standards). Unsupervised:
    y is ignored if supplied.

    Note on predict_proba: per the coding standards, predict_proba
    returns probabilities only for model_type == "classifier"; this
    model's type is "clusterer", so predict_proba returns None even
    though GMM internally computes soft posterior probabilities
    (available via predict() as hard argmax assignments).
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            **kwargs,
        )
        self._model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GMMModel":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] < self.hyperparams["n_components"]:
            raise ValueError(
                f"n_components={self.hyperparams['n_components']} requires at "
                f"least that many samples; got {X.shape[0]}."
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
        # model_type is "clusterer" -> spec requires None here.
        return None

    def get_metadata(self) -> dict:
        return {
            "model_name": "Gaussian Mixture Model",
            "model_type": "clusterer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,  # not applicable to clustering
        }
