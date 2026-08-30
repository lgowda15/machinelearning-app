"""
DBSCAN clustering model wrapper.

Course   : UM25MB653CA2
Group    : Group 9 (DBSCAN & Hierarchical Clustering)
Model    : Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

This module wraps `sklearn.cluster.DBSCAN` behind the enterprise
`BaseModel` contract. DBSCAN has no native `predict()` for unseen data
(scikit-learn only exposes `fit_predict`), so this wrapper implements a
deterministic nearest-core-sample assignment strategy for inference on
new data, while returning the exact training-time labels when the
input matches the data the model was fit on.

No data preprocessing/scaling is performed inside this model — callers
are responsible for supplying already-preprocessed features.
"""

import time

import numpy as np
from sklearn.cluster import DBSCAN

from models.base_model import BaseModel

# Sentinel label used by scikit-learn (and this wrapper) to mark noise /
# outlier points that do not belong to any dense cluster.
NOISE_LABEL = -1


class DBSCANModel(BaseModel):
    """Enterprise wrapper around `sklearn.cluster.DBSCAN`."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int | None = None,
        **kwargs,
    ):
        hyperparams = {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "n_jobs": n_jobs,
        }
        hyperparams.update(kwargs)
        super().__init__(**hyperparams)

        self._model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )

        # Training-time artifacts, populated in fit()
        self._train_time = None
        self._X_train = None
        self._train_labels = None
        self._core_sample_indices = None
        self._core_samples = None
        self._core_sample_labels = None

    @staticmethod
    def _validate_X(X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if X.dtype != np.float64:
            raise ValueError(f"X must have dtype float64, got {X.dtype}.")
        return X

    def fit(self, X, y=None):
        """Fit DBSCAN on X. y is accepted for API compatibility and ignored."""
        X = self._validate_X(X)

        start = time.perf_counter()
        self._model.fit(X)
        self._train_time = time.perf_counter() - start

        self._X_train = X
        self._train_labels = self._model.labels_.astype(np.int64)
        self._core_sample_indices = self._model.core_sample_indices_

        if self._core_sample_indices.size > 0:
            self._core_samples = X[self._core_sample_indices]
            self._core_sample_labels = self._train_labels[self._core_sample_indices]
        else:
            self._core_samples = np.empty((0, X.shape[1]), dtype=np.float64)
            self._core_sample_labels = np.empty((0,), dtype=np.int64)

        self.is_fitted = True
        self.n_features = X.shape[1]
        return self

    def predict(self, X):
        """
        Assign cluster labels to X.

        If X is identical to the training data (same shape and values),
        the exact training-time labels are returned. Otherwise, each
        point is assigned to the cluster of its nearest core sample if
        that core sample lies within `eps` distance; points with no
        core sample within `eps` are labeled as noise (-1).
        """
        if not self.is_fitted:
            raise RuntimeError("DBSCANModel.predict() called before fit().")

        X = self._validate_X(X)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features}."
            )

        # Fast path: exact match with training data.
        if (
            X.shape == self._X_train.shape
            and np.array_equal(X, self._X_train)
        ):
            return self._train_labels.copy()

        n_samples = X.shape[0]
        labels = np.full(n_samples, NOISE_LABEL, dtype=np.int64)

        if self._core_samples.shape[0] == 0:
            # No core samples were found during training -> everything is noise.
            return labels

        eps = self._model.eps
        # Compute pairwise distances from each query point to all core samples.
        # Done in chunks to bound memory usage for large inputs.
        chunk_size = 2000
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            diff = chunk[:, np.newaxis, :] - self._core_samples[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            nearest_idx = np.argmin(dists, axis=1)
            nearest_dist = dists[np.arange(dists.shape[0]), nearest_idx]

            within_eps = nearest_dist <= eps
            chunk_labels = np.full(chunk.shape[0], NOISE_LABEL, dtype=np.int64)
            chunk_labels[within_eps] = self._core_sample_labels[
                nearest_idx[within_eps]
            ]
            labels[start_idx:end_idx] = chunk_labels

        return labels

    def predict_proba(self, X):
        """DBSCAN is not a probabilistic model; no class probabilities exist."""
        return 

    def get_metadata(self):
        return {
            "model_name": "DBSCAN",
            "model_type": "clusterer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }

    def get_visualization_data(self):
        """
        Supplementary visualization payload (not required by the base
        contract but useful for downstream dashboards): number of
        clusters found and number of noise points.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "DBSCANModel.get_visualization_data() called before fit()."
            )
        unique_labels = set(self._train_labels.tolist())
        n_clusters = len(unique_labels - {NOISE_LABEL})
        n_noise = int(np.sum(self._train_labels == NOISE_LABEL))
        return {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "labels": self._train_labels.tolist(),
        }
