"""
Agglomerative (Hierarchical) clustering model wrapper.

Course   : UM25MB653CA2
Group    : Group 9 (DBSCAN & Hierarchical Clustering)
Model    : Agglomerative Hierarchical Clustering

This module wraps `sklearn.cluster.AgglomerativeClustering` behind the
enterprise `BaseModel` contract, and additionally computes a SciPy
linkage matrix (`scipy.cluster.hierarchy.linkage`) at fit time so that
dendrograms can be rendered downstream without re-running clustering.

No data preprocessing/scaling is performed inside this model — callers
are responsible for supplying already-preprocessed features.
"""

import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

from models.base_model import BaseModel

# Mapping of AgglomerativeClustering linkage methods to their SciPy
# `linkage()` equivalents. SciPy's linkage() only supports Euclidean
# distance for 'centroid', 'median', and 'ward' methods.
_SCIPY_LINKAGE_METHOD = {
    "ward": "ward",
    "complete": "complete",
    "average": "average",
    "single": "single",
}


class HierarchicalClusteringModel(BaseModel):
    """Enterprise wrapper around `sklearn.cluster.AgglomerativeClustering`."""

    def __init__(
        self,
        n_clusters: int = 2,
        linkage_method: str = "ward",
        metric: str = "euclidean",
        distance_threshold: float | None = None,
        **kwargs,
    ):
        hyperparams = {
            "n_clusters": n_clusters,
            "linkage_method": linkage_method,
            "metric": metric,
            "distance_threshold": distance_threshold,
        }
        hyperparams.update(kwargs)
        super().__init__(**hyperparams)

        # AgglomerativeClustering requires n_clusters=None when a
        # distance_threshold is supplied.
        effective_n_clusters = None if distance_threshold is not None else n_clusters

        self._model = AgglomerativeClustering(
            n_clusters=effective_n_clusters,
            linkage=linkage_method,
            metric=metric,
            distance_threshold=distance_threshold,
        )

        self.linkage_method = linkage_method
        self.metric = metric
        self._train_time = None
        self._X_train = None
        self._train_labels = None
        self._linkage_matrix = None

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
        """Fit AgglomerativeClustering on X. y is accepted and ignored."""
        X = self._validate_X(X)

        start = time.perf_counter()
        self._model.fit(X)
        scipy_method = _SCIPY_LINKAGE_METHOD.get(self.linkage_method, "average")
        self._linkage_matrix = linkage(X, method=scipy_method, metric=self.metric)
        self._train_time = time.perf_counter() - start

        self._X_train = X
        self._train_labels = self._model.labels_.astype(np.int64)
        self.is_fitted = True
        self.n_features = X.shape[1]
        return self

    def predict(self, X):
        """
        Return cluster labels for X.

        AgglomerativeClustering does not support inference on unseen
        data. When X is identical to the training data, the exact
        fitted labels are returned. Otherwise, each new point is
        assigned to the cluster of its nearest training sample
        (1-nearest-neighbor rule), the standard practical approach for
        extending hierarchical clustering assignments to new points.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "HierarchicalClusteringModel.predict() called before fit()."
            )

        X = self._validate_X(X)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features}."
            )

        # Fast path: exact match with training data.
        if X.shape == self._X_train.shape and np.array_equal(X, self._X_train):
            return self._train_labels.copy()

        return self._assign_nearest_neighbor(X)

    def _assign_nearest_neighbor(self, X):
        """Assign each row of X to the cluster label of its nearest
        training sample (Euclidean 1-nearest-neighbor), computed in
        chunks to bound memory usage for large inputs."""
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int64)

        chunk_size = 2000
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            diff = chunk[:, np.newaxis, :] - self._X_train[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            nearest_idx = np.argmin(dists, axis=1)
            labels[start_idx:end_idx] = self._train_labels[nearest_idx]

        return labels

    def predict_proba(self, X):
        """Hierarchical clustering is not a probabilistic model."""
        return 

    def get_metadata(self):
        return {
            "model_name": "Hierarchical Clustering",
            "model_type": "clusterer",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }

    def get_visualization_data(self):
        """
        Return the SciPy linkage matrix as a pure JSON-serializable
        list of lists, suitable for client-side dendrogram rendering.

        Format: an (n_samples - 1) x 4 array where each row is
        [idx1, idx2, distance, sample_count], matching the exact
        output contract of `scipy.cluster.hierarchy.linkage`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "HierarchicalClusteringModel.get_visualization_data() "
                "called before fit()."
            )
        return {"linkage_matrix": self._linkage_matrix.tolist()}
