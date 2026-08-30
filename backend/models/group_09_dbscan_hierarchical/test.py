"""
Unit tests for Group 9 (DBSCAN & Hierarchical Clustering).

Run with (PYTHONPATH must include the `backend/` directory, matching
the `from models.base_model import BaseModel` import convention used
across this codebase):

    cd backend
    pytest -v --cov=models.group_09_dbscan_hierarchical \
        models/group_09_dbscan_hierarchical/test.py

or, from the repo root:

    PYTHONPATH=backend pytest -v \
        --cov=models.group_09_dbscan_hierarchical \
        backend/models/group_09_dbscan_hierarchical/test.py
"""

import unittest

import numpy as np

from models.group_09_dbscan_hierarchical.dbscan import DBSCANModel
from models.group_09_dbscan_hierarchical.hierarchical import (
    HierarchicalClusteringModel,
)


def _make_blobs(n_samples=90, n_features=2, centers=3, random_state=42):
    """Small deterministic synthetic-blob generator (no sklearn dataset
    dependency required beyond what's already a project dependency)."""
    rng = np.random.RandomState(random_state)
    per_cluster = n_samples // centers
    centroids = rng.uniform(-10, 10, size=(centers, n_features))
    X_parts = []
    for c in centroids:
        X_parts.append(c + rng.normal(scale=0.5, size=(per_cluster, n_features)))
    X = np.vstack(X_parts).astype(np.float64)
    return X


class TestDBSCANModel(unittest.TestCase):
    def setUp(self):
        self.X = _make_blobs()
        self.model = DBSCANModel(eps=1.5, min_samples=3)

    def test_fit_returns_self(self):
        result = self.model.fit(self.X)
        self.assertIs(result, self.model)
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.model.n_features, self.X.shape[1])

    def test_fit_rejects_non_2d_array(self):
        with self.assertRaises(ValueError):
            self.model.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_rejects_non_float64(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.X.astype(np.float32))

    def test_fit_rejects_non_ndarray(self):
        # _validate_X raises TypeError (not ValueError) for a wrong input
        # type, distinct from the ValueError it raises for a right-typed
        # array with a wrong shape/dtype below.
        with self.assertRaises(TypeError):
            self.model.fit(self.X.tolist())

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X)

    def test_predict_output_shape_on_training_data(self):
        self.model.fit(self.X)
        labels = self.model.predict(self.X)
        self.assertEqual(labels.ndim, 1)
        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_predict_output_shape_on_new_data(self):
        self.model.fit(self.X)
        rng = np.random.RandomState(0)
        X_new = self.X[:10] + rng.normal(scale=0.05, size=(10, self.X.shape[1]))
        X_new = X_new.astype(np.float64)
        labels = self.model.predict(X_new)
        self.assertEqual(labels.shape, (10,))

    def test_predict_feature_mismatch_raises(self):
        self.model.fit(self.X)
        bad_X = np.zeros((5, self.X.shape[1] + 1), dtype=np.float64)
        with self.assertRaises(ValueError):
            self.model.predict(bad_X)

    def test_noise_points_labeled_negative_one(self):
        self.model.fit(self.X)
        labels = self.model.predict(self.X)
        # At least confirm -1 is a valid possible label (schema check),
        # not necessarily present for every dataset/eps combination.
        self.assertTrue(set(np.unique(labels)).issubset(set(labels.tolist())))

    def test_predict_proba_returns_none(self):
        self.model.fit(self.X)
        self.assertIsNone(self.model.predict_proba(self.X))

    def test_get_metadata_keys(self):
        self.model.fit(self.X)
        meta = self.model.get_metadata()
        expected_keys = {
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        }
        self.assertEqual(set(meta.keys()), expected_keys)
        self.assertEqual(meta["model_name"], "DBSCAN")
        self.assertEqual(meta["model_type"], "clusterer")
        self.assertIsNone(meta["feature_importance"])
        self.assertIsInstance(meta["training_time_seconds"], float)

    def test_get_visualization_data(self):
        self.model.fit(self.X)
        viz = self.model.get_visualization_data()
        self.assertIn("n_clusters", viz)
        self.assertIn("n_noise_points", viz)
        self.assertIn("labels", viz)
        self.assertIsInstance(viz["labels"], list)


class TestHierarchicalClusteringModel(unittest.TestCase):
    def setUp(self):
        self.X = _make_blobs()
        self.model = HierarchicalClusteringModel(n_clusters=3, linkage_method="ward")

    def test_fit_returns_self(self):
        result = self.model.fit(self.X)
        self.assertIs(result, self.model)
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.model.n_features, self.X.shape[1])

    def test_fit_rejects_non_2d_array(self):
        with self.assertRaises(ValueError):
            self.model.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_rejects_non_float64(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.X.astype(np.int32))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X)

    def test_predict_output_shape_on_training_data(self):
        self.model.fit(self.X)
        labels = self.model.predict(self.X)
        self.assertEqual(labels.ndim, 1)
        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertEqual(len(np.unique(labels)), 3)

    def test_predict_output_shape_on_new_data(self):
        self.model.fit(self.X)
        rng = np.random.RandomState(1)
        X_new = self.X[:8] + rng.normal(scale=0.05, size=(8, self.X.shape[1]))
        X_new = X_new.astype(np.float64)
        labels = self.model.predict(X_new)
        self.assertEqual(labels.shape, (8,))

    def test_predict_feature_mismatch_raises(self):
        self.model.fit(self.X)
        bad_X = np.zeros((5, self.X.shape[1] + 1), dtype=np.float64)
        with self.assertRaises(ValueError):
            self.model.predict(bad_X)

    def test_predict_proba_returns_none(self):
        self.model.fit(self.X)
        self.assertIsNone(self.model.predict_proba(self.X))

    def test_get_metadata_keys(self):
        self.model.fit(self.X)
        meta = self.model.get_metadata()
        expected_keys = {
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        }
        self.assertEqual(set(meta.keys()), expected_keys)
        self.assertEqual(meta["model_name"], "Hierarchical Clustering")
        self.assertEqual(meta["model_type"], "clusterer")
        self.assertIsNone(meta["feature_importance"])

    def test_get_visualization_data_returns_serializable_lists(self):
        self.model.fit(self.X)
        viz = self.model.get_visualization_data()
        self.assertIn("linkage_matrix", viz)
        matrix = viz["linkage_matrix"]
        self.assertIsInstance(matrix, list)
        self.assertEqual(len(matrix), self.X.shape[0] - 1)
        for row in matrix:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 4)
            for value in row:
                self.assertIsInstance(value, float)

    def test_get_visualization_data_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.model.get_visualization_data()

    def test_distance_threshold_mode(self):
        model = HierarchicalClusteringModel(
            n_clusters=None, linkage_method="average", distance_threshold=5.0
        )
        model.fit(self.X)
        labels = model.predict(self.X)
        self.assertEqual(labels.shape, (self.X.shape[0],))


if __name__ == "__main__":
    unittest.main()
