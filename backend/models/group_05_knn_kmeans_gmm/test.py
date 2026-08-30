"""Unit tests for KNN, K-Means, GMM. Minimum 80% coverage."""

import unittest

import numpy as np
from gmm import GMMModel
from kmeans import KMeansModel
from knn import KNNModel


class TestKNNModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100)
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = KNNModel()
        self.assertIs(m.fit(self.X, self.y), m)

    def test_fit_without_y_raises(self):
        with self.assertRaises(ValueError):
            KNNModel().fit(self.X, None)

    def test_predict_shape(self):
        m = KNNModel().fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            KNNModel().predict(self.X_test)

    def test_predict_wrong_feature_count_raises(self):
        m = KNNModel().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            m.predict(self.X_test[:, :2])

    def test_proba_rows_sum_to_one(self):
        m = KNNModel().fit(self.X, self.y)
        p = m.predict_proba(self.X_test)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_determinism(self):
        a = KNNModel().fit(self.X, self.y).predict(self.X_test)
        b = KNNModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = KNNModel().fit(self.X, self.y).get_metadata()
        for k in [
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        ]:
            self.assertIn(k, md)
        self.assertEqual(md["model_type"], "classifier")


class TestKMeansModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = KMeansModel()
        self.assertIs(m.fit(self.X), m)

    def test_fit_ignores_y(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        m = KMeansModel().fit(self.X, y)  # should not raise
        self.assertTrue(m.is_fitted)

    def test_predict_shape(self):
        m = KMeansModel().fit(self.X)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            KMeansModel().predict(self.X_test)

    def test_predict_proba_is_none(self):
        m = KMeansModel().fit(self.X)
        self.assertIsNone(m.predict_proba(self.X_test))

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            KMeansModel(n_clusters=10).fit(self.X[:5])

    def test_determinism(self):
        a = KMeansModel().fit(self.X).predict(self.X_test)
        b = KMeansModel().fit(self.X).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = KMeansModel().fit(self.X).get_metadata()
        for k in [
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        ]:
            self.assertIn(k, md)
        self.assertEqual(md["model_type"], "clusterer")


class TestGMMModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = GMMModel()
        self.assertIs(m.fit(self.X), m)

    def test_predict_shape(self):
        m = GMMModel().fit(self.X)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            GMMModel().predict(self.X_test)

    def test_predict_proba_is_none(self):
        m = GMMModel().fit(self.X)
        self.assertIsNone(m.predict_proba(self.X_test))

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            GMMModel(n_components=10).fit(self.X[:5])

    def test_determinism(self):
        a = GMMModel().fit(self.X).predict(self.X_test)
        b = GMMModel().fit(self.X).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = GMMModel().fit(self.X).get_metadata()
        for k in [
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        ]:
            self.assertIn(k, md)
        self.assertEqual(md["model_type"], "clusterer")


if __name__ == "__main__":
    unittest.main()
