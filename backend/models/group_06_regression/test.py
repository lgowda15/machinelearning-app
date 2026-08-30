"""
models/group_06_regression/test.py

Run with:
    python -m pytest test.py --cov=. --cov-report=term-missing
"""

import os
import sys
import unittest

import numpy as np

# Ensure this file's own directory is on sys.path, regardless of how
# pytest's rootdir/package detection resolves things (the __init__.py
# in this folder can otherwise cause pytest to treat this as part of a
# package rooted one level up, breaking the bare "from model import"
# below and silently collecting zero tests).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RegressionModel


class TestRegressionModelMultivariate(unittest.TestCase):
    """Covers the 2+ feature / non-linear branch."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        # nonlinear target so polynomial expansion actually matters
        self.y = (
            2.0 * self.X[:, 0] ** 2
            + 1.5 * self.X[:, 1]
            + 0.5 * self.X[:, 2] * self.X[:, 3]
            + rng.standard_normal(100) * 0.1
        )
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = RegressionModel()
        self.assertIs(m.fit(self.X, self.y), m)

    def test_predict_shape(self):
        m = RegressionModel().fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            RegressionModel().predict(self.X_test)

    def test_predict_proba_is_none(self):
        m = RegressionModel().fit(self.X, self.y)
        self.assertIsNone(m.predict_proba(self.X_test))

    def test_determinism(self):
        a = RegressionModel().fit(self.X, self.y).predict(self.X_test)
        b = RegressionModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = RegressionModel().fit(self.X, self.y).get_metadata()
        for k in ["model_name", "model_type", "hyperparameters",
                  "training_time_seconds", "n_features", "feature_importance"]:
            self.assertIn(k, md)
        self.assertEqual(md["model_type"], "regressor")
        self.assertEqual(md["n_features"], 4)

    def test_wrong_feature_count_raises(self):
        m = RegressionModel().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            m.predict(self.X_test[:, :2])


class TestRegressionModelUnivariate(unittest.TestCase):
    """Covers the single-feature / non-stationary branch."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 100
        # random-walk target: non-stationary by construction
        self.X = np.arange(n, dtype=np.float64).reshape(-1, 1)
        self.y = np.cumsum(rng.standard_normal(n)) + 50.0
        self.X_test = np.arange(n, n + 10, dtype=np.float64).reshape(-1, 1)

        # stationary target: mean-reverting noise around a constant
        self.y_stationary = rng.standard_normal(n) + 10.0

    def test_fit_returns_self(self):
        m = RegressionModel()
        self.assertIs(m.fit(self.X, self.y), m)

    def test_predict_shape(self):
        m = RegressionModel().fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (10,))

    def test_detects_non_stationary_series(self):
        m = RegressionModel().fit(self.X, self.y)
        self.assertTrue(m._differenced)

    def test_detects_stationary_series(self):
        m = RegressionModel().fit(self.X, self.y_stationary)
        self.assertFalse(m._differenced)

    def test_determinism(self):
        a = RegressionModel().fit(self.X, self.y).predict(self.X_test)
        b = RegressionModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = RegressionModel().fit(self.X, self.y).get_metadata()
        for k in ["model_name", "model_type", "hyperparameters",
                  "training_time_seconds", "n_features", "feature_importance"]:
            self.assertIn(k, md)
        self.assertEqual(md["n_features"], 1)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            RegressionModel().predict(self.X_test)


if __name__ == "__main__":
    unittest.main()
