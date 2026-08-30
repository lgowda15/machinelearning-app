"""Reference tests. Same pattern applies to your model."""
import unittest

import numpy as np
from model import LogisticRegressionModel


class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100)
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = LogisticRegressionModel()
        self.assertIs(m.fit(self.X, self.y), m)

    def test_predict_shape(self):
        m = LogisticRegressionModel().fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LogisticRegressionModel().predict(self.X_test)

    def test_proba_rows_sum_to_one(self):
        m = LogisticRegressionModel().fit(self.X, self.y)
        p = m.predict_proba(self.X_test)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_determinism(self):
        a = LogisticRegressionModel().fit(self.X, self.y).predict(self.X_test)
        b = LogisticRegressionModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = LogisticRegressionModel().fit(self.X, self.y).get_metadata()
        for k in ["model_name", "model_type", "hyperparameters",
                  "training_time_seconds", "n_features", "feature_importance"]:
            self.assertIn(k, md)


if __name__ == "__main__":
    unittest.main()