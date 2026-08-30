import unittest

import numpy as np

from models.group_15_pca.model import PCAModel


class TestPCAModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = PCAModel()
        self.assertIs(m.fit(self.X), m)

    def test_fit_bad_shape_raises(self):
        m = PCAModel()
        with self.assertRaises(ValueError):
            m.fit(np.array([1, 2, 3]))

    def test_predict_returns_transformed_matrix(self):
        m = PCAModel(n_components=2).fit(self.X)
        transformed = m.predict(self.X_test)
        self.assertEqual(transformed.shape, (20, 2))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            PCAModel().predict(self.X_test)
            
    def test_predict_wrong_features_raises(self):
        m = PCAModel().fit(self.X)
        with self.assertRaises(ValueError):
            m.predict(np.random.default_rng(42).standard_normal((20, 5)))

    def test_predict_proba_returns_none(self):
        m = PCAModel().fit(self.X)
        self.assertIsNone(m.predict_proba(self.X_test))

    def test_determinism(self):
        a = PCAModel().fit(self.X).predict(self.X_test)
        b = PCAModel().fit(self.X).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = PCAModel().fit(self.X).get_metadata()
        for k in ["model_name", "model_type", "hyperparameters",
                  "training_time_seconds", "n_features", "feature_importance"]:
            self.assertIn(k, md)
        self.assertEqual(md["model_type"], "dimensionality_reducer")

    def test_visualization_data(self):
        m = PCAModel(n_components=2).fit(self.X)
        viz = m.get_visualization_data()
        self.assertIn("explained_variance_ratio", viz)
        self.assertIsInstance(viz["explained_variance_ratio"], list)
        self.assertEqual(len(viz["explained_variance_ratio"]), 2)
        
    def test_visualization_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            PCAModel().get_visualization_data()

if __name__ == "__main__":
    unittest.main()