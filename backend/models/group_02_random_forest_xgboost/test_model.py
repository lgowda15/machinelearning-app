import unittest

import numpy as np

from models.group_02_random_forest_xgboost.random_forest import RandomForestModel
from models.group_02_random_forest_xgboost.xgboost_model import XGBoostModel


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((120, 5)).astype(np.float64)
        self.y = np.array(
            [0, 1, 2] * 40
        )
        self.X_test = rng.standard_normal((20, 5)).astype(np.float64)

    def test_random_forest_fit_returns_self(self):
        model = RandomForestModel(n_estimators=20).fit(self.X, self.y)
        self.assertIsInstance(model, RandomForestModel)
        self.assertTrue(model.is_fitted)

    def test_xgboost_fit_returns_self(self):
        model = XGBoostModel(n_estimators=20).fit(self.X, self.y)
        self.assertIsInstance(model, XGBoostModel)
        self.assertTrue(model.is_fitted)

    def test_prediction_shape(self):
        for model_class in (RandomForestModel, XGBoostModel):
            model = model_class(n_estimators=20).fit(self.X, self.y)
            self.assertEqual(model.predict(self.X_test).shape, (20,))

    def test_probability_shape_and_sum(self):
        for model_class in (RandomForestModel, XGBoostModel):
            model = model_class(n_estimators=20).fit(self.X, self.y)
            probabilities = model.predict_proba(self.X_test)
            self.assertEqual(probabilities.shape, (20, 3))
            np.testing.assert_allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-6
            )

    def test_predict_before_fit_raises(self):
        for model_class in (RandomForestModel, XGBoostModel):
            with self.assertRaises(RuntimeError):
                model_class().predict(self.X_test)

    def test_wrong_feature_count_raises(self):
        bad_X = np.ones((5, 4), dtype=np.float64)
        for model_class in (RandomForestModel, XGBoostModel):
            model = model_class(n_estimators=20).fit(self.X, self.y)
            with self.assertRaises(ValueError):
                model.predict(bad_X)

    def test_invalid_dtype_raises(self):
        bad_X = self.X.astype(np.float32)
        for model_class in (RandomForestModel, XGBoostModel):
            with self.assertRaises(TypeError):
                model_class(n_estimators=20).fit(bad_X, self.y)

    def test_metadata_keys(self):
        expected = {
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        }

        for model_class in (RandomForestModel, XGBoostModel):
            model = model_class(n_estimators=20).fit(self.X, self.y)
            metadata = model.get_metadata()
            self.assertEqual(set(metadata.keys()), expected)
            self.assertEqual(metadata["model_type"], "classifier")

    def test_determinism(self):
        for model_class in (RandomForestModel, XGBoostModel):
            a = model_class(n_estimators=20).fit(self.X, self.y)
            b = model_class(n_estimators=20).fit(self.X, self.y)
            np.testing.assert_array_equal(
                a.predict(self.X_test),
                b.predict(self.X_test),
            )
            np.testing.assert_allclose(
                a.predict_proba(self.X_test),
                b.predict_proba(self.X_test),
                atol=1e-12,
            )

    def test_single_class_training_raises(self):
        single_class_y = np.zeros(120, dtype=int)
        for model_class in (RandomForestModel, XGBoostModel):
            with self.assertRaises(ValueError):
                model_class(n_estimators=20).fit(self.X, single_class_y)

    def test_visualization_data_is_json_serializable(self):
        import json

        for model_class in (RandomForestModel, XGBoostModel):
            model = model_class(n_estimators=20).fit(self.X, self.y)
            visualization = model.get_visualization_data()
            json.dumps(visualization)


if __name__ == "__main__":
    unittest.main()
