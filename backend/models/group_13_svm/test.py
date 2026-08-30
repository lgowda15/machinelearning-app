"""Unit tests for SVMModel.

Run: python -m pytest test.py --cov=. --cov-report=term-missing
"""

import unittest

import numpy as np

from models.group_13_svm.model import SVMModel


class TestSVMModelFitting(unittest.TestCase):
    """fit() behaviour: return value, state updates, validation."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100).astype(np.int64)
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        model = SVMModel()
        self.assertIs(model.fit(self.X, self.y), model)

    def test_fit_sets_is_fitted(self):
        model = SVMModel().fit(self.X, self.y)
        self.assertTrue(model.is_fitted)

    def test_fit_sets_n_features(self):
        model = SVMModel().fit(self.X, self.y)
        self.assertEqual(model.n_features, 4)

    def test_fit_sets_classes(self):
        model = SVMModel().fit(self.X, self.y)
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))

    def test_fit_none_y_raises(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(self.X, None)

    def test_fit_sample_mismatch_raises(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(self.X, self.y[:-1])

    def test_fit_non_ndarray_X_raises_type_error(self):
        with self.assertRaises(TypeError):
            SVMModel().fit(self.X.tolist(), self.y)

    def test_fit_non_ndarray_y_raises_type_error(self):
        with self.assertRaises(TypeError):
            SVMModel().fit(self.X, self.y.tolist())

    def test_fit_wrong_dtype_raises_type_error(self):
        with self.assertRaises(TypeError):
            SVMModel().fit(self.X.astype(np.float32), self.y)

    def test_fit_1d_X_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(self.X[:, 0], self.y)

    def test_fit_3d_X_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(self.X.reshape(50, 2, 4), self.y)

    def test_fit_2d_y_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(self.X, self.y.reshape(-1, 1))

    def test_fit_empty_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(np.empty((0, 4), dtype=np.float64), np.empty((0,)))

    def test_fit_empty_features_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel().fit(np.empty((10, 0), dtype=np.float64), self.y[:10])

    def test_fit_empty_labels_raises_value_error(self):
        X_one_row = np.empty((1, 4), dtype=np.float64)
        with self.assertRaises(ValueError):
            SVMModel().fit(X_one_row, np.empty((0,), dtype=np.int64))

    def test_repeated_fit_overwrites_previous_state(self):
        model = SVMModel()
        model.fit(self.X, self.y)
        rng = np.random.default_rng(7)
        X2 = rng.standard_normal((60, 6))
        y2 = rng.integers(0, 3, 60).astype(np.int64)
        model.fit(X2, y2)
        self.assertEqual(model.n_features, 6)
        self.assertEqual(len(model.classes_), 3)

    def test_invalid_kernel_raises_value_error(self):
        with self.assertRaises(ValueError):
            SVMModel(kernel="not_a_real_kernel")


class TestSVMModelPredict(unittest.TestCase):
    """predict() behaviour: shape, dtype, validation."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100).astype(np.int64)
        self.X_test = rng.standard_normal((20, 4))
        self.model = SVMModel().fit(self.X, self.y)

    def test_predict_shape(self):
        self.assertEqual(self.model.predict(self.X_test).shape, (20,))

    def test_predict_is_ndarray(self):
        self.assertIsInstance(self.model.predict(self.X_test), np.ndarray)

    def test_predict_values_are_known_classes(self):
        preds = self.model.predict(self.X_test)
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_predict_before_fit_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            SVMModel().predict(self.X_test)

    def test_predict_before_fit_error_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            SVMModel().predict(self.X_test)
        self.assertIn("fit()", str(ctx.exception))

    def test_predict_feature_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test[:, :2])

    def test_predict_feature_mismatch_error_message(self):
        with self.assertRaises(ValueError) as ctx:
            self.model.predict(self.X_test[:, :2])
        self.assertIn("features", str(ctx.exception))

    def test_predict_non_ndarray_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.model.predict(self.X_test.tolist())

    def test_predict_wrong_dtype_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.model.predict(self.X_test.astype(np.float32))


class TestSVMModelPredictProba(unittest.TestCase):
    """predict_proba() behaviour: shape, normalisation, validation."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100).astype(np.int64)
        self.X_test = rng.standard_normal((20, 4))
        self.model = SVMModel().fit(self.X, self.y)

    def test_predict_proba_shape(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (20, 2))

    def test_predict_proba_rows_sum_to_one(self):
        proba = self.model.predict_proba(self.X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_is_float64(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.dtype, np.float64)

    def test_predict_proba_values_in_unit_interval(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_predict_proba_before_fit_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            SVMModel().predict_proba(self.X_test)

    def test_predict_proba_feature_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.predict_proba(self.X_test[:, :2])

    def test_predict_proba_column_count_matches_classes(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape[1], len(self.model.classes_))


class TestSVMModelMetadata(unittest.TestCase):
    """get_metadata() contract: exact keys, correct values."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100).astype(np.int64)
        self.model = SVMModel(kernel="linear", C=2.0).fit(self.X, self.y)
        self.metadata = self.model.get_metadata()

    def test_metadata_has_exactly_required_keys(self):
        expected = {
            "model_name",
            "model_type",
            "hyperparameters",
            "training_time_seconds",
            "n_features",
            "feature_importance",
        }
        self.assertEqual(set(self.metadata.keys()), expected)

    def test_metadata_model_name(self):
        self.assertEqual(self.metadata["model_name"], "Support Vector Machine")

    def test_metadata_model_type(self):
        self.assertEqual(self.metadata["model_type"], "classifier")

    def test_metadata_n_features(self):
        self.assertEqual(self.metadata["n_features"], 4)

    def test_metadata_feature_importance_is_none(self):
        self.assertIsNone(self.metadata["feature_importance"])

    def test_metadata_hyperparameters_reflect_constructor_args(self):
        hyperparams = self.metadata["hyperparameters"]
        self.assertEqual(hyperparams["kernel"], "linear")
        self.assertEqual(hyperparams["C"], 2.0)

    def test_metadata_training_time_is_non_negative_float(self):
        training_time = self.metadata["training_time_seconds"]
        self.assertIsInstance(training_time, float)
        self.assertGreaterEqual(training_time, 0.0)


class TestSVMModelDeterminism(unittest.TestCase):
    """Identical input and seed must produce identical output (Section 10)."""

    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.standard_normal((80, 5))
        self.y = rng.integers(0, 3, 80).astype(np.int64)
        self.X_test = rng.standard_normal((15, 5))

    def test_predict_is_deterministic_across_fits(self):
        a = SVMModel().fit(self.X, self.y).predict(self.X_test)
        b = SVMModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_predict_proba_is_deterministic_across_fits(self):
        a = SVMModel().fit(self.X, self.y).predict_proba(self.X_test)
        b = SVMModel().fit(self.X, self.y).predict_proba(self.X_test)
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestSVMModelKernelsAndClassCounts(unittest.TestCase):
    """Every supported kernel, and both binary and multiclass targets."""

    def setUp(self):
        rng = np.random.default_rng(99)
        self.X = rng.standard_normal((120, 4))
        self.X_test = rng.standard_normal((10, 4))

    def test_linear_kernel_fits_and_predicts(self):
        y = np.random.default_rng(1).integers(0, 2, 120).astype(np.int64)
        model = SVMModel(kernel="linear").fit(self.X, y)
        self.assertEqual(model.predict(self.X_test).shape, (10,))

    def test_poly_kernel_fits_and_predicts(self):
        y = np.random.default_rng(1).integers(0, 2, 120).astype(np.int64)
        model = SVMModel(kernel="poly", degree=3).fit(self.X, y)
        self.assertEqual(model.predict(self.X_test).shape, (10,))

    def test_rbf_kernel_fits_and_predicts(self):
        y = np.random.default_rng(1).integers(0, 2, 120).astype(np.int64)
        model = SVMModel(kernel="rbf").fit(self.X, y)
        self.assertEqual(model.predict(self.X_test).shape, (10,))

    def test_sigmoid_kernel_fits_and_predicts(self):
        y = np.random.default_rng(1).integers(0, 2, 120).astype(np.int64)
        model = SVMModel(kernel="sigmoid", coef0=0.1).fit(self.X, y)
        self.assertEqual(model.predict(self.X_test).shape, (10,))

    def test_binary_classification(self):
        y = np.random.default_rng(1).integers(0, 2, 120).astype(np.int64)
        model = SVMModel().fit(self.X, y)
        self.assertEqual(len(model.classes_), 2)

    def test_multiclass_classification(self):
        y = np.random.default_rng(1).integers(0, 4, 120).astype(np.int64)
        model = SVMModel().fit(self.X, y)
        self.assertEqual(len(model.classes_), 4)
        proba = model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (10, 4))


if __name__ == "__main__":
    unittest.main()
