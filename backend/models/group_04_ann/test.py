"""Unit tests for Artificial Neural Network (MLP). Minimum 80% coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Allow imports to work when pytest is run from this folder.
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.group_04_ann.model import ANNModel, _build_network


class TestANNModel(unittest.TestCase):
    """Unit tests for the ANNModel implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)

        cls.X_classification = rng.normal(size=(24, 4)).astype(np.float64)
        cls.y_classification = (
            cls.X_classification[:, 0] + cls.X_classification[:, 1] > 0
        ).astype(np.int64)

        cls.X_regression = rng.normal(size=(24, 4)).astype(np.float64)
        cls.y_regression = (
            2.0 * cls.X_regression[:, 0]
            - 1.5 * cls.X_regression[:, 1]
            + 0.5
        ).astype(np.float64)

    def build(self, **kwargs: object) -> ANNModel:
        """Build a small ANN so tests run quickly on CPU."""
        defaults = {
            "hidden_sizes": [8, 4],
            "epochs": 3,
            "batch_size": 8,
            "dropout_rate": 0.0,
            "random_state": 42,
        }
        defaults.update(kwargs)
        return ANNModel(**defaults)

    def test_default_initialization(self) -> None:
        model = ANNModel()
        self.assertEqual(model.hidden_sizes, [128, 64])
        self.assertEqual(model.activation, "relu")
        self.assertEqual(model.lr, 1e-3)
        self.assertEqual(model.epochs, 100)
        self.assertEqual(model.batch_size, 64)

    def test_invalid_parameters_raise_value_error(self) -> None:
        invalid_parameters = [
            {"hidden_sizes": []},
            {"hidden_sizes": [0, 4]},
            {"lr": 0},
            {"epochs": 0},
            {"batch_size": 0},
            {"dropout_rate": -0.1},
            {"dropout_rate": 1.0},
        ]

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    ANNModel(**parameters)

    def test_all_supported_activations_build_network(self) -> None:
        for activation in ("relu", "tanh", "leaky_relu", "elu"):
            with self.subTest(activation=activation):
                network = _build_network(
                    in_features=4,
                    hidden_sizes=[8, 4],
                    out_features=2,
                    activation=activation,
                    dropout_rate=0.0,
                )
                self.assertIsNotNone(network)

    def test_invalid_activation_raises_value_error(self) -> None:
        model = self.build(activation="invalid_activation")

        with self.assertRaises(ValueError):
            model.fit(self.X_classification, self.y_classification)

    def test_classification_fit_predict_and_probabilities(self) -> None:
        model = self.build().fit(
            self.X_classification,
            self.y_classification,
        )

        self.assertTrue(model.is_fitted)
        self.assertEqual(model._model_type, "classifier")
        self.assertEqual(model.n_features, 4)

        predictions = model.predict(self.X_classification)
        probabilities = model.predict_proba(self.X_classification)

        self.assertEqual(predictions.shape, (24,))
        self.assertEqual(probabilities.shape, (24, 2))
        self.assertTrue(np.all(np.isin(predictions, model.classes_)))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(24),
            atol=1e-6,
        )

    def test_classification_metadata_and_visualization(self) -> None:
        model = self.build().fit(
            self.X_classification,
            self.y_classification,
        )

        metadata = model.get_metadata()
        visualization = model.get_visualization_data()

        self.assertEqual(
            metadata["model_name"],
            "Artificial Neural Network (MLP)",
        )
        self.assertEqual(metadata["model_type"], "classifier")
        self.assertEqual(metadata["n_features"], 4)
        self.assertIsNone(metadata["feature_importance"])

        self.assertIsNotNone(visualization)
        self.assertEqual(visualization["model_type"], "classifier")
        self.assertEqual(len(visualization["loss_history"]), 3)
        self.assertEqual(visualization["epochs"], [1, 2, 3])

    def test_regression_fit_predict_and_no_probabilities(self) -> None:
        model = self.build().fit(
            self.X_regression,
            self.y_regression,
        )

        self.assertTrue(model.is_fitted)
        self.assertEqual(model._model_type, "regressor")

        predictions = model.predict(self.X_regression)
        probabilities = model.predict_proba(self.X_regression)

        self.assertEqual(predictions.shape, (24,))
        self.assertEqual(predictions.dtype, np.float64)
        self.assertIsNone(probabilities)

    def test_predict_before_fit_raises_runtime_error(self) -> None:
        model = self.build()

        with self.assertRaises(RuntimeError):
            model.predict(self.X_classification)

    def test_predict_proba_before_fit_raises_runtime_error(self) -> None:
        model = self.build()

        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X_classification)

    def test_fit_input_validation(self) -> None:
        model = self.build()

        with self.assertRaises(TypeError):
            model.fit(self.X_classification.tolist(), self.y_classification)

        with self.assertRaises(ValueError):
            model.fit(self.X_classification, None)

        with self.assertRaises(TypeError):
            model.fit(self.X_classification, self.y_classification.tolist())

        with self.assertRaises(ValueError):
            model.fit(self.X_classification[0], self.y_classification)

        with self.assertRaises(ValueError):
            model.fit(
                self.X_classification,
                self.y_classification[:-1],
            )

        with self.assertRaises(ValueError):
            model.fit(
                self.X_classification[:1],
                self.y_classification[:1],
            )

    def test_predict_input_validation(self) -> None:
        model = self.build().fit(
            self.X_classification,
            self.y_classification,
        )

        with self.assertRaises(TypeError):
            model.predict(self.X_classification.tolist())

        with self.assertRaises(ValueError):
            model.predict(self.X_classification[0])

        wrong_features = np.zeros((4, 5), dtype=np.float64)

        with self.assertRaises(ValueError):
            model.predict(wrong_features)

    def test_predict_proba_input_validation(self) -> None:
        model = self.build().fit(
            self.X_classification,
            self.y_classification,
        )

        with self.assertRaises(TypeError):
            model.predict_proba(self.X_classification.tolist())

        wrong_features = np.zeros((4, 5), dtype=np.float64)

        with self.assertRaises(ValueError):
            model.predict_proba(wrong_features)

        one_dimensional = self.X_classification[0]

        self.assertIsNone(model.predict_proba(one_dimensional))


if __name__ == "__main__":
    unittest.main()
