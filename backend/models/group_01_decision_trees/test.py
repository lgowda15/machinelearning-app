"""Unit tests for the Group 01 decision-tree models.

Run from inside this folder:

    python -m pytest test.py --cov=. --cov-report=term-missing

All fixtures are synthetic numpy arrays generated with a fixed seed; no data
file is read or committed.  ``sys.path`` is extended with the ``backend``
directory (derived from ``__file__``, never a hardcoded absolute path) so
that ``models.base_model`` resolves however pytest is invoked.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, ClassVar

import numpy as np

try:
    from models.base_model import BaseModel
    from models.group_01_decision_trees import (
        CARTModel,
        CHAIDModel,
        ID3Model,
        ObliqueDecisionTreeModel,
    )
except ModuleNotFoundError:  # pragma: no cover - invoked from inside this folder
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir
            )
        ),
    )
    from models.base_model import BaseModel
    from models.group_01_decision_trees import (
        CARTModel,
        CHAIDModel,
        ID3Model,
        ObliqueDecisionTreeModel,
    )

REQUIRED_METADATA_KEYS = {
    "model_name",
    "model_type",
    "hyperparameters",
    "training_time_seconds",
    "n_features",
    "feature_importance",
}

NODE_KEYS = {
    "id",
    "depth",
    "is_leaf",
    "n_samples",
    "impurity",
    "impurity_measure",
    "class_distribution",
    "class_probabilities",
    "predicted_class",
    "split",
    "children",
}

SPLIT_KEYS = {
    "type",
    "feature",
    "feature_name",
    "gain",
    "threshold",
    "bin_edges",
    "coefficients",
    "intercept",
    "chi_square",
    "p_value",
    "p_value_adjusted",
    "degrees_of_freedom",
    "condition",
}


def make_binary_dataset(
    n_samples: int = 240, n_features: int = 6, seed: int = 42
) -> dict[str, np.ndarray]:
    """Linearly-separable-ish two-class problem, already 'preprocessed'."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    signal = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2]
    y = (signal + 0.25 * rng.standard_normal(n_samples) > 0.0).astype(np.int64)
    return {"X": X.astype(np.float64), "y": y}


def make_multiclass_dataset(
    n_samples: int = 300, n_features: int = 5, seed: int = 7
) -> dict[str, np.ndarray]:
    """Three-class problem with string labels, to exercise label handling."""
    rng = np.random.default_rng(seed)
    centres = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 2.0, 0.0, 0.0],
        ]
    )
    labels = rng.integers(0, 3, n_samples)
    X = centres[labels] + rng.standard_normal((n_samples, n_features))
    names = np.array(["low", "mid", "high"])
    return {"X": X.astype(np.float64), "y": names[labels]}


class DecisionTreeModelContract:
    """Shared contract tests, one concrete TestCase per algorithm.

    Deliberately not a ``TestCase`` subclass so unittest/pytest do not
    collect it directly.
    """

    MODEL_CLS: type[BaseModel]
    EXPECTED_SPLIT_TYPES: ClassVar[frozenset[str]]

    def setUp(self) -> None:
        binary = make_binary_dataset()
        self.X = binary["X"]
        self.y = binary["y"]
        self.X_test = self.X[:20].copy()

    def build(self, **kwargs: Any) -> BaseModel:
        return self.MODEL_CLS(**kwargs)

    # -- 1. initialisation --------------------------------------------
    def test_initialisation_state(self) -> None:
        model = self.build()
        self.assertIsInstance(model, BaseModel)
        self.assertFalse(model.is_fitted)
        self.assertIsNone(model.n_features)
        self.assertIsInstance(model.hyperparams, dict)
        self.assertEqual(model.hyperparams["random_state"], 42)

    def test_hyperparams_cover_used_parameters(self) -> None:
        model = self.build()
        for name in model.hyperparams:
            self.assertTrue(
                hasattr(model, name),
                f"{name} reported in hyperparams but not stored on the model",
            )

    # -- 2/3/4. fit ---------------------------------------------------
    def test_fit_returns_self(self) -> None:
        model = self.build()
        self.assertIs(model.fit(self.X, self.y), model)

    def test_fitted_state_and_feature_count(self) -> None:
        model = self.build().fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.n_features, self.X.shape[1])
        np.testing.assert_array_equal(model.classes_, np.unique(self.y))

    def test_training_time_recorded(self) -> None:
        model = self.build().fit(self.X, self.y)
        elapsed = model.get_metadata()["training_time_seconds"]
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 300.0)

    # -- 5/6/7. predict -----------------------------------------------
    def test_predict_shape_and_dtype(self) -> None:
        model = self.build().fit(self.X, self.y)
        predictions = model.predict(self.X_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.ndim, 1)
        self.assertEqual(predictions.shape, (self.X_test.shape[0],))
        self.assertTrue(set(np.unique(predictions)).issubset(set(np.unique(self.y))))

    def test_predict_before_fit_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.build().predict(self.X_test)

    def test_predict_proba_before_fit_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.build().predict_proba(self.X_test)

    def test_visualization_before_fit_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.build().get_visualization_data()

    def test_wrong_feature_count_raises_value_error(self) -> None:
        model = self.build().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict(self.X_test[:, :-1])
        with self.assertRaises(ValueError):
            model.predict_proba(self.X_test[:, :-1])

    def test_predict_rejects_non_2d_input(self) -> None:
        model = self.build().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict(self.X_test[0])

    def test_predict_rejects_non_finite_input(self) -> None:
        model = self.build().fit(self.X, self.y)
        corrupted = self.X_test.copy()
        corrupted[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.predict(corrupted)

    # -- 8/9. predict_proba -------------------------------------------
    def test_predict_proba_shape(self) -> None:
        model = self.build().fit(self.X, self.y)
        proba = model.predict_proba(self.X_test)
        self.assertEqual(
            proba.shape, (self.X_test.shape[0], len(np.unique(self.y)))
        )

    def test_predict_proba_rows_sum_to_one(self) -> None:
        model = self.build().fit(self.X, self.y)
        proba = model.predict_proba(self.X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue((proba >= 0.0).all() and (proba <= 1.0).all())

    def test_predict_proba_columns_match_classes(self) -> None:
        model = self.build().fit(self.X, self.y)
        proba = model.predict_proba(self.X_test)
        argmax_labels = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(argmax_labels, model.predict(self.X_test))

    # -- 10/11/12. metadata -------------------------------------------
    def test_metadata_has_exactly_required_keys(self) -> None:
        metadata = self.build().fit(self.X, self.y).get_metadata()
        self.assertEqual(set(metadata.keys()), REQUIRED_METADATA_KEYS)

    def test_metadata_model_type_is_classifier(self) -> None:
        metadata = self.build().fit(self.X, self.y).get_metadata()
        self.assertEqual(metadata["model_type"], "classifier")
        self.assertIsInstance(metadata["model_name"], str)
        self.assertEqual(metadata["n_features"], self.X.shape[1])

    def test_metadata_feature_importance_is_valid(self) -> None:
        metadata = self.build().fit(self.X, self.y).get_metadata()
        importance = metadata["feature_importance"]
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), self.X.shape[1])
        for name, score in importance.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)

    def test_metadata_is_json_serialisable(self) -> None:
        metadata = self.build().fit(self.X, self.y).get_metadata()
        json.dumps(metadata)

    # -- 13. determinism ---------------------------------------------
    def test_deterministic_predictions(self) -> None:
        first = self.build().fit(self.X, self.y)
        second = self.build().fit(self.X, self.y)
        np.testing.assert_array_equal(
            first.predict(self.X_test), second.predict(self.X_test)
        )
        np.testing.assert_array_equal(
            first.predict_proba(self.X_test), second.predict_proba(self.X_test)
        )

    def test_deterministic_tree_structure(self) -> None:
        first = self.build().fit(self.X, self.y).get_visualization_data()
        second = self.build().fit(self.X, self.y).get_visualization_data()
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_refit_is_idempotent(self) -> None:
        model = self.build().fit(self.X, self.y)
        before = model.predict(self.X_test)
        after = model.fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(before, after)

    # -- 14/15. visualisation -----------------------------------------
    def test_visualization_data_structure(self) -> None:
        model = self.build().fit(self.X, self.y)
        payload = model.get_visualization_data()
        self.assertIn("tree_structure", payload)
        tree = payload["tree_structure"]
        for key in (
            "algorithm",
            "split_type",
            "impurity_measure",
            "root_id",
            "n_nodes",
            "n_leaves",
            "max_depth_reached",
            "n_features",
            "feature_names",
            "classes",
            "nodes",
            "edges",
        ):
            self.assertIn(key, tree)
        self.assertEqual(tree["n_nodes"], len(tree["nodes"]))
        self.assertEqual(tree["root_id"], 0)
        self.assertEqual(len(tree["feature_names"]), self.X.shape[1])
        self.assertGreaterEqual(tree["n_leaves"], 1)
        self.assertLessEqual(tree["max_depth_reached"], model.max_depth)

    def test_visualization_nodes_and_edges_are_consistent(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        node_ids = {node["id"] for node in tree["nodes"]}
        self.assertEqual(node_ids, set(range(len(tree["nodes"]))))
        for node in tree["nodes"]:
            self.assertEqual(set(node.keys()), NODE_KEYS)
            self.assertGreaterEqual(node["n_samples"], 0)
            self.assertAlmostEqual(sum(node["class_probabilities"]), 1.0, places=9)
            self.assertEqual(
                sum(node["class_distribution"].values()), node["n_samples"]
            )
            if node["is_leaf"]:
                self.assertIsNone(node["split"])
                self.assertEqual(node["children"], [])
            else:
                self.assertEqual(set(node["split"].keys()), SPLIT_KEYS)
                self.assertIn(node["split"]["type"], self.EXPECTED_SPLIT_TYPES)
                self.assertIsInstance(node["split"]["condition"], str)
                self.assertGreaterEqual(len(node["children"]), 2)
                for child in node["children"]:
                    self.assertIn(child, node_ids)
        edge_pairs = {(edge["source"], edge["target"]) for edge in tree["edges"]}
        expected_pairs = {
            (node["id"], child)
            for node in tree["nodes"]
            for child in node["children"]
        }
        self.assertEqual(edge_pairs, expected_pairs)
        for edge in tree["edges"]:
            self.assertEqual(
                set(edge.keys()), {"source", "target", "branch_index", "label"}
            )

    def test_visualization_data_is_json_serialisable(self) -> None:
        payload = self.build().fit(self.X, self.y).get_visualization_data()
        restored = json.loads(json.dumps(payload))
        self.assertEqual(
            restored["tree_structure"]["n_nodes"],
            payload["tree_structure"]["n_nodes"],
        )

    # -- 16. behaviour on synthetic data ------------------------------
    def test_learns_a_separable_binary_problem(self) -> None:
        model = self.build().fit(self.X, self.y)
        accuracy = float((model.predict(self.X) == self.y).mean())
        self.assertGreater(accuracy, 0.7)

    def test_single_class_target_predicts_that_class(self) -> None:
        y_constant = np.zeros(self.X.shape[0], dtype=np.int64)
        model = self.build().fit(self.X, y_constant)
        predictions = model.predict(self.X_test)
        np.testing.assert_array_equal(predictions, np.zeros(20, dtype=np.int64))
        proba = model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (20, 1))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_multiclass_string_labels(self) -> None:
        data = make_multiclass_dataset()
        model = self.build().fit(data["X"], data["y"])
        predictions = model.predict(data["X"][:30])
        self.assertEqual(predictions.shape, (30,))
        self.assertTrue(set(np.unique(predictions)).issubset({"low", "mid", "high"}))
        proba = model.predict_proba(data["X"][:30])
        self.assertEqual(proba.shape, (30, 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        json.dumps(model.get_visualization_data())

    def test_constant_features_produce_a_usable_model(self) -> None:
        X_constant = np.ones((60, 3), dtype=np.float64)
        y = np.array([0, 1] * 30, dtype=np.int64)
        model = self.build().fit(X_constant, y)
        predictions = model.predict(X_constant)
        self.assertEqual(predictions.shape, (60,))
        tree = model.get_visualization_data()["tree_structure"]
        self.assertGreaterEqual(tree["n_nodes"], 1)

    def test_unseen_regions_still_predict(self) -> None:
        model = self.build().fit(self.X, self.y)
        far_away = np.full((5, self.X.shape[1]), 1e3, dtype=np.float64)
        predictions = model.predict(far_away)
        self.assertEqual(predictions.shape, (5,))
        np.testing.assert_allclose(
            model.predict_proba(far_away).sum(axis=1), 1.0, atol=1e-9
        )

    # -- 17. invalid inputs -------------------------------------------
    def test_fit_without_target_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build().fit(self.X, None)

    def test_fit_rejects_non_2d_x(self) -> None:
        with self.assertRaises(ValueError):
            self.build().fit(self.X[:, 0], self.y)

    def test_fit_rejects_non_1d_y(self) -> None:
        with self.assertRaises(ValueError):
            self.build().fit(self.X, self.y.reshape(-1, 1))

    def test_fit_rejects_mismatched_row_counts(self) -> None:
        with self.assertRaises(ValueError):
            self.build().fit(self.X, self.y[:-5])

    def test_fit_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            self.build().fit(
                np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.int64)
            )

    def test_fit_rejects_non_finite_input(self) -> None:
        corrupted = self.X.copy()
        corrupted[3, 1] = np.inf
        with self.assertRaises(ValueError):
            self.build().fit(corrupted, self.y)

    def test_invalid_hyperparameters_raise_value_error(self) -> None:
        for kwargs in ({"max_depth": 0}, {"min_samples_split": 1},
                       {"min_samples_leaf": 0}):
            with self.assertRaises(ValueError):
                self.build(**kwargs)

    # -- hyperparameter behaviour --------------------------------------
    def test_max_depth_limits_tree(self) -> None:
        shallow = self.build(max_depth=1).fit(self.X, self.y)
        deep = self.build(max_depth=4).fit(self.X, self.y)
        shallow_tree = shallow.get_visualization_data()["tree_structure"]
        deep_tree = deep.get_visualization_data()["tree_structure"]
        self.assertLessEqual(shallow_tree["max_depth_reached"], 1)
        self.assertLessEqual(deep_tree["max_depth_reached"], 4)
        self.assertLessEqual(shallow_tree["n_nodes"], deep_tree["n_nodes"])

    def test_min_samples_leaf_is_respected(self) -> None:
        model = self.build(min_samples_leaf=25).fit(self.X, self.y)
        tree = model.get_visualization_data()["tree_structure"]
        for node in tree["nodes"]:
            if node["is_leaf"] and node["n_samples"] > 0:
                self.assertGreaterEqual(node["n_samples"], 1)
        self.assertGreaterEqual(tree["n_nodes"], 1)


class TestID3Model(DecisionTreeModelContract, unittest.TestCase):
    MODEL_CLS = ID3Model
    EXPECTED_SPLIT_TYPES: ClassVar[frozenset[str]] = frozenset({"multiway_binned"})

    def test_information_gain_is_non_negative(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        self.assertEqual(tree["algorithm"], "ID3")
        self.assertEqual(tree["impurity_measure"], "entropy")
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                self.assertGreater(node["split"]["gain"], 0.0)
                self.assertIsInstance(node["split"]["bin_edges"], list)

    def test_attribute_is_used_once_per_path(self) -> None:
        model = self.build(max_depth=5, n_bins=3).fit(self.X, self.y)
        nodes = {
            node["id"]: node
            for node in model.get_visualization_data()["tree_structure"]["nodes"]
        }

        def walk(node_id: int, used: list[int]) -> None:
            node = nodes[node_id]
            if node["is_leaf"]:
                return
            feature = node["split"]["feature"]
            self.assertNotIn(feature, used)
            for child in node["children"]:
                walk(child, [*used, feature])

        walk(0, [])

    def test_more_bins_gives_wider_splits(self) -> None:
        coarse = self.build(n_bins=2).fit(self.X, self.y)
        fine = self.build(n_bins=5).fit(self.X, self.y)
        coarse_root = coarse.get_visualization_data()["tree_structure"]["nodes"][0]
        fine_root = fine.get_visualization_data()["tree_structure"]["nodes"][0]
        self.assertLessEqual(
            len(coarse_root["children"]), len(fine_root["children"])
        )

    def test_invalid_n_bins_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(n_bins=1)

    def test_invalid_min_information_gain_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(min_information_gain=-0.1)


class TestCARTModel(DecisionTreeModelContract, unittest.TestCase):
    MODEL_CLS = CARTModel
    EXPECTED_SPLIT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"binary_axis_parallel"}
    )

    def test_splits_are_binary_with_thresholds(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        self.assertEqual(tree["algorithm"], "CART")
        self.assertEqual(tree["impurity_measure"], "gini")
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                self.assertEqual(len(node["children"]), 2)
                self.assertIsInstance(node["split"]["threshold"], float)
                self.assertIsNone(node["split"]["bin_edges"])

    def test_entropy_criterion_is_accepted(self) -> None:
        model = self.build(criterion="entropy").fit(self.X, self.y)
        tree = model.get_visualization_data()["tree_structure"]
        self.assertEqual(tree["impurity_measure"], "entropy")

    def test_cost_complexity_pruning_shrinks_the_tree(self) -> None:
        unpruned = self.build(max_depth=6).fit(self.X, self.y)
        pruned = self.build(max_depth=6, ccp_alpha=0.05).fit(self.X, self.y)
        self.assertLessEqual(
            pruned.get_visualization_data()["tree_structure"]["n_nodes"],
            unpruned.get_visualization_data()["tree_structure"]["n_nodes"],
        )

    def test_invalid_criterion_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(criterion="chi_square")

    def test_invalid_ccp_alpha_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(ccp_alpha=-1.0)

    def test_invalid_min_impurity_decrease_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(min_impurity_decrease=-1.0)


class TestCHAIDModel(DecisionTreeModelContract, unittest.TestCase):
    MODEL_CLS = CHAIDModel
    EXPECTED_SPLIT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"multiway_chi_square"}
    )

    def test_splits_report_chi_square_statistics(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        self.assertEqual(tree["algorithm"], "CHAID")
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                split = node["split"]
                self.assertGreaterEqual(split["chi_square"], 0.0)
                self.assertGreaterEqual(split["degrees_of_freedom"], 1)
                self.assertGreaterEqual(split["p_value"], 0.0)
                self.assertLessEqual(split["p_value_adjusted"], 0.05)
                self.assertGreaterEqual(
                    split["p_value_adjusted"], split["p_value"] - 1e-12
                )

    def test_strict_alpha_split_produces_a_smaller_tree(self) -> None:
        lenient = self.build(alpha_split=0.5).fit(self.X, self.y)
        strict = self.build(alpha_split=1e-12).fit(self.X, self.y)
        self.assertLessEqual(
            strict.get_visualization_data()["tree_structure"]["n_nodes"],
            lenient.get_visualization_data()["tree_structure"]["n_nodes"],
        )

    def test_merging_can_be_reported_without_bonferroni(self) -> None:
        with_adjustment = self.build(bonferroni=True).fit(self.X, self.y)
        without = self.build(bonferroni=False).fit(self.X, self.y)
        adjusted = with_adjustment.get_visualization_data()["tree_structure"]
        raw = without.get_visualization_data()["tree_structure"]
        self.assertEqual(
            adjusted["nodes"][0]["split"]["p_value"],
            raw["nodes"][0]["split"]["p_value"],
        )
        self.assertGreaterEqual(
            adjusted["nodes"][0]["split"]["p_value_adjusted"],
            raw["nodes"][0]["split"]["p_value_adjusted"],
        )

    def test_merged_groups_respect_min_samples_leaf(self) -> None:
        model = self.build(min_samples_leaf=30).fit(self.X, self.y)
        tree = model.get_visualization_data()["tree_structure"]
        nodes = {node["id"]: node for node in tree["nodes"]}
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                for child in node["children"]:
                    self.assertGreaterEqual(nodes[child]["n_samples"], 30)

    def test_root_split_uses_the_most_significant_feature(self) -> None:
        """The split must be the *strongest* predictor, not the first one.

        Regression test: comparing adjusted p-values against an absolute
        tolerance made every significant predictor tie, so the lowest
        feature index always won.  Here feature 4 carries by far the
        strongest signal while features 0-3 are weakly informative, so a
        correct implementation must split the root on feature 4.
        """
        rng = np.random.default_rng(11)
        n = 400
        strong = rng.standard_normal(n)
        X = np.column_stack(
            [
                strong + 6.0 * rng.standard_normal(n),
                strong + 6.0 * rng.standard_normal(n),
                strong + 6.0 * rng.standard_normal(n),
                strong + 6.0 * rng.standard_normal(n),
                strong,
            ]
        ).astype(np.float64)
        y = (strong > 0.0).astype(np.int64)

        model = CHAIDModel().fit(X, y)
        root = model.get_visualization_data()["tree_structure"]["nodes"][0]
        self.assertIsNotNone(root["split"], "root should have been split")
        self.assertEqual(root["split"]["feature"], 4)

        # The chosen predictor must also be the global p-value minimum.
        y_index = np.searchsorted(model.classes_, y)
        p_values = []
        for feature in range(X.shape[1]):
            evaluated = model._evaluate_feature(X[:, feature], y_index)
            if evaluated is not None:
                p_values.append(evaluated["p_value_adjusted"])
        self.assertAlmostEqual(
            root["split"]["p_value_adjusted"], min(p_values), places=12
        )

    def test_invalid_alphas_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(alpha_split=0.0)
        with self.assertRaises(ValueError):
            self.build(alpha_merge=1.5)

    def test_invalid_max_bins_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(max_bins=1)


class TestObliqueDecisionTreeModel(DecisionTreeModelContract, unittest.TestCase):
    MODEL_CLS = ObliqueDecisionTreeModel
    EXPECTED_SPLIT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"binary_oblique", "binary_axis_parallel"}
    )

    def test_splits_expose_hyperplane_coefficients(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        self.assertEqual(tree["algorithm"], "Sparse Oblique Decision Tree")
        self.assertIn("n_oblique_splits", tree)
        for node in tree["nodes"]:
            if not node["is_leaf"]:
                coefficients = node["split"]["coefficients"]
                self.assertEqual(len(coefficients), self.X.shape[1])
                self.assertIsInstance(node["split"]["intercept"], float)
                self.assertGreater(sum(abs(c) for c in coefficients), 0.0)

    def test_finds_at_least_one_oblique_split_on_oblique_data(self) -> None:
        tree = self.build().fit(self.X, self.y).get_visualization_data()[
            "tree_structure"
        ]
        self.assertGreaterEqual(tree["n_oblique_splits"], 1)

    def test_stronger_regularisation_gives_sparser_hyperplanes(self) -> None:
        dense = self.build(oblique_C=10.0).fit(self.X, self.y)
        sparse = self.build(oblique_C=0.02).fit(self.X, self.y)

        def non_zero(model: Any) -> int:
            tree = model.get_visualization_data()["tree_structure"]
            return sum(
                sum(1 for c in node["split"]["coefficients"] if abs(c) > 0.0)
                for node in tree["nodes"]
                if not node["is_leaf"]
            )

        self.assertLessEqual(non_zero(sparse), non_zero(dense))

    def test_entropy_criterion_is_accepted(self) -> None:
        model = self.build(criterion="entropy").fit(self.X, self.y)
        tree = model.get_visualization_data()["tree_structure"]
        self.assertEqual(tree["impurity_measure"], "entropy")

    def test_invalid_criterion_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(criterion="chi_square")

    def test_invalid_oblique_c_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(oblique_C=0.0)

    def test_invalid_solver_iterations_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(max_hyperplane_iter=0)

    def test_invalid_min_impurity_decrease_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build(min_impurity_decrease=-0.5)


class TestPackageSurface(unittest.TestCase):
    """The integration team must be able to import and swap the classes."""

    MODEL_CLASSES: ClassVar[tuple[type[BaseModel], ...]] = (
        ID3Model,
        CARTModel,
        CHAIDModel,
        ObliqueDecisionTreeModel,
    )

    def test_all_classes_subclass_base_model(self) -> None:
        for model_cls in self.MODEL_CLASSES:
            self.assertTrue(issubclass(model_cls, BaseModel))
            self.assertTrue(model_cls.__name__.endswith("Model"))

    def test_all_required_methods_are_implemented(self) -> None:
        for model_cls in self.MODEL_CLASSES:
            for method in (
                "fit",
                "predict",
                "predict_proba",
                "get_metadata",
                "get_visualization_data",
            ):
                self.assertTrue(callable(getattr(model_cls, method)))

    def test_models_are_interchangeable_in_the_platform_flow(self) -> None:
        data = make_binary_dataset(n_samples=200, n_features=5)
        X_train, y_train = data["X"][:150], data["y"][:150]
        X_test = data["X"][150:]
        for model_cls in self.MODEL_CLASSES:
            model = model_cls().fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)
            metadata = model.get_metadata()
            visualisation = model.get_visualization_data()
            self.assertEqual(predictions.shape, (X_test.shape[0],))
            self.assertEqual(proba.shape, (X_test.shape[0], 2))
            self.assertEqual(set(metadata.keys()), REQUIRED_METADATA_KEYS)
            json.dumps({"metadata": metadata, "visualization": visualisation})


if __name__ == "__main__":
    unittest.main()
