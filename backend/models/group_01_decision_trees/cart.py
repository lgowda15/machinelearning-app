"""CART decision tree classifier (Group 01, ML integration platform).

CART (Breiman, Friedman, Olshen & Stone, 1984) differs from ID3 in three
ways that matter here:

* every split is **binary** -- ``feature <= threshold`` -- found by an
  exhaustive scan over candidate thresholds on the continuous feature,
* the default splitting criterion is the **Gini index** rather than
  entropy,
* a feature may be **re-used** any number of times along the same path,
  which is what lets a binary axis-parallel tree approximate an oblique
  boundary as a staircase,
* trees are grown large and then reduced by **cost-complexity pruning**
  (exposed here as ``ccp_alpha``).

This class wraps :class:`sklearn.tree.DecisionTreeClassifier`, which *is* an
optimised CART implementation, following the reference pattern in Section 5
of the coding standards.  No preprocessing happens here: the already
scaled, encoded, imputed ``float64`` matrix from the backend goes straight
into the tree.  ``random_state=42`` makes the internal tie-breaking among
equally good splits reproducible.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from models.base_model import BaseModel

MODEL_NAME = "CART Decision Tree"
SPLIT_TYPE = "binary_axis_parallel"
VALID_CRITERIA = ("gini", "entropy", "log_loss")


def _json_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class CARTModel(BaseModel):
    """CART classifier: binary, Gini-based, cost-complexity prunable tree.

    Parameters
    ----------
    criterion:
        ``"gini"`` (the CART criterion), ``"entropy"`` or ``"log_loss"``.
    max_depth:
        Maximum depth of the tree.
    min_samples_split:
        Minimum samples required at a node before it may be split.
    min_samples_leaf:
        Minimum samples that must remain in each child of a split.
    min_impurity_decrease:
        A split is only accepted if it reduces weighted impurity by at
        least this much.
    ccp_alpha:
        Cost-complexity pruning strength.  ``0.0`` disables pruning;
        larger values prune the tree back harder.
    random_state:
        Seed used for tie-breaking between equally good splits.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        if criterion not in VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {VALID_CRITERIA}, got {criterion!r}."
            )
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError(f"max_depth must be an int >= 1, got {max_depth!r}.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be an int >= 2, got {min_samples_split!r}."
            )
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be an int >= 1, got {min_samples_leaf!r}."
            )
        if min_impurity_decrease < 0.0:
            raise ValueError(
                f"min_impurity_decrease must be >= 0, got {min_impurity_decrease!r}."
            )
        if ccp_alpha < 0.0:
            raise ValueError(f"ccp_alpha must be >= 0, got {ccp_alpha!r}.")
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
            **kwargs,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self._model = DecisionTreeClassifier(
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self._train_time: float | None = None

    # ------------------------------------------------------------------
    # input validation
    # ------------------------------------------------------------------
    def _check_fit_inputs(
        self, X: np.ndarray, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if y is None:
            raise ValueError(
                f"{MODEL_NAME} is a supervised classifier; y must not be None."
            )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must be non-empty, got shape {X.shape}.")
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values; the backend is expected to "
                "supply imputed, finite features."
            )
        return X, y

    def _check_predict_input(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Call fit() before using {MODEL_NAME}.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X

    # ------------------------------------------------------------------
    # training and inference
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CARTModel:
        """Grow (and optionally cost-complexity prune) the tree; return self."""
        X, y = self._check_fit_inputs(X, y)
        start = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - start
        self.classes_ = self._model.classes_
        self.n_features = int(X.shape[1])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels, shape ``(n_samples,)``."""
        X = self._check_predict_input(X)
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, shape ``(n_samples, n_classes)``."""
        X = self._check_predict_input(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    # ------------------------------------------------------------------
    # metadata and visualisation
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict[str, Any]:
        """Return the six-key metadata dict required by Section 8."""
        return {
            "model_name": MODEL_NAME,
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": self._feature_importance(),
        }

    def _feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted:
            return None
        scores = np.asarray(self._model.feature_importances_, dtype=np.float64)
        return {f"feature_{i}": float(scores[i]) for i in range(scores.size)}

    @staticmethod
    def _node_counts(tree: Any, node_id: int, n_classes: int) -> np.ndarray:
        """Class counts for one sklearn tree node.

        ``tree_.value`` holds class *fractions* in modern scikit-learn and
        raw counts in older releases; both are handled here.
        """
        raw = np.asarray(tree.value[node_id, 0, :], dtype=np.float64)
        total = raw.sum()
        if total > 0.0 and abs(total - 1.0) < 1e-9:
            raw = raw * float(tree.weighted_n_node_samples[node_id])
        counts = np.rint(raw).astype(np.int64)
        if counts.size != n_classes:  # pragma: no cover - defensive
            padded = np.zeros(n_classes, dtype=np.int64)
            padded[: counts.size] = counts[:n_classes]
            counts = padded
        return counts

    def get_visualization_data(self) -> dict[str, Any] | None:
        """Return the JSON-serialisable ``tree_structure`` payload."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Call fit() before get_visualization_data() on {MODEL_NAME}."
            )
        tree = self._model.tree_
        class_labels = [_json_scalar(c) for c in self.classes_]
        n_classes = len(class_labels)

        depths = np.zeros(tree.node_count, dtype=np.int64)
        for node_id in range(tree.node_count):
            left = int(tree.children_left[node_id])
            right = int(tree.children_right[node_id])
            if left != -1:
                depths[left] = depths[node_id] + 1
                depths[right] = depths[node_id] + 1

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for node_id in range(tree.node_count):
            left = int(tree.children_left[node_id])
            right = int(tree.children_right[node_id])
            is_leaf = left == -1
            counts = self._node_counts(tree, node_id, n_classes)
            total = float(counts.sum())
            probabilities = (
                (counts / total).tolist()
                if total > 0
                else [1.0 / n_classes] * n_classes
            )
            split: dict[str, Any] | None = None
            children: list[int] = []
            if not is_leaf:
                feature = int(tree.feature[node_id])
                threshold = float(tree.threshold[node_id])
                split = {
                    "type": SPLIT_TYPE,
                    "feature": feature,
                    "feature_name": f"feature_{feature}",
                    "gain": float(self._impurity_decrease(tree, node_id)),
                    "threshold": threshold,
                    "bin_edges": None,
                    "coefficients": None,
                    "intercept": None,
                    "chi_square": None,
                    "p_value": None,
                    "p_value_adjusted": None,
                    "degrees_of_freedom": None,
                    "condition": f"feature_{feature} <= {threshold:.4f}",
                }
                children = [left, right]
                edges.append(
                    {
                        "source": node_id,
                        "target": left,
                        "branch_index": 0,
                        "label": f"feature_{feature} <= {threshold:.4f}",
                    }
                )
                edges.append(
                    {
                        "source": node_id,
                        "target": right,
                        "branch_index": 1,
                        "label": f"feature_{feature} > {threshold:.4f}",
                    }
                )
            nodes.append(
                {
                    "id": node_id,
                    "depth": int(depths[node_id]),
                    "is_leaf": bool(is_leaf),
                    "n_samples": int(tree.n_node_samples[node_id]),
                    "impurity": float(tree.impurity[node_id]),
                    "impurity_measure": self.criterion,
                    "class_distribution": {
                        str(class_labels[i]): int(counts[i])
                        for i in range(n_classes)
                    },
                    "class_probabilities": [float(p) for p in probabilities],
                    "predicted_class": class_labels[int(np.argmax(counts))],
                    "split": split,
                    "children": children,
                }
            )

        leaves = sum(1 for node in nodes if node["is_leaf"])
        return {
            "tree_structure": {
                "algorithm": "CART",
                "split_type": SPLIT_TYPE,
                "impurity_measure": self.criterion,
                "root_id": 0,
                "n_nodes": len(nodes),
                "n_leaves": leaves,
                "max_depth_reached": int(depths.max()),
                "n_features": int(self.n_features),
                "feature_names": [
                    f"feature_{i}" for i in range(int(self.n_features))
                ],
                "classes": class_labels,
                "nodes": nodes,
                "edges": edges,
            }
        }

    @staticmethod
    def _impurity_decrease(tree: Any, node_id: int) -> float:
        """Weighted impurity decrease produced by the split at ``node_id``."""
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        n_parent = float(tree.weighted_n_node_samples[node_id])
        if n_parent <= 0.0:  # pragma: no cover - defensive
            return 0.0
        n_left = float(tree.weighted_n_node_samples[left])
        n_right = float(tree.weighted_n_node_samples[right])
        return float(
            tree.impurity[node_id]
            - (n_left / n_parent) * tree.impurity[left]
            - (n_right / n_parent) * tree.impurity[right]
        )
