"""Sparse oblique decision tree classifier (Group 01, "latest technique").

Why this technique
-----------------
ID3, CART and CHAID are all *axis-parallel*: every test looks at one
feature at a time, so a boundary such as ``0.7 * x0 - 0.4 * x1 <= 0.2`` can
only be approximated by a staircase of many single-feature cuts.  That
makes the tree deeper, less stable and harder to read than the boundary
actually is.  **Oblique** (multivariate) decision trees test a *linear
combination* of features at each node instead, and the sparse-oblique line
of work -- Murthy's OC1, Wickramarachchi's HHCART (2016), and
Carreira-Perpinan & Tavallali's Tree Alternating Optimization (NeurIPS
2018), continued in the sparse-oblique-tree literature since -- shows that
such trees reach the accuracy of much larger axis-parallel trees while
staying small enough to read.

This implementation follows that idea in the form that fits the platform's
constraints:

* at every node an **L1-regularised logistic regression** is fitted to a
  one-vs-rest recoding of the node's labels, giving a *sparse* hyperplane
  ``w`` (most coefficients are exactly zero, so the split stays readable),
* the samples are projected onto ``w`` and the best threshold on that
  projection is chosen by impurity decrease, exactly as CART chooses a
  threshold on a single feature,
* an **axis-parallel candidate is always evaluated too** and wins ties, so
  the model degrades gracefully to a CART-style split whenever a
  hyperplane does not actually help.

Both split kinds share one traversal rule -- ``x . w + b <= threshold``
goes left -- because an axis-parallel split is just a hyperplane whose
weight vector is a unit vector.  Everything is CPU-only, deterministic
under ``random_state=42``, and needs nothing beyond scikit-learn.

Linear splits assume comparable feature scales, which is exactly what the
input contract in Section 4 of the coding standards guarantees (the backend
has already applied ``StandardScaler``).  This model itself performs no
scaling, encoding or imputation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel

MODEL_NAME = "Sparse Oblique Decision Tree"
SPLIT_TYPE_OBLIQUE = "binary_oblique"
SPLIT_TYPE_AXIS = "binary_axis_parallel"
VALID_CRITERIA = ("gini", "entropy")
MAX_SHOWN_TERMS = 6


def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Row-wise Gini or entropy impurity of a 2D array of class counts."""
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    totals = counts.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals <= 0.0, 1.0, totals)
    probs = counts / safe_totals
    if criterion == "gini":
        return 1.0 - (probs**2).sum(axis=1)
    log_probs = np.zeros_like(probs)
    np.log2(probs, out=log_probs, where=probs > 0.0)
    return -(probs * log_probs).sum(axis=1)


def _json_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class ObliqueDecisionTreeModel(BaseModel):
    """Sparse oblique decision tree: hyperplane splits with axis fallback.

    Parameters
    ----------
    max_depth:
        Maximum number of splits on any root-to-leaf path.
    min_samples_split:
        A node with fewer samples than this becomes a leaf.  Hyperplanes
        need a reasonable number of samples, so the default is larger than
        a CART default.
    min_samples_leaf:
        Minimum samples that must remain on each side of a split.
    criterion:
        ``"gini"`` or ``"entropy"``; used to score candidate thresholds.
    oblique_C:
        Inverse L1 regularisation strength of the per-node logistic
        regression.  Smaller values give sparser (fewer non-zero
        coefficients), more readable hyperplanes.
    max_hyperplane_iter:
        Maximum solver iterations for the per-node logistic regression.
    min_impurity_decrease:
        A split is accepted only if it reduces weighted impurity by more
        than this value.
    random_state:
        Seed passed to the per-node solver; fixes the fitted tree.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        criterion: str = "gini",
        oblique_C: float = 1.0,
        max_hyperplane_iter: int = 200,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
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
        if criterion not in VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {VALID_CRITERIA}, got {criterion!r}."
            )
        if oblique_C <= 0.0:
            raise ValueError(f"oblique_C must be > 0, got {oblique_C!r}.")
        if not isinstance(max_hyperplane_iter, int) or max_hyperplane_iter < 1:
            raise ValueError(
                "max_hyperplane_iter must be an int >= 1, got "
                f"{max_hyperplane_iter!r}."
            )
        if min_impurity_decrease < 0.0:
            raise ValueError(
                f"min_impurity_decrease must be >= 0, got {min_impurity_decrease!r}."
            )
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            oblique_C=oblique_C,
            max_hyperplane_iter=max_hyperplane_iter,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            **kwargs,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.oblique_C = oblique_C
        self.max_hyperplane_iter = max_hyperplane_iter
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self._nodes: list[dict[str, Any]] = []
        self._importance: np.ndarray | None = None
        self._n_train: int = 0
        self._n_classes: int = 0
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
    # split search
    # ------------------------------------------------------------------
    def _best_threshold(
        self, projection: np.ndarray, y_index: np.ndarray
    ) -> tuple[float, float] | None:
        """Best threshold on a 1D projection.

        Returns ``(threshold, weighted_child_impurity)`` or ``None`` when no
        threshold respects ``min_samples_leaf``.
        """
        n_samples = projection.size
        order = np.argsort(projection, kind="stable")
        sorted_projection = projection[order]
        sorted_labels = y_index[order]

        one_hot = np.zeros((n_samples, self._n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), sorted_labels] = 1.0
        cumulative = np.cumsum(one_hot, axis=0)
        totals = cumulative[-1]

        boundaries = np.nonzero(sorted_projection[:-1] < sorted_projection[1:])[0]
        if boundaries.size == 0:
            return None
        left_sizes = boundaries + 1
        right_sizes = n_samples - left_sizes
        keep = (left_sizes >= self.min_samples_leaf) & (
            right_sizes >= self.min_samples_leaf
        )
        boundaries = boundaries[keep]
        if boundaries.size == 0:
            return None
        left_sizes = boundaries + 1
        right_sizes = n_samples - left_sizes

        left_counts = cumulative[boundaries]
        right_counts = totals - left_counts
        weighted = (
            left_sizes * _impurity(left_counts, self.criterion)
            + right_sizes * _impurity(right_counts, self.criterion)
        ) / n_samples
        best = int(np.argmin(weighted))
        boundary = int(boundaries[best])
        threshold = float(
            (sorted_projection[boundary] + sorted_projection[boundary + 1]) / 2.0
        )
        return threshold, float(weighted[best])

    def _axis_candidate(
        self, X_node: np.ndarray, y_index: np.ndarray
    ) -> dict[str, Any] | None:
        """Best single-feature split, used as the fallback candidate."""
        best: dict[str, Any] | None = None
        for feature in range(X_node.shape[1]):
            result = self._best_threshold(X_node[:, feature], y_index)
            if result is None:
                continue
            threshold, cost = result
            if best is None or cost < best["cost"] - 1e-12:
                weights = np.zeros(X_node.shape[1], dtype=np.float64)
                weights[feature] = 1.0
                best = {
                    "split_type": SPLIT_TYPE_AXIS,
                    "feature": feature,
                    "weights": weights,
                    "intercept": 0.0,
                    "threshold": threshold,
                    "cost": cost,
                }
        return best

    def _oblique_candidate(
        self, X_node: np.ndarray, y_index: np.ndarray
    ) -> dict[str, Any] | None:
        """Best sparse-hyperplane split found by one-vs-rest L1 logistic fits."""
        present = np.unique(y_index)
        targets = present[:-1] if present.size == 2 else present
        best: dict[str, Any] | None = None
        for target in targets:
            binary = (y_index == target).astype(np.int64)
            if np.unique(binary).size < 2:
                continue
            solver = LogisticRegression(
                penalty="l1",
                C=self.oblique_C,
                solver="liblinear",
                max_iter=self.max_hyperplane_iter,
                random_state=self.random_state,
            )
            solver.fit(X_node, binary)
            weights = np.asarray(solver.coef_[0], dtype=np.float64)
            if not np.any(np.abs(weights) > 0.0):
                continue
            intercept = float(solver.intercept_[0])
            result = self._best_threshold(X_node @ weights + intercept, y_index)
            if result is None:
                continue
            threshold, cost = result
            if best is None or cost < best["cost"] - 1e-12:
                best = {
                    "split_type": SPLIT_TYPE_OBLIQUE,
                    "feature": None,
                    "weights": weights,
                    "intercept": intercept,
                    "threshold": threshold,
                    "cost": cost,
                }
        return best

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> ObliqueDecisionTreeModel:
        """Grow the sparse oblique tree and return ``self``."""
        X, y = self._check_fit_inputs(X, y)
        start = time.perf_counter()

        self.classes_ = np.unique(y)
        self._n_classes = int(self.classes_.size)
        y_index = np.searchsorted(self.classes_, y).astype(np.int64)

        n_features = int(X.shape[1])
        self._nodes = []
        self._importance = np.zeros(n_features, dtype=np.float64)
        self._n_train = int(X.shape[0])
        self._build(X, y_index, np.arange(self._n_train, dtype=np.int64), depth=0)

        self._train_time = time.perf_counter() - start
        self.n_features = n_features
        self.is_fitted = True
        return self

    def _new_node(
        self, depth: int, rows: np.ndarray, counts: np.ndarray
    ) -> dict[str, Any]:
        return {
            "id": len(self._nodes),
            "depth": int(depth),
            "is_leaf": True,
            "n_samples": int(rows.size),
            "impurity": float(_impurity(counts, self.criterion)[0]),
            "class_counts": counts.astype(np.int64),
            "predicted_index": int(np.argmax(counts)),
            "split_type": None,
            "feature": None,
            "weights": None,
            "intercept": None,
            "threshold": None,
            "gain": None,
            "children": [],
        }

    def _build(
        self,
        X: np.ndarray,
        y_index: np.ndarray,
        rows: np.ndarray,
        depth: int,
    ) -> int:
        counts = np.bincount(y_index[rows], minlength=self._n_classes)
        node = self._new_node(depth, rows, counts)
        node_id = node["id"]
        self._nodes.append(node)

        n_samples = int(rows.size)
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or int((counts > 0).sum()) <= 1
            or n_samples < 2 * self.min_samples_leaf
        ):
            return node_id

        X_node = X[rows, :]
        y_node = y_index[rows]
        axis = self._axis_candidate(X_node, y_node)
        oblique = self._oblique_candidate(X_node, y_node)

        candidate = axis
        if oblique is not None and (
            axis is None or oblique["cost"] < axis["cost"] - 1e-12
        ):
            candidate = oblique
        if candidate is None:
            return node_id

        gain = node["impurity"] - candidate["cost"]
        if gain <= self.min_impurity_decrease:
            return node_id

        projection = X_node @ candidate["weights"] + candidate["intercept"]
        go_left = projection <= candidate["threshold"]
        left_rows = rows[go_left]
        right_rows = rows[~go_left]
        if left_rows.size == 0 or right_rows.size == 0:  # pragma: no cover
            return node_id

        node["is_leaf"] = False
        node["split_type"] = candidate["split_type"]
        node["feature"] = candidate["feature"]
        node["weights"] = candidate["weights"]
        node["intercept"] = float(candidate["intercept"])
        node["threshold"] = float(candidate["threshold"])
        node["gain"] = float(gain)

        magnitudes = np.abs(candidate["weights"])
        magnitude_sum = magnitudes.sum()
        if magnitude_sum > 0.0:
            self._importance += (
                (n_samples / self._n_train) * gain * magnitudes / magnitude_sum
            )

        node["children"] = [
            self._build(X, y_index, left_rows, depth + 1),
            self._build(X, y_index, right_rows, depth + 1),
        ]
        return node_id

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """Vectorised traversal: node id reached by each sample."""
        assigned = np.zeros(X.shape[0], dtype=np.int64)
        frontier = [(0, np.arange(X.shape[0], dtype=np.int64))]
        while frontier:
            node_id, rows = frontier.pop()
            node = self._nodes[node_id]
            if node["is_leaf"] or rows.size == 0:
                assigned[rows] = node_id
                continue
            projection = X[rows, :] @ node["weights"] + node["intercept"]
            go_left = projection <= node["threshold"]
            frontier.append((node["children"][0], rows[go_left]))
            frontier.append((node["children"][1], rows[~go_left]))
        return assigned

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels, shape ``(n_samples,)``."""
        X = self._check_predict_input(X)
        leaves = self._leaf_indices(X)
        indices = np.array(
            [self._nodes[node_id]["predicted_index"] for node_id in leaves],
            dtype=np.int64,
        )
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, shape ``(n_samples, n_classes)``."""
        X = self._check_predict_input(X)
        leaves = self._leaf_indices(X)
        proba = np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        for row_index, node_id in enumerate(leaves):
            counts = self._nodes[node_id]["class_counts"].astype(np.float64)
            total = counts.sum()
            if total <= 0.0:  # pragma: no cover - defensive
                proba[row_index, :] = 1.0 / self._n_classes
            else:
                proba[row_index, :] = counts / total
        return proba

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
        if not self.is_fitted or self._importance is None:
            return None
        scores = self._importance.astype(np.float64)
        total = scores.sum()
        if total > 0.0:
            scores = scores / total
        return {
            f"feature_{i}": float(scores[i]) for i in range(int(self.n_features))
        }

    @staticmethod
    def _expression_label(node: dict[str, Any]) -> str:
        """Readable form of the quantity a split node compares."""
        if node["split_type"] == SPLIT_TYPE_AXIS:
            return f"feature_{int(node['feature'])}"
        weights = node["weights"]
        active = np.nonzero(np.abs(weights) > 0.0)[0]
        terms = " ".join(
            f"{weights[j]:+.4f}*feature_{int(j)}" for j in active[:MAX_SHOWN_TERMS]
        )
        if active.size > MAX_SHOWN_TERMS:
            terms += f" ... ({active.size - MAX_SHOWN_TERMS} more terms)"
        return f"({terms}) {float(node['intercept']):+.4f}"

    def get_visualization_data(self) -> dict[str, Any] | None:
        """Return the JSON-serialisable ``tree_structure`` payload."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Call fit() before get_visualization_data() on {MODEL_NAME}."
            )
        class_labels = [_json_scalar(c) for c in self.classes_]
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for node in self._nodes:
            counts = node["class_counts"].astype(np.float64)
            total = counts.sum()
            probabilities = (
                (counts / total).tolist()
                if total > 0
                else [1.0 / len(class_labels)] * len(class_labels)
            )
            split: dict[str, Any] | None = None
            if not node["is_leaf"]:
                expression = self._expression_label(node)
                threshold = float(node["threshold"])
                condition = f"{expression} <= {threshold:.4f}"
                feature = node["feature"]
                split = {
                    "type": node["split_type"],
                    "feature": None if feature is None else int(feature),
                    "feature_name": (
                        None if feature is None else f"feature_{int(feature)}"
                    ),
                    "gain": float(node["gain"]),
                    "threshold": threshold,
                    "bin_edges": None,
                    "coefficients": [float(w) for w in node["weights"]],
                    "intercept": float(node["intercept"]),
                    "chi_square": None,
                    "p_value": None,
                    "p_value_adjusted": None,
                    "degrees_of_freedom": None,
                    "condition": condition,
                }
                edges.append(
                    {
                        "source": int(node["id"]),
                        "target": int(node["children"][0]),
                        "branch_index": 0,
                        "label": condition,
                    }
                )
                edges.append(
                    {
                        "source": int(node["id"]),
                        "target": int(node["children"][1]),
                        "branch_index": 1,
                        "label": f"{expression} > {threshold:.4f}",
                    }
                )
            nodes.append(
                {
                    "id": int(node["id"]),
                    "depth": int(node["depth"]),
                    "is_leaf": bool(node["is_leaf"]),
                    "n_samples": int(node["n_samples"]),
                    "impurity": float(node["impurity"]),
                    "impurity_measure": self.criterion,
                    "class_distribution": {
                        str(class_labels[i]): int(node["class_counts"][i])
                        for i in range(len(class_labels))
                    },
                    "class_probabilities": [float(p) for p in probabilities],
                    "predicted_class": class_labels[node["predicted_index"]],
                    "split": split,
                    "children": [int(c) for c in node["children"]],
                }
            )

        leaves = sum(1 for node in nodes if node["is_leaf"])
        oblique_splits = sum(
            1
            for node in nodes
            if node["split"] is not None
            and node["split"]["type"] == SPLIT_TYPE_OBLIQUE
        )
        return {
            "tree_structure": {
                "algorithm": "Sparse Oblique Decision Tree",
                "split_type": SPLIT_TYPE_OBLIQUE,
                "impurity_measure": self.criterion,
                "root_id": 0,
                "n_nodes": len(nodes),
                "n_leaves": leaves,
                "n_oblique_splits": oblique_splits,
                "max_depth_reached": max(node["depth"] for node in nodes),
                "n_features": int(self.n_features),
                "feature_names": [
                    f"feature_{i}" for i in range(int(self.n_features))
                ],
                "classes": class_labels,
                "nodes": nodes,
                "edges": edges,
            }
        }
