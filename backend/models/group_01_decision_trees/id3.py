"""ID3 decision tree classifier (Group 01, ML integration platform).

Genuine ID3 as described by Quinlan (1986):

* splitting criterion is Shannon **entropy / information gain** (not Gini,
  not gain ratio -- gain ratio would make it C4.5),
* splits are **multi-way**: one child per attribute value,
* an attribute is **consumed**: it is never reused further down the same
  root-to-leaf path,
* the tree is grown until a stopping rule fires; there is **no pruning**.

ID3 is defined for *discrete* attributes.  The backend hands this model
continuous ``float64`` features (Section 4 of the coding standards), so the
attribute alphabet is created inside the algorithm: at ``fit`` time each
column is discretised into at most ``n_bins`` ordinal bins from the
*training split only*, and the resulting bin edges are stored on the model
and reused unchanged by ``predict``.  This discretisation is part of the ID3
algorithm itself, not preprocessing of the feature values: nothing is
scaled, encoded or imputed, and no statistic is ever computed from data
passed to ``predict``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from models.base_model import BaseModel

MODEL_NAME = "ID3 Decision Tree"
SPLIT_TYPE = "multiway_binned"
IMPURITY_MEASURE = "entropy"


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a vector of class counts."""
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _bin_edges_for_column(column: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges for one column, derived from training values only.

    Few distinct values -> midpoints between them (so no bin is empty).
    Many distinct values -> interior quantiles, de-duplicated.
    A constant column yields an empty edge array and is therefore never a
    usable splitting attribute.
    """
    unique = np.unique(column)
    if unique.size <= 1:
        return np.empty(0, dtype=np.float64)
    if unique.size <= n_bins:
        return ((unique[:-1] + unique[1:]) / 2.0).astype(np.float64)
    quantile_levels = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(column, quantile_levels))
    edges = edges[(edges >= unique[0]) & (edges < unique[-1])]
    if edges.size == 0:
        edges = np.array([(unique[0] + unique[-1]) / 2.0], dtype=np.float64)
    return edges.astype(np.float64)


def _json_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class ID3Model(BaseModel):
    """ID3 classifier: entropy-based, multi-way, attribute-consuming tree.

    Parameters
    ----------
    max_depth:
        Maximum number of splits on any root-to-leaf path.  ID3 splits
        multi-way and does not prune, so a level costs up to ``n_bins``
        times as many nodes as a binary level; the default is
        correspondingly shallower than CART's.
    min_samples_split:
        A node with fewer samples than this becomes a leaf.
    min_samples_leaf:
        An attribute is rejected at a node if any of the children it would
        create holds fewer samples than this.
    n_bins:
        Maximum number of ordinal bins per feature, i.e. the size of the
        attribute alphabet ID3 sees.
    min_information_gain:
        A node is only split when the best information gain exceeds this
        value, so zero-gain splits never happen.
    random_state:
        Accepted for interface consistency and reported in the metadata.
        This implementation contains no stochastic component, so the value
        does not change the fitted tree.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 20,
        min_samples_leaf: int = 1,
        n_bins: int = 4,
        min_information_gain: float = 0.0,
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
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError(f"n_bins must be an int >= 2, got {n_bins!r}.")
        if min_information_gain < 0.0:
            raise ValueError(
                f"min_information_gain must be >= 0, got {min_information_gain!r}."
            )
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_bins=n_bins,
            min_information_gain=min_information_gain,
            random_state=random_state,
            **kwargs,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.min_information_gain = min_information_gain
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self._nodes: list[dict[str, Any]] = []
        self._bin_edges: list[np.ndarray] = []
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
    # discretisation
    # ------------------------------------------------------------------
    def _digitize(self, X: np.ndarray) -> np.ndarray:
        """Map continuous columns onto the ordinal bins learnt at fit time."""
        binned = np.empty(X.shape, dtype=np.int64)
        for feature, edges in enumerate(self._bin_edges):
            binned[:, feature] = np.searchsorted(edges, X[:, feature], side="right")
        return binned

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ID3Model:
        """Grow the ID3 tree and return ``self``."""
        X, y = self._check_fit_inputs(X, y)
        start = time.perf_counter()

        self.classes_ = np.unique(y)
        n_classes = int(self.classes_.size)
        y_index = np.searchsorted(self.classes_, y).astype(np.int64)

        n_features = int(X.shape[1])
        self._bin_edges = [
            _bin_edges_for_column(X[:, j], self.n_bins) for j in range(n_features)
        ]
        binned = self._digitize(X)

        self._nodes = []
        self._importance = np.zeros(n_features, dtype=np.float64)
        self._n_train = int(X.shape[0])
        self._n_classes = n_classes
        self._build(
            binned,
            y_index,
            np.arange(self._n_train, dtype=np.int64),
            depth=0,
            available=set(range(n_features)),
        )

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
            "impurity": _entropy(counts),
            "class_counts": counts.astype(np.int64),
            "predicted_index": int(np.argmax(counts)),
            "feature": None,
            "gain": None,
            "children": {},
        }

    def _build(
        self,
        binned: np.ndarray,
        y_index: np.ndarray,
        rows: np.ndarray,
        depth: int,
        available: set[int],
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
            or not available
        ):
            return node_id

        best_feature: int | None = None
        best_gain = 0.0
        best_groups: dict[int, np.ndarray] = {}
        parent_entropy = node["impurity"]

        for feature in sorted(available):
            column = binned[rows, feature]
            bin_values = np.unique(column)
            if bin_values.size < 2:
                continue
            groups: dict[int, np.ndarray] = {}
            weighted_child_entropy = 0.0
            usable = True
            for bin_value in bin_values:
                subset = rows[column == bin_value]
                if subset.size < self.min_samples_leaf:
                    usable = False
                    break
                groups[int(bin_value)] = subset
                child_counts = np.bincount(
                    y_index[subset], minlength=self._n_classes
                )
                weighted_child_entropy += (
                    subset.size / n_samples
                ) * _entropy(child_counts)
            if not usable:
                continue
            gain = parent_entropy - weighted_child_entropy
            if best_feature is None or gain > best_gain + 1e-12:
                best_feature = feature
                best_gain = gain
                best_groups = groups

        if best_feature is None or best_gain <= self.min_information_gain:
            return node_id

        node["is_leaf"] = False
        node["feature"] = int(best_feature)
        node["gain"] = float(best_gain)
        self._importance[best_feature] += (
            n_samples / self._n_train
        ) * float(best_gain)

        child_available = available - {best_feature}
        children: dict[int, int] = {}
        for bin_value in sorted(best_groups):
            children[bin_value] = self._build(
                binned,
                y_index,
                best_groups[bin_value],
                depth + 1,
                child_available,
            )
        node["children"] = children
        return node_id

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _route(self, binned_row: np.ndarray) -> dict[str, Any]:
        """Walk one sample to a leaf.

        If a node never saw the sample's bin during training (an unseen
        attribute value) traversal stops there and the node's own class
        distribution is used -- the classical ID3 answer for unseen values.
        """
        node = self._nodes[0]
        while not node["is_leaf"]:
            child_id = node["children"].get(int(binned_row[node["feature"]]))
            if child_id is None:
                break
            node = self._nodes[child_id]
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels, shape ``(n_samples,)``."""
        X = self._check_predict_input(X)
        binned = self._digitize(X)
        indices = np.fromiter(
            (self._route(row)["predicted_index"] for row in binned),
            dtype=np.int64,
            count=binned.shape[0],
        )
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, shape ``(n_samples, n_classes)``."""
        X = self._check_predict_input(X)
        binned = self._digitize(X)
        proba = np.zeros((binned.shape[0], self._n_classes), dtype=np.float64)
        for row_index, row in enumerate(binned):
            counts = self._route(row)["class_counts"].astype(np.float64)
            total = counts.sum()
            if total <= 0.0:
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

    def _bin_label(self, feature: int, bin_value: int) -> str:
        edges = self._bin_edges[feature]
        name = f"feature_{feature}"
        if edges.size == 0:
            return f"{name} (constant)"
        if bin_value == 0:
            return f"{name} <= {edges[0]:.4f}"
        if bin_value >= edges.size:
            return f"{name} > {edges[-1]:.4f}"
        return f"{edges[bin_value - 1]:.4f} < {name} <= {edges[bin_value]:.4f}"

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
                feature = int(node["feature"])
                split = {
                    "type": SPLIT_TYPE,
                    "feature": feature,
                    "feature_name": f"feature_{feature}",
                    "gain": float(node["gain"]),
                    "threshold": None,
                    "bin_edges": self._bin_edges[feature].tolist(),
                    "coefficients": None,
                    "intercept": None,
                    "chi_square": None,
                    "p_value": None,
                    "p_value_adjusted": None,
                    "degrees_of_freedom": None,
                    "condition": (
                        f"multi-way split on feature_{feature} "
                        f"({len(node['children'])} bins)"
                    ),
                }
                for branch, (bin_value, child_id) in enumerate(
                    sorted(node["children"].items())
                ):
                    edges.append(
                        {
                            "source": int(node["id"]),
                            "target": int(child_id),
                            "branch_index": int(branch),
                            "label": self._bin_label(feature, int(bin_value)),
                        }
                    )
            nodes.append(
                {
                    "id": int(node["id"]),
                    "depth": int(node["depth"]),
                    "is_leaf": bool(node["is_leaf"]),
                    "n_samples": int(node["n_samples"]),
                    "impurity": float(node["impurity"]),
                    "impurity_measure": IMPURITY_MEASURE,
                    "class_distribution": {
                        str(class_labels[i]): int(node["class_counts"][i])
                        for i in range(len(class_labels))
                    },
                    "class_probabilities": [float(p) for p in probabilities],
                    "predicted_class": class_labels[node["predicted_index"]],
                    "split": split,
                    "children": (
                        [int(c) for _, c in sorted(node["children"].items())]
                        if not node["is_leaf"]
                        else []
                    ),
                }
            )

        leaves = sum(1 for node in nodes if node["is_leaf"])
        return {
            "tree_structure": {
                "algorithm": "ID3",
                "split_type": SPLIT_TYPE,
                "impurity_measure": IMPURITY_MEASURE,
                "root_id": 0,
                "n_nodes": len(nodes),
                "n_leaves": leaves,
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
