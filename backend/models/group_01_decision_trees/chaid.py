"""CHAID decision tree classifier (Group 01, ML integration platform).

CHAID (Kass, 1980) is not an impurity-based algorithm at all.  It selects
splits by **statistical significance**:

1. every predictor is expressed as a set of ordered categories,
2. **adjacent categories are merged** while the chi-square test of the
   2 x C sub-table for the most similar neighbouring pair is not
   significant (``alpha_merge``),
3. the merged category-by-class contingency table is tested with a
   Pearson chi-square test and the p-value is **Bonferroni-adjusted** for
   the number of ways the original categories could have been merged,
4. the predictor with the smallest adjusted p-value splits the node, and
   only if that p-value is at most ``alpha_split``; otherwise the node
   becomes a leaf.  The split is **multi-way**: one child per merged
   category group.

How continuous input is handled
-------------------------------
The backend supplies continuous, already-scaled ``float64`` features
(Section 4 of the coding standards), while CHAID needs ordered categories.
This implementation therefore builds the categories **locally inside each
node** from that node's own training samples: up to ``max_bins``
equal-frequency (quantile) bins, which are then fed to CHAID's merge step.
Local binning is faithful to CHAID -- the merge step is precisely the
mechanism CHAID uses to collapse an over-fine ordinal predictor -- and it
lets the same feature be re-categorised differently in different parts of
the tree.

The bin edges and the merged groups are stored *on the node*, so they are
part of the split rule and not a global transformation of the feature
matrix: nothing is scaled, encoded or imputed, and nothing is ever fitted
on data passed to ``predict``.  A category that no training sample fell
into at a node (possible because the bin edges cover the whole real line)
is routed at predict time to the group of the nearest observed category,
which respects the ordinal nature of the binned feature.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.stats import chi2

from models.base_model import BaseModel

MODEL_NAME = "CHAID Decision Tree"
SPLIT_TYPE = "multiway_chi_square"
IMPURITY_MEASURE = "entropy"


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a vector of class counts."""
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _chi_square_test(table: np.ndarray) -> tuple[float, int, float]:
    """Pearson chi-square statistic, degrees of freedom and p-value."""
    table = np.asarray(table, dtype=np.float64)
    total = table.sum()
    if total <= 0.0:
        return 0.0, 0, 1.0
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    expected = (row_totals @ col_totals) / total
    mask = expected > 0.0
    statistic = float(
        (((table[mask] - expected[mask]) ** 2) / expected[mask]).sum()
    )
    n_rows = int((table.sum(axis=1) > 0).sum())
    n_cols = int((table.sum(axis=0) > 0).sum())
    dof = (n_rows - 1) * (n_cols - 1)
    if dof <= 0:
        return statistic, 0, 1.0
    return statistic, dof, float(chi2.sf(statistic, dof))


def _adjacent_p_values(rows: np.ndarray) -> np.ndarray:
    """Chi-square p-values of every adjacent pair of category groups.

    Vectorised equivalent of calling :func:`_chi_square_test` on each 2 x C
    sub-table in turn; the merge step runs this once per iteration.
    """
    upper = rows[:-1]
    lower = rows[1:]
    col_totals = upper + lower
    upper_totals = upper.sum(axis=1)
    lower_totals = lower.sum(axis=1)
    totals = upper_totals + lower_totals
    safe_totals = np.where(totals <= 0.0, 1.0, totals)[:, None]
    expected_upper = upper_totals[:, None] * col_totals / safe_totals
    expected_lower = lower_totals[:, None] * col_totals / safe_totals
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            expected_upper > 0.0,
            (upper - expected_upper) ** 2 / expected_upper,
            0.0,
        ) + np.where(
            expected_lower > 0.0,
            (lower - expected_lower) ** 2 / expected_lower,
            0.0,
        )
    statistics = terms.sum(axis=1)
    n_cols = (col_totals > 0).sum(axis=1)
    n_rows = (upper_totals > 0).astype(np.int64) + (lower_totals > 0).astype(
        np.int64
    )
    dof = (n_rows - 1) * (n_cols - 1)
    p_values = np.ones(statistics.size, dtype=np.float64)
    usable = dof > 0
    if usable.any():
        p_values[usable] = chi2.sf(statistics[usable], dof[usable])
    return p_values


def _merge_at(
    rows: np.ndarray, groups: list[list[int]], index: int
) -> tuple[np.ndarray, list[list[int]]]:
    """Merge category group ``index`` with its right-hand neighbour."""
    merged_row = rows[index] + rows[index + 1]
    rows = np.vstack(
        [rows[:index], merged_row[None, :], rows[index + 2 :]]
    )
    groups = (
        groups[:index]
        + [groups[index] + groups[index + 1]]
        + groups[index + 2 :]
    )
    return rows, groups


def _bin_edges_for_values(values: np.ndarray, max_bins: int) -> np.ndarray:
    """Interior bin edges for one column of node-local values."""
    unique = np.unique(values)
    if unique.size <= 1:
        return np.empty(0, dtype=np.float64)
    if unique.size <= max_bins:
        return ((unique[:-1] + unique[1:]) / 2.0).astype(np.float64)
    quantile_levels = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantile_levels))
    edges = edges[(edges >= unique[0]) & (edges < unique[-1])]
    if edges.size == 0:
        edges = np.array([(unique[0] + unique[-1]) / 2.0], dtype=np.float64)
    return edges.astype(np.float64)


def _json_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the equivalent Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class CHAIDModel(BaseModel):
    """CHAID classifier: chi-square driven, multi-way, merge-based tree.

    Parameters
    ----------
    max_depth:
        Maximum number of splits on any root-to-leaf path.  CHAID splits
        multi-way, and on a large sample the chi-square test finds almost
        any association significant, so the depth cap is the practical
        guard on tree size and is set shallower than CART's.
    min_samples_split:
        A node with fewer samples than this becomes a leaf.  CHAID relies
        on chi-square tests, so the default is deliberately larger than a
        CART default.
    min_samples_leaf:
        Merged category groups smaller than this are force-merged into a
        neighbour; a predictor that cannot satisfy this is rejected.
    max_bins:
        Maximum number of node-local equal-frequency categories created
        for a continuous predictor before merging starts.
    alpha_merge:
        Adjacent categories are merged while the most similar neighbouring
        pair has a chi-square p-value above this value.
    alpha_split:
        A node is split only if the best Bonferroni-adjusted p-value is at
        most this value.
    bonferroni:
        Apply Kass's Bonferroni adjustment to the p-values.
    random_state:
        Accepted for interface consistency and reported in the metadata.
        This implementation contains no stochastic component, so the value
        does not change the fitted tree.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_bins: int = 10,
        alpha_merge: float = 0.05,
        alpha_split: float = 0.05,
        bonferroni: bool = True,
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
        if not isinstance(max_bins, int) or max_bins < 2:
            raise ValueError(f"max_bins must be an int >= 2, got {max_bins!r}.")
        if not 0.0 < alpha_merge <= 1.0:
            raise ValueError(
                f"alpha_merge must lie in (0, 1], got {alpha_merge!r}."
            )
        if not 0.0 < alpha_split <= 1.0:
            raise ValueError(
                f"alpha_split must lie in (0, 1], got {alpha_split!r}."
            )
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            alpha_merge=alpha_merge,
            alpha_split=alpha_split,
            bonferroni=bool(bonferroni),
            random_state=random_state,
            **kwargs,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.alpha_merge = alpha_merge
        self.alpha_split = alpha_split
        self.bonferroni = bool(bonferroni)
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
    # training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CHAIDModel:
        """Grow the CHAID tree and return ``self``."""
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
            "impurity": _entropy(counts),
            "class_counts": counts.astype(np.int64),
            "predicted_index": int(np.argmax(counts)),
            "feature": None,
            "bin_edges": np.empty(0, dtype=np.float64),
            "groups": [],
            "category_map": [],
            "children": [],
            "chi_square": None,
            "dof": None,
            "p_value": None,
            "p_value_adjusted": None,
            "gain": None,
        }

    def _merge_categories(
        self, category_counts: np.ndarray
    ) -> list[list[int]]:
        """CHAID's adjacent-category merge step.

        ``category_counts`` has one row per *observed* category (ascending)
        and one column per class.  Returns the surviving groups as lists of
        row positions.
        """
        groups: list[list[int]] = [[i] for i in range(category_counts.shape[0])]
        rows = category_counts.astype(np.float64)

        # Merge the most similar adjacent pair while it is not significant.
        while len(groups) > 2:
            p_values = _adjacent_p_values(rows)
            index = int(np.argmax(p_values))
            if p_values[index] <= self.alpha_merge:
                break
            rows, groups = _merge_at(rows, groups, index)

        # Force-merge groups that are too small to become a leaf.
        while len(groups) > 1:
            sizes = rows.sum(axis=1)
            smallest = int(np.argmin(sizes))
            if sizes[smallest] >= self.min_samples_leaf:
                break
            if smallest == 0:
                index = 0
            elif smallest == len(groups) - 1:
                index = smallest - 1
            else:
                p_values = _adjacent_p_values(rows)
                index = (
                    smallest - 1
                    if p_values[smallest - 1] >= p_values[smallest]
                    else smallest
                )
            rows, groups = _merge_at(rows, groups, index)
        return groups

    def _evaluate_feature(
        self,
        values: np.ndarray,
        y_index: np.ndarray,
    ) -> dict[str, Any] | None:
        """Run the CHAID merge-and-test procedure for one predictor."""
        edges = _bin_edges_for_values(values, self.max_bins)
        if edges.size == 0:
            return None
        categories = np.searchsorted(edges, values, side="right")
        observed, positions = np.unique(categories, return_inverse=True)
        if observed.size < 2:
            return None

        # One-pass contingency table: rows are observed categories in
        # ascending order, columns are classes.
        flat = positions.ravel() * self._n_classes + y_index
        category_counts = np.bincount(
            flat, minlength=observed.size * self._n_classes
        ).reshape(observed.size, self._n_classes)

        groups = self._merge_categories(category_counts)
        if len(groups) < 2:
            return None

        merged = np.vstack(
            [category_counts[group, :].sum(axis=0) for group in groups]
        )
        statistic, dof, p_value = _chi_square_test(merged)
        if dof <= 0:
            return None
        multiplier = 1.0
        if self.bonferroni:
            multiplier = float(
                math.comb(int(observed.size) - 1, len(groups) - 1)
            )
        p_adjusted = min(1.0, p_value * multiplier)

        n_samples = float(merged.sum())
        parent_entropy = _entropy(merged.sum(axis=0))
        child_entropy = sum(
            (row.sum() / n_samples) * _entropy(row) for row in merged
        )
        return {
            "edges": edges,
            "observed": observed,
            "groups": groups,
            "chi_square": statistic,
            "dof": int(dof),
            "p_value": p_value,
            "p_value_adjusted": p_adjusted,
            "gain": float(parent_entropy - child_entropy),
        }

    @staticmethod
    def _category_map(
        observed: np.ndarray, groups: list[list[int]], n_categories: int
    ) -> list[int]:
        """Group index for every category, including never-observed ones."""
        position_to_group = {}
        for group_index, group in enumerate(groups):
            for position in group:
                position_to_group[int(observed[position])] = group_index
        seen = sorted(position_to_group)
        mapping: list[int] = []
        for category in range(n_categories):
            if category in position_to_group:
                mapping.append(position_to_group[category])
            else:
                nearest = min(seen, key=lambda c: (abs(c - category), c))
                mapping.append(position_to_group[nearest])
        return mapping

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
        ):
            return node_id

        best_feature: int | None = None
        best: dict[str, Any] | None = None
        best_key: tuple[float, float] | None = None
        for feature in range(X.shape[1]):
            result = self._evaluate_feature(X[rows, feature], y_index[rows])
            if result is None:
                continue
            # Rank by adjusted p-value, breaking exact ties by the stronger
            # chi-square statistic and then by the lower feature index.
            # No absolute tolerance is used: adjusted p-values are routinely
            # many orders of magnitude below any fixed epsilon (and underflow
            # to 0.0 on large samples), so comparing them with one would
            # collapse every significant predictor into a single tie and hand
            # the split to whichever feature happened to come first.
            key = (result["p_value_adjusted"], -result["chi_square"])
            if best_key is None or key < best_key:
                best_feature, best, best_key = feature, result, key

        if best is None or best["p_value_adjusted"] > self.alpha_split:
            return node_id

        edges = best["edges"]
        categories = np.searchsorted(edges, X[rows, best_feature], side="right")
        mapping = self._category_map(
            best["observed"], best["groups"], int(edges.size) + 1
        )
        mapping_array = np.asarray(mapping, dtype=np.int64)
        assigned = mapping_array[categories]

        node["is_leaf"] = False
        node["feature"] = int(best_feature)
        node["bin_edges"] = edges
        node["groups"] = [
            [int(best["observed"][p]) for p in group] for group in best["groups"]
        ]
        node["category_map"] = mapping
        node["chi_square"] = float(best["chi_square"])
        node["dof"] = int(best["dof"])
        node["p_value"] = float(best["p_value"])
        node["p_value_adjusted"] = float(best["p_value_adjusted"])
        node["gain"] = float(best["gain"])
        self._importance[best_feature] += (
            n_samples / self._n_train
        ) * float(best["gain"])

        children: list[int] = []
        for group_index in range(len(best["groups"])):
            subset = rows[assigned == group_index]
            children.append(self._build(X, y_index, subset, depth + 1))
        node["children"] = children
        return node_id

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _route(self, row: np.ndarray) -> dict[str, Any]:
        node = self._nodes[0]
        while not node["is_leaf"]:
            category = int(
                np.searchsorted(node["bin_edges"], row[node["feature"]], side="right")
            )
            group_index = node["category_map"][category]
            node = self._nodes[node["children"][group_index]]
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels, shape ``(n_samples,)``."""
        X = self._check_predict_input(X)
        indices = np.fromiter(
            (self._route(row)["predicted_index"] for row in X),
            dtype=np.int64,
            count=X.shape[0],
        )
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, shape ``(n_samples, n_classes)``."""
        X = self._check_predict_input(X)
        proba = np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        for row_index, row in enumerate(X):
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

    @staticmethod
    def _group_label(feature: int, edges: np.ndarray, group: list[int]) -> str:
        """Readable interval label for one merged category group."""
        name = f"feature_{feature}"
        low = min(group)
        high = max(group)
        lower = None if low == 0 else float(edges[low - 1])
        upper = None if high >= edges.size else float(edges[high])
        if lower is None and upper is None:
            return f"{name} (all values)"
        if lower is None:
            return f"{name} <= {upper:.4f}"
        if upper is None:
            return f"{name} > {lower:.4f}"
        return f"{lower:.4f} < {name} <= {upper:.4f}"

    def get_visualization_data(self) -> dict[str, Any] | None:
        """Return the JSON-serialisable ``tree_structure`` payload."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Call fit() before get_visualization_data() on {MODEL_NAME}."
            )
        class_labels = [_json_scalar(c) for c in self.classes_]
        nodes: list[dict[str, Any]] = []
        edges_out: list[dict[str, Any]] = []

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
                    "bin_edges": node["bin_edges"].tolist(),
                    "coefficients": None,
                    "intercept": None,
                    "chi_square": float(node["chi_square"]),
                    "p_value": float(node["p_value"]),
                    "p_value_adjusted": float(node["p_value_adjusted"]),
                    "degrees_of_freedom": int(node["dof"]),
                    "condition": (
                        f"chi-square split on feature_{feature} "
                        f"({len(node['children'])} merged groups, "
                        f"adjusted p={node['p_value_adjusted']:.4g})"
                    ),
                }
                for branch, child_id in enumerate(node["children"]):
                    edges_out.append(
                        {
                            "source": int(node["id"]),
                            "target": int(child_id),
                            "branch_index": int(branch),
                            "label": self._group_label(
                                feature, node["bin_edges"], node["groups"][branch]
                            ),
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
                    "children": [int(c) for c in node["children"]],
                }
            )

        leaves = sum(1 for node in nodes if node["is_leaf"])
        return {
            "tree_structure": {
                "algorithm": "CHAID",
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
                "edges": edges_out,
            }
        }
