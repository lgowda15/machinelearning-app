"""Stage 4 -- metrics switch (DATA_FLOW_GUIDE.md SS5.1, .claude/rules/backend.md).

`model_type` from get_metadata() is not documentation, it's a switch
(CLAUDE.md "Hard rules"): it decides which metrics get computed here. All
four branches take the model's predict(X_test) output and compute metrics
from it uniformly -- never from training data -- so an unrecognised
model_type raises rather than falling through to a default.

Generic definitions used where the BaseModel contract doesn't expose a
model-internal value:

- Clusterer `inertia` is not a BaseModel attribute (e.g. DBSCAN has no
  `.inertia_`), so it's computed here as the within-cluster sum of squared
  distances from each non-noise point (label != -1, CLAUDE.md contract
  exceptions) to its own cluster's centroid, from X_test + predicted
  labels alone. None if every point is noise.
- Dimensionality-reducer `explained_variance_ratio` is computed the same
  way, generically: per-component variance of the transformed output
  divided by total variance of X_test. This does not assume PCA and will
  not necessarily sum to 1 for a non-PCA reducer -- it's descriptive, not
  a guarantee. It is a separate value from a PCA model's own
  get_visualization_data()["explained_variance_ratio"], which is the
  model's real internal value; both may be present and may differ.

`compute_plot_data` is the per-sample counterpart, for the screen-5 charts
that scalar metrics can't feed (frontend.md's per-type chart contract:
cluster scatter, predicted-vs-actual). It is backend-generic, never
model-owned -- kept off `metrics` (whose keys are asserted exactly by the
API contract tests) and off `visualization_data` (which
DATA_FLOW_GUIDE.md SS5.3 reserves for whatever a model itself returns).
Classifier and dimensionality_reducer return None here: the confusion
matrix and explained-variance chart are already fully fed by `metrics`.
Clusterer scatter needs 2D points regardless of how many features X_test
actually has, so a fixed, seeded, model-agnostic projection is used for
display only -- PCA(n_components=2) when there are enough features/samples
to support it, otherwise a degenerate placement that never fabricates a
second axis of real information.
"""
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)

VALID_MODEL_TYPES = {"classifier", "clusterer", "regressor", "dimensionality_reducer"}


def _classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted({*np.unique(y_true).tolist(), *np.unique(y_pred).tolist()})
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }


def _regressor_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _cluster_inertia(X: np.ndarray, labels: np.ndarray) -> float | None:
    non_noise = labels != -1
    if not non_noise.any():
        return None
    total = 0.0
    for cluster_id in np.unique(labels[non_noise]):
        members = X[labels == cluster_id]
        centroid = members.mean(axis=0)
        total += float(np.sum((members - centroid) ** 2))
    return total


def _clusterer_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    n_unique = np.unique(labels).shape[0]
    # Silhouette and Davies-Bouldin are undefined with fewer than 2 clusters
    # or as many clusters as samples -- a real mathematical non-applicability,
    # not an error to swallow.
    definable = 2 <= n_unique < labels.shape[0]
    return {
        "silhouette_score": float(silhouette_score(X, labels)) if definable else None,
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)) if definable else None,
        "inertia": _cluster_inertia(X, labels),
    }


def _dimensionality_reducer_metrics(X: np.ndarray, transformed: np.ndarray) -> dict[str, Any]:
    component_variance = transformed.var(axis=0, ddof=0)
    total_variance = float(X.var(axis=0, ddof=0).sum())
    if total_variance == 0.0:
        ratio = [None] * component_variance.shape[0]
    else:
        ratio = (component_variance / total_variance).tolist()
    return {"explained_variance_ratio": ratio}


def compute_metrics(
    model_type: str,
    X: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Route to the metrics for `model_type`, computed from X_test and
    predict(X_test) uniformly. Raises on an unrecognised model_type rather
    than falling through to a default (.claude/rules/backend.md)."""
    if model_type == "classifier":
        if y_true is None:
            raise ValueError("Classifier metrics require y_true.")
        return _classifier_metrics(y_true, y_pred)
    if model_type == "regressor":
        if y_true is None:
            raise ValueError("Regressor metrics require y_true.")
        return _regressor_metrics(y_true, y_pred)
    if model_type == "clusterer":
        return _clusterer_metrics(X, y_pred)
    if model_type == "dimensionality_reducer":
        return _dimensionality_reducer_metrics(X, y_pred)
    raise ValueError(f"Unrecognised model_type: {model_type!r}")


def _cluster_scatter_points(X: np.ndarray) -> list[list[float]]:
    """2D points for the cluster-scatter chart, for display only -- never
    treated as a real embedding. PCA(n_components=2, random_state=42) when
    there's enough shape to support it; otherwise place points along
    whatever single axis exists (or the origin) rather than fabricate a
    second axis of information that isn't there."""
    n_samples, n_features = X.shape
    if n_features >= 2 and n_samples >= 2:
        return PCA(n_components=2, random_state=42).fit_transform(X).tolist()
    x = X[:, 0] if n_features >= 1 else np.zeros(n_samples)
    y = np.zeros(n_samples)
    return np.column_stack([x, y]).tolist()


def compute_plot_data(
    model_type: str,
    X: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
) -> dict[str, Any] | None:
    """Per-sample data for the screen-5 charts scalar `metrics` can't feed
    (frontend.md's per-type chart contract): predicted-vs-actual pairs for
    regressors, 2D scatter points for clusterers. Backend-generic, computed
    from X_test/y_test/predictions alone, same as `metrics`' own generic
    branches -- never from the model. None for classifier and
    dimensionality_reducer, whose charts (confusion matrix, explained
    variance) are already fully fed by `metrics`."""
    if model_type in ("classifier", "dimensionality_reducer"):
        return None
    if model_type == "regressor":
        if y_true is None:
            raise ValueError("Regressor plot data requires y_true.")
        return {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}
    if model_type == "clusterer":
        return {"points": _cluster_scatter_points(X), "labels": y_pred.tolist()}
    raise ValueError(f"Unrecognised model_type: {model_type!r}")
