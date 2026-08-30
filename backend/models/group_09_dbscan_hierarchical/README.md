# Group 9 — DBSCAN & Hierarchical Clustering

**Course:** UM25MB653CA2
**Folder:** `backend/models/group_09_dbscan_hierarchical/`
**Stack:** Python 3.12, scikit-learn 1.9.0, SciPy, CPU only

## Description

This package provides two enterprise-contract-compliant unsupervised
clustering models:

| File | Class | Wraps |
|---|---|---|
| `dbscan.py` | `DBSCANModel` | `sklearn.cluster.DBSCAN` |
| `hierarchical.py` | `HierarchicalClusteringModel` | `sklearn.cluster.AgglomerativeClustering` + `scipy.cluster.hierarchy.linkage` |
| `model.py` | N/A | **Validator Bridge:** Imports and exposes both models to satisfy the automated `validate_submission.py` script requirement for multi-algorithm groups. |

Both classes inherit from `models.base_model.BaseModel` and implement
the standard `fit` / `predict` / `predict_proba` / `get_metadata`
contract used across all model groups in this codebase.

**Neither model performs data preprocessing or scaling.** Callers must
supply already-cleaned, already-scaled `float64` NumPy feature
matrices. Both underlying scikit-learn estimators (`DBSCAN`,
`AgglomerativeClustering`) are **deterministic algorithms with no
`random_state` parameter** — there is no source of randomness to seed.
Any synthetic data generated for testing purposes uses
`np.random.RandomState(42)` for reproducibility.

---

## Usage

```python
import numpy as np
from models.group_09_dbscan_hierarchical import (
    DBSCANModel,
    HierarchicalClusteringModel,
)

# X must be a 2D float64 NumPy array, already preprocessed/scaled.
X = np.random.RandomState(42).normal(size=(200, 4)).astype(np.float64)

# --- DBSCAN ---
dbscan = DBSCANModel(eps=0.8, min_samples=5)
dbscan.fit(X)
labels = dbscan.predict(X)          # -1 marks noise/outlier points
print(dbscan.get_metadata())
print(dbscan.get_visualization_data())   # n_clusters, n_noise_points, labels

# --- Hierarchical (Agglomerative) Clustering ---
hier = HierarchicalClusteringModel(n_clusters=3, linkage_method="ward")
hier.fit(X)
labels = hier.predict(X)
print(hier.get_metadata())
viz = hier.get_visualization_data()
linkage_matrix = viz["linkage_matrix"]   # ready for dendrogram rendering
```

### Inference on new (unseen) data

Neither `DBSCAN` nor `AgglomerativeClustering` supports native
inference for points outside the training set, since both are
transductive clustering algorithms exposed only via `fit_predict()`
in scikit-learn. This wrapper package handles two cases:

- **Exact match with training data** (same shape, same values): the
  stored training-time labels are returned directly.
- **New/unseen data**: labels are assigned via a deterministic
  nearest-neighbor rule —
  - `DBSCANModel`: each point is assigned to the cluster of its
    nearest **core sample** if that core sample is within `eps`;
    otherwise the point is labeled `-1` (noise).
  - `HierarchicalClusteringModel`: each point is assigned to the
    cluster label of its single nearest training sample (1-NN rule).

---

## Hyperparameters

### `DBSCANModel`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eps` | float | `0.5` | Maximum neighborhood distance between two samples for one to be considered a neighbor of the other. |
| `min_samples` | int | `5` | Minimum number of samples in a neighborhood for a point to be a core point. |
| `metric` | str | `"euclidean"` | Distance metric used to calculate neighborhoods. |
| `algorithm` | str | `"auto"` | Neighbor-search algorithm: `"auto"`, `"ball_tree"`, `"kd_tree"`, `"brute"`. |
| `leaf_size` | int | `30` | Leaf size passed to `BallTree` / `KDTree`. |
| `n_jobs` | int or `None` | `None` | Number of parallel jobs (CPU only). |

### `HierarchicalClusteringModel`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int or `None` | `2` | Number of clusters to find. Set to `None` when using `distance_threshold`. |
| `linkage_method` | str | `"ward"` | Linkage criterion: `"ward"`, `"complete"`, `"average"`, `"single"`. |
| `metric` | str | `"euclidean"` | Distance metric (must be `"euclidean"` when `linkage_method="ward"`). |
| `distance_threshold` | float or `None` | `None` | Linkage distance above which clusters will not be merged; if set, `n_clusters` must be `None`. |

---

## Testing

Both model classes are covered by `test.py`, which achieves **≥80% line
coverage** (measured at 98% in local verification).

```bash
# From the repository's backend/ directory (matches the
# `from models.base_model import BaseModel` import convention):
cd backend
pytest -v --cov=models.group_09_dbscan_hierarchical \
    models/group_09_dbscan_hierarchical/test.py

# Alternative (unittest runner):
PYTHONPATH=backend python -m unittest \
    models.group_09_dbscan_hierarchical.test -v
```

Test coverage includes:
- `fit()` returns `self`, sets `is_fitted`/`n_features`, and validates
  input (rejects non-`ndarray`, non-2D, and non-`float64` input).
- `predict()` before `fit()` raises `RuntimeError`.
- `predict()` output is a 1D array of shape `(n_samples,)` on both
  training data and new/unseen data.
- `predict()` raises `ValueError` on feature-count mismatch.
- `predict_proba()` returns `None` for both models.
- `get_metadata()` returns exactly the documented keys.
- `get_visualization_data()` returns a JSON-serializable
  `linkage_matrix` (Hierarchical) and noise/cluster counts (DBSCAN).

---

## `linkage_matrix` format

`HierarchicalClusteringModel.get_visualization_data()` returns:

```json
{
  "linkage_matrix": [
    [idx1, idx2, distance, sample_count],
    ...
  ]
}
```

This is the exact output contract of `scipy.cluster.hierarchy.linkage`,
converted to a pure Python `list[list[float]]` via `.tolist()` for JSON
serialization:

- **Shape:** `(n_samples - 1, 4)` — one row per merge step.
- **`idx1`, `idx2`:** indices of the two clusters merged at this step.
  Indices `< n_samples` refer to original data points; indices
  `>= n_samples` refer to clusters formed in earlier merge steps
  (i.e., `n_samples + i` refers to the cluster formed at row `i`).
- **`distance`:** cophenetic distance between the two merged clusters.
- **`sample_count`:** total number of original observations in the
  newly formed cluster.

This format can be passed directly to any standard dendrogram
renderer (e.g., `scipy.cluster.hierarchy.dendrogram`, D3.js, Plotly)
without further transformation.
