# Group 02 — Random Forest + XGBoost

## Model

This submission contains two supervised classification algorithms:

1. `RandomForestModel`
2. `XGBoostModel`

Both implement the integration platform's `BaseModel` interface.

The backend supplies a 2D `numpy.ndarray` with `float64` values. Preprocessing is owned by the backend, so these models do not scale, encode, or impute features.

## Usage

```python
import numpy as np
from models.group_02_random_forest_xgboost import (
    RandomForestModel,
    XGBoostModel,
)

X = np.random.default_rng(42).standard_normal((100, 5)).astype(np.float64)
y = np.random.default_rng(42).integers(0, 2, size=100)

model = RandomForestModel()
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)
metadata = model.get_metadata()
```

The same interface is available for `XGBoostModel`.

## Hyperparameters

| Parameter | Random Forest default | XGBoost default | Purpose |
|---|---:|---:|---|
| n_estimators | 200 | 200 | Number of trees |
| max_depth | None | 6 | Maximum tree depth |
| learning_rate | — | 0.1 | XGBoost boosting step size |
| subsample | — | 1.0 | Fraction of rows per boosting round |
| colsample_bytree | sqrt | 1.0 | Feature sampling |
| min_samples_split | 2 | — | Minimum samples needed to split an RF node |
| min_samples_leaf | 1 | — | Minimum samples in an RF leaf |
| min_child_weight | — | 1.0 | Minimum child weight in XGBoost |
| reg_alpha | — | 0.0 | L1 regularization |
| reg_lambda | — | 1.0 | L2 regularization |
| random_state | 42 | 42 | Deterministic training |
| n_jobs | -1 | -1 | CPU parallelism |

## Input contract

`fit` and prediction methods expect:

- `numpy.ndarray`
- `float64`
- two dimensions: `(n_samples, n_features)`
- finite values
- no missing values
- features already encoded and scaled by the backend

The model does not perform preprocessing.

## Output contract

`predict(X)` returns a one-dimensional array of class labels with length `n_samples`.

`predict_proba(X)` returns a two-dimensional array with shape:

```text
(n_samples, n_classes)
```

Rows are normalized probabilities and columns follow `classes_`.

## SHAP visualization

`get_visualization_data()` returns JSON-serializable SHAP values:

```python
{
    "shap_values": nested_lists
}
```

The model does not create or return a matplotlib figure, image, base64 string, or file path. The frontend is responsible for rendering the visualization.

The integration interface does not pass a separate test matrix to `get_visualization_data()`. Therefore this implementation stores the matrix supplied to `fit()` and computes SHAP values for that stored matrix. If the integration backend later exposes a test-set argument for visualization, the method can be adapted without changing the core model interface.

## Design decisions

### Random Forest

Random Forest is used as a bagging-based tree ensemble. It is robust for general tabular data and does not require assumptions about feature distributions.

### XGBoost

XGBoost is used as a boosting-based tree ensemble. It builds trees sequentially and can capture nonlinear relationships and feature interactions.

### Determinism

Both implementations use `random_state=42`.

### Feature names

The backend supplies arrays rather than pandas DataFrames. Therefore metadata uses generic names:

```text
feature_0
feature_1
...
feature_n
```

The integration frontend can map these names to the original backend feature names if available.

## Testing

Run:

```bash
python -m pytest test.py --cov=. --cov-report=term-missing
```

The tests cover:

- fitting
- prediction shape
- probability normalization
- prediction before fitting
- feature-count validation
- metadata keys
- deterministic output
- SHAP visualization structure
- invalid input handling

## Environment

Target environment:

- Python 3.12.x
- scikit-learn 1.9.0
- CPU only

Additional dependencies are pinned in `requirements.txt`.

## Important integration note

The supplied integration `BaseModel` is imported from:

```python
from models.base_model import BaseModel
```

Do not copy or modify `base_model.py`.
Project update for GitHub workflow demonstration.