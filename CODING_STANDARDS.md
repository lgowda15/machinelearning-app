# Coding Standards — ML Integration Project

**Course:** Machine Learning (UM25MB653CA2) · Trimester 3, 2025–26
**Version:** 1.0
**Project start:** July 14 · **Final submission deadline:** August 15

---

## 1. Purpose

Your group builds one model. The integration team combines all twelve into a single web application where a user uploads a dataset, picks a model, trains it, and views results.

For twelve independently-written models to run inside one application, each must present the same interface to the rest of the system: the same method names, the same input shape, and the same output format, regardless of the algorithm inside. This document defines that interface. Section 11 gives a validator that checks your code against it — passing the validator means your model integrates.

---

## 2. Environment (fixed)

| Component | Version |
|---|---|
| Python | 3.12.x |
| scikit-learn | 1.9.0 |
| Deep learning framework | PyTorch (TensorFlow/Keras not permitted) |
| Target hardware | CPU only |
| Maximum training time | 5 minutes on CPU |

Additional libraries are permitted provided they are open source and pinned to an exact version in your `requirements.txt` (for example `xgboost==2.0.3`, `hmmlearn==0.3.0`). The integration team reviews every added dependency before merge.

---

## 3. The interface: subclass `BaseModel`

Every model subclasses `BaseModel` and implements its four methods. The class lives at `models/base_model.py` in the repository — import it, do not redefine it.

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseModel(ABC):
    """Common interface for all models in the integration platform."""

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.is_fitted = False
        self.n_features = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseModel":
        """Train the model and return self.

        X: 2D float64 array, shape (n_samples, n_features), already
           preprocessed by the backend (Section 4).
        y: 1D array, shape (n_samples,); None for clustering and
           dimensionality-reduction models.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of shape (n_samples,). See Section 6."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Classifiers: 2D array (n_samples, n_classes), rows summing
        to 1, columns ordered to match self.classes_.
        All other model types: return None."""
        ...

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Return the metadata dict specified in Section 7."""
        ...
```

Non-classifiers still implement `predict_proba`; it returns `None`. Omitting the method fails validation.

### Model-specific visualisations (optional fifth method)

Some models produce visuals the generic metrics cannot express: SHAP values, dendrogram linkage, explained variance, tree structure. If yours does, implement one additional method. It is optional — models without special visuals simply omit it.

```python
def get_visualization_data(self) -> Optional[Dict]:
    """Return JSON-serialisable data for model-specific visuals, or None.

    Return raw numbers and arrays, never figure objects or images.
    The frontend renders the chart; your model supplies the data.
    """
```

What to return, by group:

| Group | Key | Value |
|---|---|---|
| 2 (SHAP) | `shap_values` | per-feature SHAP values for the test set, as nested lists |
| 9 (Hierarchical) | `linkage_matrix` | the scipy linkage matrix, as nested lists |
| 15 (PCA) | `explained_variance_ratio` | list of floats, one per component |
| 1 (Decision Trees) | `tree_structure` | nodes and edges as a nested dict |

Every value must survive `json.dumps` — lists, dicts, numbers, strings only. Convert numpy arrays with `.tolist()`. Do **not** return matplotlib figures, base64 images, or file paths; rendering is the frontend's job, and shipping images would bloat responses and lock the UI to your styling. Document the exact structure you return in your README so the integration team can build the matching chart.


---

## 4. Input contract

When your `fit` and `predict` methods run, `X` is guaranteed to be:

- a numpy `ndarray` (never a pandas DataFrame),
- dtype `float64`,
- 2D, shape `(n_samples, n_features)`,
- already scaled (StandardScaler, fit on the training split only),
- already numeric (categorical columns encoded by the backend),
- free of missing values (imputation already applied).

**The backend owns all preprocessing. Your model performs none.** Do not scale, encode, or impute inside your model.

If you scale data the backend has already scaled, it is double-scaled and the model underperforms. And the comparison across models is only fair if every model receives identical input — otherwise a difference in results reflects preprocessing rather than the algorithm.

**Sequence and image models (Groups 3 and 7)** receive the same 2D `(n_samples, n_features)` input. Any reshaping into `(samples, timesteps, features)` or `(samples, channels, height, width)` happens inside your `fit`/`predict`. Document the column order you expect in your README so the backend supplies features consistently. This is the only case where a model reshapes input; it still never scales or encodes.

---

## 5. Worked example

A complete model that satisfies every rule in this document. Logistic Regression is used here because it is not one of the twelve allotted topics — adapt the pattern to your own algorithm rather than copying it.

```python
# models/example_logistic_regression/model.py
import time
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier.

    Wraps sklearn LogisticRegression. Receives preprocessed float64
    input from the backend (Section 4).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 random_state: int = 42, **kwargs):
        super().__init__(C=C, max_iter=max_iter, **kwargs)
        self.random_state = random_state
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state,
        )
        self.classes_ = None
        self._train_time = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LogisticRegressionModel":
        if y is None:
            raise ValueError("LogisticRegressionModel is supervised; y must not be None.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")

        t0 = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, "
                f"got {X.shape[1]}."
            )
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._model.predict_proba(X)

    def get_metadata(self) -> Dict:
        return {
            "model_name": "Logistic Regression",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": dict(zip(
                [f"feature_{i}" for i in range(self.n_features)],
                np.abs(self._model.coef_[0]).tolist(),
            )),
        }
```

The example validates its inputs (the `y is None` check, the dimension checks, the feature-count check on predict), records training time, and sets `classes_`. Follow the same pattern.

A full runnable copy lives in the repository at `backend/models/example_logistic_regression/` with matching tests — use it as a reference implementation.

---

## 6. Groups with more than one algorithm

Several groups own multiple algorithms (KNN, K-Means and GMM in one group; ID3, CART and CHAID in another). The rule: **one class per algorithm, one file per class**, all inside your one folder. Each class subclasses `BaseModel` independently and appears as its own entry in the platform, so a user can pick "K-Means" or "GMM" separately.

```
models/group_05_knn_kmeans_gmm/
├── __init__.py          # re-exports all classes
├── knn.py               # KNNModel
├── kmeans.py            # KMeansModel
├── gmm.py               # GMMModel
├── test_knn.py
├── test_kmeans.py
├── test_gmm.py
├── requirements.txt
└── README.md            # documents all three
```

`__init__.py` re-exports the classes so the platform imports from the package, not from your file layout:

```python
from .knn import KNNModel
from .kmeans import KMeansModel
from .gmm import GMMModel
```

Splitting one file per algorithm also means members of your group can work in parallel without editing the same file.

Variants of one algorithm that differ only by a hyperparameter (linkage criteria in hierarchical clustering, kernel choice in SVM) are **one class with a hyperparameter**, not separate classes. Distinct algorithms (K-Means vs GMM) are separate classes.

Groups with a single algorithm keep the single `model.py` layout.

---

## 7. Output contract

`predict` returns a 1D numpy array of shape `(n_samples,)`. Meaning by model type:

| Model type | `predict` returns | Example |
|---|---|---|
| Classifier | class labels, same dtype as training `y` | `[0, 1, 0, 2]` |
| Clusterer | integer cluster ids; noise labelled `-1` | `[0, 1, -1, 0]` |
| Regressor | continuous floats | `[3.4, 1.1, 9.8]` |
| Dimensionality reducer | reduced matrix — see note | — |

Watch the return shape:

```python
# Correct — classifier returns 1D labels
return np.array([0, 1, 0])
# Incorrect — do not one-hot encode predictions
return np.array([[1, 0], [0, 1], [1, 0]])
```

```python
# Correct — probabilities sum to 1 per row, columns match self.classes_
return np.array([[0.8, 0.2], [0.3, 0.7]])
# Incorrect — log-probabilities or unnormalised scores
return np.array([[-0.22, -1.61], [-1.20, -0.36]])
```

```python
# Correct — DBSCAN noise points retain the -1 label
return np.array([0, 0, -1, 1, -1])
# Incorrect — noise relabelled, information lost
return np.array([0, 0, 1, 1])
```

**Dimensionality reduction (Group 15, PCA)** is the exception: `predict` returns the transformed matrix of shape `(n_samples, n_components)`, and the metadata carries `explained_variance_ratio`. Document this in your README so the backend renders a variance plot rather than a prediction table.

---

## 8. Metadata contract

`get_metadata` returns a dict with exactly these keys:

```python
{
    "model_name": str,               # e.g. "Random Forest"
    "model_type": str,               # "classifier" | "clusterer" |
                                     #   "regressor" | "dimensionality_reducer"
    "hyperparameters": dict,         # every hyperparameter used
    "training_time_seconds": float,  # or None
    "n_features": int,               # features seen at fit time
    "feature_importance": dict | None,  # {feature_name: score} or None
}
```

`model_type` determines which metrics the backend computes and which chart it renders, so it must be one of the four listed values and must match what your model actually does.

Constructor arguments are supplied from the UI as JSON, so every hyperparameter must be a JSON-friendly type: number, string, boolean, or a list/dict of these. No callables, no numpy types, no custom objects as constructor arguments.


---

## 9. README.md

Every submission includes a `README.md` containing:

1. **Model** — name and a one-sentence description.
2. **Usage** — a short snippet: construct, `fit`, `predict`.
3. **Hyperparameters** — a table of name, default, and what each controls.
4. **Feature layout** (sequence and image models only) — the column order `fit` expects and how the model reshapes internally.
5. **Running the tests** — the exact command.
6. **Design decisions** — why this algorithm, why these hyperparameter choices, and how behaviour changes if key choices are altered.

The README serves the integration and testing teams.

---

## 10. Correctness rules

**No data leakage.** The backend scales using training-split statistics before your model sees the data, so the rule for you is simple: never fit or transform on the test set, and never recombine train and test inside your model. Leakage inflates reported performance.

**Determinism.** Identical input and identical seed must produce identical output on every run and every machine. Set `random_state=42` on every stochastic component. PyTorch groups (3, 4, 7) additionally set once, at import:

```python
import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
```

If a testing team gets different numbers than you did, that is a defect.

**Explicit failure.** When input genuinely does not suit your model, raise a specific exception. Never return `None` or an empty array to signal failure, and never let a bare assertion or unhandled error propagate.

```python
# Correct
raise ValueError("HMM requires sequential data; received "
                 "single-timestep tabular input.")

# Incorrect
return None
assert X.shape[1] == 4
```

The backend catches the exception and shows the user a clean message. An unhandled error surfaces as a server error instead.

---

## 11. Self-validation

Run the validator against your folder before opening a pull request:

```bash
python validate_submission.py models/group_05_knn_kmeans_gmm/
```

It checks that the model imports and subclasses `BaseModel`; that all four methods are implemented; that `fit` accepts `(X, y)`, returns self, and sets `is_fitted`; that `predict` returns a 1D array of length `n_samples`; that `predict_proba` returns correctly-shaped normalised probabilities or `None`; that `get_metadata` returns exactly the required keys with a valid `model_type`; that a fit-then-predict cycle is reproducible under a fixed seed; that training completes within five minutes on CPU; that unit tests pass with at least 80% coverage; that `ruff` reports no issues; and that the code contains no hardcoded absolute paths and no `print` calls.


The validator tests classifiers, clusterers and regressors against generic tabular fixtures. Sequence models (Groups 3 and 8) and image models (Group 7) are tested against per-group fixture datasets maintained by the integration team, since a correct sequence model should reject non-sequential input. If your model needs a specific input layout, say so in your README and the fixture will match it.

Do not open a pull request while the validator reports failures.

---

## 12. Unit tests (minimum 80% coverage)

Tooling: **pytest** with **pytest-cov**. Minimum meaningful tests, adapted to your model type:

```python
import unittest
import numpy as np
from model import LogisticRegressionModel


class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 4))
        self.y = rng.integers(0, 2, 100)
        self.X_test = rng.standard_normal((20, 4))

    def test_fit_returns_self(self):
        m = LogisticRegressionModel()
        self.assertIs(m.fit(self.X, self.y), m)

    def test_predict_shape(self):
        m = LogisticRegressionModel().fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (20,))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LogisticRegressionModel().predict(self.X_test)

    def test_proba_rows_sum_to_one(self):
        m = LogisticRegressionModel().fit(self.X, self.y)
        p = m.predict_proba(self.X_test)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_determinism(self):
        a = LogisticRegressionModel().fit(self.X, self.y).predict(self.X_test)
        b = LogisticRegressionModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_metadata_keys(self):
        md = LogisticRegressionModel().fit(self.X, self.y).get_metadata()
        for k in ["model_name", "model_type", "hyperparameters",
                  "training_time_seconds", "n_features", "feature_importance"]:
            self.assertIn(k, md)


if __name__ == "__main__":
    unittest.main()
```

Run: `python -m pytest test.py --cov=. --cov-report=term-missing`

---

## 13. Repository layout

The repository follows a standard main/test separation. Your group touches only your own folder under `backend/models/`.

```
ml-integration/
├── backend/
│   ├── models/
│   │   ├── base_model.py                  # provided — import, do not edit
│   │   ├── example_logistic_regression/   # reference implementation
│   │   ├── group_01_decision_trees/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── test.py
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── ...one folder per group...
│   ├── tests/                             # integration-team tests — do not edit
│   └── validate_submission.py
├── frontend/                              # integration team only
└── .github/workflows/ci.yml               # do not edit
```

Naming: files `snake_case`; model classes `PascalCase` ending in `Model` (e.g. `KMeansModel`); constants `UPPER_CASE`; the folder `group_<NN>_<shortname>`.

Files you never edit: `base_model.py`, the model registry (`backend/app/core/registry.py`), anything under `backend/tests/`, anything under `frontend/`, and the CI workflow. Pull requests touching files outside your own folder are rejected.

Registration is not your job: after your pull request merges, the integration team adds your classes to the registry. Because every group's pull request touches only its own folder, pull requests from different groups never conflict — even when several are opened the same day.

---

## 14. Submission — pull request process

1. Branch from `main`: `git checkout -b feature/group_<NN>_<model>`.
2. Add your folder under `models/group_<NN>_<model>/`.
3. Push and open a pull request against `main`.
4. Continuous integration runs automatically (ruff, pytest with coverage, the validator, and an integration smoke test). A pull request with failing checks is not reviewed.
5. Two approvals are required to merge, one being the integration lead.
6. Push fixes to the same branch; the pull request updates in place.

Note on class imbalance: the backend detects it (any class below 20% or above 80% of samples) and displays an informational note in the EDA view. You neither implement this nor rebalance your data — build your model on the data as provided.

---

## 15. Deadlines

| Submit by | Groups | Models |
|---|---|---|
| July 22 | 1, 11, 13 | Decision Trees · LDA & QDA · SVM |
| July 29 | 6, 15 | Regression · PCA |
| August 5 | 5, 8, 9 | KNN/K-Means/GMM · HMM/Naïve Bayes · DBSCAN/Hierarchical |
| August 12 | 2, 4, 7, 3 | Random Forest + XGBoost · ANN · CNN · RNN/LSTM/GRU |

Absolute deadline for all submissions: **August 15**. Models depending only on scikit-learn submit first so integration can begin early. Deep-learning groups (3, 4, 7) get the additional time.

---

## 16. Common questions

**Do I fork the repository again every week?** No. Fork once. Each week, sync your fork with the original repository (GitHub shows a "Sync fork" button), then branch and work.

**Can I develop in a Jupyter notebook or Colab?** Develop wherever you like, but submit plain `.py` files. Notebook files (`.ipynb`) are rejected.

**Where do I get data to test my model?** Generate synthetic numpy arrays in your tests (the reference implementation shows how) or use a UCI dataset locally. Your model must work on any numeric array of the right shape, not on one specific dataset.

**Do I commit my dataset to the repository?** No. No data files in pull requests. Tests generate their own synthetic data.

**Does my model need to handle missing values or text columns?** No. The backend cleans, encodes and scales before your model sees anything. Assume clean numeric float64 input, always.

**My model's accuracy is low on my test data. Is that a problem?** The requirement is a correct, conforming, deterministic model — not a benchmark score. Performance depends on whatever data a user uploads.

**Can I use TensorFlow or Keras?** No. PyTorch for deep learning, scikit-learn for the rest.

**My model is slow. Can I assume a GPU?** No. CPU only, five-minute training ceiling. Reduce epochs, network size, or default iterations until it fits.

**Do I build the chart for my SHAP values / dendrogram / variance plot?** No. Return the numbers from `get_visualization_data()`; the frontend renders them. Never return images or figure objects.

**Can I import code from another group's folder?** No. Your folder is self-contained apart from `base_model.py`.

**The validator passes locally but CI fails.** Almost always an environment mismatch: confirm Python 3.12, install exactly what your `requirements.txt` pins, and rerun in a fresh virtual environment.

**I left `print()` calls in for debugging.** Remove them before the pull request; the validator rejects them. Same for hardcoded absolute paths.

**Two of us pushed to the same branch and it's a mess.** Both of you work in the same fork: pull before you push, and split algorithms across separate files (Section 6) so you rarely touch the same file.

**I accidentally edited `base_model.py` or another shared file.** Revert that file before opening the pull request; changed shared files fail review automatically.

---

## 17. Quick reference

- Subclass `BaseModel`; implement `fit`, `predict`, `predict_proba`, `get_metadata`. One class per algorithm if your group owns several. Add `get_visualization_data` only if your model has special visuals.
- Input `X` arrives preprocessed: 2D float64, scaled, encoded, no missing values. Your model does not preprocess.
- `predict` returns a 1D array of length `n_samples` (labels, cluster ids with `-1` for noise, or floats).
- `predict_proba` returns normalised `(n_samples, n_classes)` probabilities, or `None`.
- `get_metadata` returns exactly the specified keys with a valid `model_type`.
- Seed with 42; guarantee determinism; raise specific exceptions on bad input.
- No preprocessing, no leakage, no `print`, no hardcoded paths.
- Pass `validate_submission.py`, then open a pull request for two-approval review.
