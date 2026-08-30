# Support Vector Machine — Group 13

A `BaseModel`-conformant classifier wrapping `sklearn.svm.SVC`, selectable
between the linear, polynomial, RBF, and sigmoid kernels through a single
constructor hyperparameter.

## 1. Model

**Support Vector Machine (SVM) classifier.** Finds the hyperplane that
separates classes with the widest possible margin, using the kernel trick
to do so in linear, polynomial, RBF, or sigmoid feature space.

## 2. Usage

```python
import numpy as np
from models.group_13_svm import SVMModel

X_train = np.array(...)  # float64, shape (n_samples, n_features)
y_train = np.array(...)  # shape (n_samples,)

model = SVMModel(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

labels = model.predict(X_test)          # shape (n_samples,)
probabilities = model.predict_proba(X_test)  # shape (n_samples, n_classes)
metadata = model.get_metadata()
```

## 3. Hyperparameters

| Name | Default | Controls |
|---|---|---|
| `kernel` | `"rbf"` | Which of `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"` transforms the input before separating classes. |
| `C` | `1.0` | Regularization strength. Lower values allow a wider margin at the cost of more misclassified training points; higher values fit the training data more tightly. |
| `gamma` | `"scale"` | Kernel coefficient for `"rbf"`, `"poly"`, and `"sigmoid"`. Higher values let each training point influence a smaller neighborhood, producing a more tightly-fit decision boundary. Ignored by `"linear"`. |
| `degree` | `3` | Degree of the polynomial kernel. Ignored by all other kernels. |
| `coef0` | `0.0` | Independent term in the `"poly"` and `"sigmoid"` kernels. Ignored by `"linear"` and `"rbf"`. |
| `random_state` | `42` | Seeds the internal probability-calibration routine so identical input reproduces identical output. |

## 4. Mathematics behind SVM

**Hyperplane and margin.** For linearly separable data, an SVM looks for
the hyperplane `w·x + b = 0` that separates the classes while maximizing
the margin — the distance from the hyperplane to the nearest point of
either class. Maximizing this margin (equivalent to minimizing `‖w‖²`
subject to every point being correctly classified) tends to generalize
better than any separating hyperplane chosen arbitrarily.

**Support vectors.** The optimization above depends only on the points
that lie exactly on the margin boundary — the support vectors. Every
other training point could be removed without changing the solution,
which is why the model's complexity scales with the number of support
vectors rather than the full training set size.

**Soft margin (`C`).** Real data is rarely perfectly separable, so the
soft-margin formulation allows some points to violate the margin, at a
penalty controlled by `C`. Small `C` tolerates more violations for a
wider, more generalizable margin; large `C` penalizes violations heavily,
tightening the boundary around the training data (and risking overfit).

**The kernel trick.** Rather than mapping features into a higher
dimensional space explicitly, an SVM only ever needs the *inner product*
between mapped points, `φ(x)·φ(x')`. A kernel function `K(x, x')`
computes that inner product directly, without ever materializing `φ`.
This model supports four:

- **Linear** — `K(x, x') = x·x'`. No mapping at all; the fastest and
  most interpretable choice, appropriate when classes are already close
  to linearly separable in the input space.
- **Polynomial** — `K(x, x') = (γ·x·x' + c₀)^d`. Captures interactions
  between features up to degree `d`; higher `degree` fits more complex,
  curved boundaries at higher computational cost.
- **RBF (Gaussian)** — `K(x, x') = exp(-γ‖x - x'‖²)`. Maps into an
  infinite-dimensional space; the default here because it makes the
  fewest assumptions about the shape of the decision boundary and
  performs well across a broad range of tabular datasets without manual
  feature engineering.
- **Sigmoid** — `K(x, x') = tanh(γ·x·x' + c₀)`. Behaves like a two-layer
  perceptron; included for completeness, though it is not guaranteed to
  satisfy Mercer's condition (positive semi-definiteness) for all `γ`
  and `c₀`, which can make optimization less stable than the other three.

## 5. Advantages and disadvantages

**Advantages**
- Effective in high-dimensional feature spaces, including cases where
  the number of features exceeds the number of samples.
- The margin-maximizing objective tends to generalize well and resist
  overfitting relative to models with no analogous regularization.
- Memory-efficient at inference time: only the support vectors are
  needed, not the full training set.
- The kernel trick lets one algorithm express linear, polynomial, and
  Gaussian-similarity decision boundaries without separate code paths.

**Disadvantages**
- Training scales poorly with dataset size — roughly quadratic to cubic
  in the number of samples — which is why the platform imposes a
  five-minute CPU ceiling (Section 2 of the Coding Standards).
- No native probability output; this implementation calibrates
  probabilities via an auxiliary cross-validated step (Section 8 below),
  which adds training cost and requires a minimum number of samples per
  class.
- Performance is sensitive to `C` and `gamma`; poor choices can
  under- or overfit substantially, and there is no closed-form way to
  pick them without a validation set.
- Provides no native, kernel-independent notion of feature importance
  (see Section 8, `feature_importance`).

## 6. Input contract

`X` passed to `fit` and `predict` is guaranteed by the backend (Section 4
of the Coding Standards) to be a numpy `ndarray`, dtype `float64`, 2D with
shape `(n_samples, n_features)`, already scaled, encoded, and free of
missing values. This model performs no scaling, encoding, or imputation
of its own, and validates these properties defensively rather than
assuming them, so that a defect anywhere upstream surfaces here as a
clear exception instead of a confusing failure inside scikit-learn.

## 7. Output contract

- `predict(X)` → 1D `np.ndarray` of shape `(n_samples,)`, values drawn
  from `self.classes_`, matching the dtype of the training labels.
- `predict_proba(X)` → 2D `np.ndarray` of shape `(n_samples, n_classes)`.
  Rows sum to 1; columns are ordered to match `self.classes_`.

## 8. Metadata contract

`get_metadata()` returns exactly:

```python
{
    "model_name": "Support Vector Machine",
    "model_type": "classifier",
    "hyperparameters": {...},        # every constructor argument used
    "training_time_seconds": float,
    "n_features": int,
    "feature_importance": None,
}
```

`feature_importance` is always `None`. `SVC` has no feature-importance
attribute that is meaningful across every supported kernel: only the
linear kernel exposes a `coef_` vector, and since inputs are already
scaled identically regardless of which kernel is chosen, returning a
value for one kernel and `None` for the other three would make the
metadata contract's shape depend on a hyperparameter choice. Returning
`None` uniformly keeps `get_metadata()`'s return type stable across every
valid configuration of this class.

## 9. Design decisions

**Why SVM.** The assignment fixes this group's algorithm; the interesting
design space is in the kernel, the probability strategy, and the
validation this wrapper adds on top of `SVC`.

**Why RBF as the default kernel.** RBF assumes the least about the shape
of the decision boundary of the four supported kernels and, in practice,
is a robust default across tabular datasets a user might upload without
prior knowledge of their structure. Linear is preferable when the caller
already believes classes are linearly separable (faster, more
interpretable, no `gamma`/`degree` to tune); polynomial suits data with
known low-order feature interactions; sigmoid is included for
completeness and kernel-choice symmetry with the other three, but is the
least commonly the best choice in practice.

**One class, one hyperparameter.** Per Section 6 of the Coding Standards,
kernel choice is a hyperparameter of one `SVMModel` class rather than four
separate classes (`LinearSVMModel`, `PolySVMModel`, ...), since the four
kernels are variants of the same algorithm rather than distinct
algorithms.

**Probability estimation via `CalibratedClassifierCV`, not
`SVC(probability=True)`.** `SVC`'s own `probability=True` flag is
deprecated as of scikit-learn 1.9 — the exact version this project pins —
and is scheduled for removal in 1.11. scikit-learn's own deprecation
notice recommends `CalibratedClassifierCV(SVC(), ensemble=False)` as the
replacement, which is what this model wraps internally. This produces
equivalent Platt-scaled probabilities through the currently-supported
public API instead of a flag already marked for removal.

**Practical consequence:** the calibration step performs 5-fold internal
cross-validation, so `fit` requires **at least 5 training samples in
every class**. This is not a limitation introduced by this design choice
— any Platt-scaling approach to SVM probabilities has an equivalent
requirement — but it is worth stating explicitly. Below that threshold,
`fit` raises `ValueError` with scikit-learn's own message identifying the
under-represented class, consistent with Section 10's "never let a bare
assertion or unhandled error propagate; raise a specific exception."

**Determinism.** `random_state=42` seeds both the underlying `SVC` and
the calibration step. The calibration folds themselves come from an
unshuffled `StratifiedKFold`, which is already deterministic given the
input order, so the only source of randomness is the seeded estimator.
Identical `(X, y)` and the same `random_state` reproduce identical
`predict` and `predict_proba` output on every run (verified in
`test.py`).

## 10. Complexity analysis

- **Training:** roughly `O(n_samples² · n_features)` to
  `O(n_samples³ · n_features)` depending on how separable the data is,
  since the underlying quadratic program scales with the number of
  support vectors, which itself scales with the sample count in the
  worst case. The 5-fold calibration step multiplies this by a small
  constant factor (it refits the base estimator across folds plus once
  on the full training set).
- **Prediction:** `O(n_support_vectors · n_features)` per sample —
  independent of the original training set size once support vectors are
  fixed.

## 11. Repository structure

```
backend/models/group_13_svm/
├── __init__.py       # re-exports SVMModel
├── model.py          # SVMModel implementation
├── test.py           # unit tests (unittest, run via pytest)
├── README.md          # this file
└── requirements.txt   # pinned dependencies
```

## 12. Installation

From the repository root, with `backend/` on `PYTHONPATH` (so that
`models.base_model` resolves):

```bash
pip install -r backend/models/group_13_svm/requirements.txt
```

## 13. Running the tests

From inside this folder:

```bash
python -m pytest test.py --cov=. --cov-report=term-missing
```

Verified locally: **48 tests, 100% statement coverage on `model.py`**,
`ruff check .` reports no issues, no runtime warnings.

## 14. References

- Cortes, C. and Vapnik, V. (1995). Support-Vector Networks. *Machine
  Learning*, 20(3).
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and
  Comparisons to Regularized Likelihood Methods.
- scikit-learn documentation: `sklearn.svm.SVC`,
  `sklearn.calibration.CalibratedClassifierCV`.
