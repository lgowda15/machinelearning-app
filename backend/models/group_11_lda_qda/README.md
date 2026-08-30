# Linear and Quadratic Discriminant Analysis

> This folder is the complete, self-contained deliverable for the course pull
> request. Nothing outside `backend/models/group_11_lda_qda/` (including this
> file's parent directories' own `README.md`/`pyproject.toml` in the standalone
> demo copy) should be part of the PR — see that top-level README's submission
> checklist.

## 1. Models

- **Linear Discriminant Analysis (LDA):** a classifier that estimates one shared
  covariance structure and creates linear decision boundaries.
- **Quadratic Discriminant Analysis (QDA):** a classifier that estimates a separate
  covariance structure for every class and permits quadratic decision boundaries.

`LDAModel` and `QDAModel` independently subclass the integration platform's
`BaseModel`. Both implement `fit`, `predict`, `predict_proba`, and `get_metadata`.

## 2. Input contract and preprocessing boundary

Both model classes accept the platform contract exactly:

- `X` is a two-dimensional `numpy.ndarray` with `dtype=float64`.
- `X` is already numeric, scaled, and free of missing or infinite values.
- `y` is a one-dimensional numpy array containing at least two classes.

The model classes validate this contract but do not transform the data. They do not
scale, encode, impute, select features, remove rows, or rebalance classes. All such
preprocessing remains the backend's responsibility and must be learned from training
data only.

## 3. Usage

Run from the repository root with `backend` on the Python path:

```python
import numpy as np

from models.group_11_lda_qda import (
    LDAModel,
    QDAModel,
    analyze_suitability,
    compare_lda_qda,
)

# X_clean is produced by the backend preprocessing pipeline.
X_clean = np.array(
    [
        [-1.2, 0.4],
        [-0.9, 0.1],
        [0.8, -0.2],
        [1.1, -0.4],
        [-1.0, 0.3],
        [0.9, -0.3],
    ],
    dtype=np.float64,
)
y = np.array(["No", "No", "Yes", "Yes", "No", "Yes"])

lda = LDAModel().fit(X_clean, y)
qda = QDAModel(reg_param=0.1).fit(X_clean, y)

lda_labels = lda.predict(X_clean)
qda_probabilities = qda.predict_proba(X_clean)

suitability = analyze_suitability(
    X_clean,
    y,
    cv_splits=3,
    qda_reg_param=0.1,
)

comparison = compare_lda_qda(
    X_clean,
    y,
    n_splits=3,
    qda_params={"reg_param": 0.1},
)
```

The small example uses QDA regularization because each class has very few samples.
For research results, use a substantially larger dataset.

## 4. Target-label handling

LDA and QDA do not require binary targets to be converted to `0` and `1`.
Scikit-learn discovers the classes internally, while these wrappers retain the original
numpy dtype. For example:

- training labels `"No"` and `"Yes"` produce string predictions;
- training labels stored as `int16` produce `int16` predictions;
- multiclass labels such as `"Low"`, `"Medium"`, and `"High"` remain unchanged.

`classes_` records probability-column order. If it is
`["No", "Yes"]`, column 0 of `predict_proba` belongs to `"No"` and column 1 belongs
to `"Yes"`.

The suitability layer automatically treats string, boolean, and low-cardinality numeric
targets as classification. A number alone cannot reveal its business meaning, so a
high-cardinality numeric target is conservatively reported as regression-like. After a
user confirms that such values are category codes, pass `target_kind="classification"`.
This confirmation changes only the interpretation; it does not encode or mutate `y`.

If the wider backend deliberately uses a label encoder, it must own both encoding and
inverse transformation outside these model classes. Direct use of original labels is the
simpler default.

## 5. Hyperparameters

### LDAModel

| Name | Default | Purpose |
|---|---:|---|
| `solver` | `"svd"` | LDA solution method: `svd`, `lsqr`, or `eigen`. |
| `shrinkage` | `None` | Covariance shrinkage for `lsqr` or `eigen`; use `"auto"` or a value from 0 to 1. |
| `priors` | `None` | Optional class prior probabilities in class order. |
| `n_components` | `None` | Number of discriminant components retained by applicable operations. |
| `store_covariance` | `False` | Whether the estimator stores the covariance matrix. |
| `tol` | `1e-4` | Rank-estimation tolerance used by the SVD solver. |

### QDAModel

| Name | Default | Purpose |
|---|---:|---|
| `priors` | `None` | Optional class prior probabilities in class order. |
| `reg_param` | `0.0` | Regularizes class covariance estimates from 0 (none) to 1 (full). |
| `store_covariance` | `False` | Whether the estimator stores per-class covariance matrices. |
| `tol` | `1e-4` | Threshold used to warn about rank-deficient covariance estimates. |

## 6. Suitability analysis

`analyze_suitability` is deliberately outside both model classes. It reports:

- automatic binary, multiclass, or regression-like target detection;
- class counts and the platform's below-20% or above-80% imbalance flag;
- constant features and maximum absolute feature correlation;
- pooled and per-class covariance ranks and condition numbers;
- a descriptive class-covariance difference score;
- whether each model is suitable and whether both can be compared safely;
- whether the covariance structure favors LDA's simplicity or QDA's flexibility.

The covariance-difference score is a diagnostic heuristic, not a formal hypothesis test.
The final recommendation comes from held-out predictive performance.

Unregularized QDA is marked unsuitable when a cross-validation training fold would not
have more observations per class than features, or when a class covariance matrix is
rank-deficient. A positive `reg_param` permits regularized estimation but leaves a clear
warning in the report.

Class imbalance is reported only. No SMOTE, over-sampling, or under-sampling occurs.

## 7. Model comparison

`compare_lda_qda` uses shuffled `StratifiedKFold` with `random_state=42`. Both models
receive identical folds. Its report contains:

- fold means and standard deviations for accuracy, weighted precision, weighted recall,
  weighted F1, and ROC-AUC;
- an aggregate out-of-fold confusion matrix;
- out-of-fold predictions using original labels;
- a recommendation ranked by mean weighted F1, then ROC-AUC, then accuracy.

Binary ROC-AUC uses the second value in sorted `classes_` as the positive class.
Multiclass ROC-AUC uses weighted one-vs-rest averaging. If every ranking metric ties,
LDA is selected because it is the simpler model.

The comparison refuses to run by default unless the suitability layer approves both
models. Regularize QDA when justified, correct the upstream data issue, or set
`require_suitable=False` only for an explicitly documented diagnostic experiment.

## 8. Metadata

Both classes return exactly these keys:

```text
model_name
model_type
hyperparameters
training_time_seconds
n_features
feature_importance
```

LDA reports mean absolute discriminant coefficients as feature-importance scores. QDA
returns `None` because it has no single global linear coefficient per feature.

## 9. Running the tests

From the repository root:

```bash
python -m pytest
python -m ruff check .
```

The pytest configuration enforces branch coverage of at least 80%. Tests cover the
shared interface, errors, shape and probability contracts, metadata, label dtype,
determinism, target detection, covariance/sample-size checks, binary and multiclass
metrics, JSON serialization of reports, and that every constructor hyperparameter
(`priors`, `store_covariance`, `n_components`, `tol`) has an observable effect, not
just a passing validation check.

Before opening the pull request, also run the integration team's own validator,
which this local test suite does not replace:

```bash
python validate_submission.py models/group_11_lda_qda/
```

## 10. Design decisions

- **Separate classes:** LDA and QDA are distinct algorithms and platform entries, so
  each has its own class and file.
- **SVD as the LDA default:** it is deterministic, CPU-friendly, and tolerates
  high-dimensional inputs better than explicit covariance inversion. `lsqr` or `eigen`
  enables shrinkage when that assumption is appropriate.
- **Unregularized QDA as the default:** this preserves textbook QDA behavior. The
  suitability layer blocks weak covariance estimation; `reg_param` is an explicit,
  documented choice rather than a hidden correction.
- **External analysis:** suitability inspection and cross-validation do not contaminate
  the platform model interface or duplicate backend preprocessing.
- **Original labels:** no unnecessary target conversion is introduced, and prediction
  dtype is explicitly restored to the training target's dtype.
- **Determinism:** the estimators are deterministic; the only randomized component,
  fold shuffling, uses the fixed seed 42.
- **CPU budget:** the defaults use scikit-learn's CPU implementations and do not perform
  expensive hyperparameter searches. The test suite explicitly checks the five-minute
  training limit on representative data.

In the official integration repository, run its validator as an additional final check:

```bash
python validate_submission.py models/group_11_lda_qda/
```
