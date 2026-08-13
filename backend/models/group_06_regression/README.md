# Regression (uni/bi/multivariate)

## Model

Adaptive regression that handles non-linearity in bivariate/multivariate data via polynomial feature expansion, and non-stationarity in univariate data via automatic differencing.

## Usage

```python
from models.group_06_regression.model import RegressionModel

model = RegressionModel()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Hyperparameters

| Name | Default | Controls |
|---|---|---|
| `poly_degree` | `2` | Degree of the polynomial feature expansion used in multivariate mode (2+ input features) to capture non-linear relationships. |
| `variance_ratio_threshold` | `1.5` | Threshold on Var(y) / Var(diff(y)) in univariate mode (1 input feature); above this, the series is treated as non-stationary and differenced before fitting. |
| `random_state` | `42` | Seed stored for reproducibility; the underlying `LinearRegression` fit is itself deterministic. |

## Running the tests

```
python -m pytest test.py --cov=. --cov-report=term-missing
```

## Design decisions

**Why one class instead of three.** The brief asks for bivariate, multivariate, and univariate regression in one submission. Rather than three separate classes, `RegressionModel` branches internally on `X.shape[1]`: one feature routes to the univariate/non-stationary path, two or more route to the multivariate/non-linear path. This keeps a single, simple entry point for the integration backend while still covering all three cases named in the brief.

**Why polynomial expansion for non-linearity.** `PolynomialFeatures` + `LinearRegression` is the standard, cheap way to let a linear model fit curved relationships and interaction terms, without needing a heavier model class, extra dependencies, or long training time (well within the 5-minute CPU budget). Degree 2 is the default because it captures the most common non-linear patterns (quadratic curvature, pairwise interactions) without overfitting on small datasets; users can raise `poly_degree` for more flexibility.

**Why a variance-ratio heuristic instead of a formal ADF test.** An Augmented Dickey-Fuller test (from `statsmodels`) is the textbook way to detect non-stationarity, but it's an extra dependency that needs pinning and review, and in testing it conflicted with the pinned `pandas`/`numpy` versions in this environment. Instead, the model compares `Var(y)` to `Var(diff(y))`: a non-stationary (e.g. random-walk-like) series has a level variance that grows with sample size while its differenced variance stays roughly constant, so the ratio is large; an already-stationary series shows the opposite pattern. This captures the same underlying signal an ADF test targets, using only `numpy`, which is already a project dependency.

**How behaviour changes if key choices are altered.** Raising `poly_degree` increases flexibility on curved multivariate data but raises overfitting risk on small samples. Lowering `variance_ratio_threshold` makes the model more eager to difference univariate series (more sensitive to mild non-stationarity); raising it makes differencing rarer (only very strong non-stationarity triggers it).

**Limitation to flag in the report.** Because the univariate branch anchors predictions to the last training value when the series is differenced, `predict()` gives a genuine forecast only for X immediately following the training range — it isn't designed for arbitrary out-of-sample X inputs unrelated to the training series' time ordering.
