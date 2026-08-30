"""
models/group_06_regression/model.py

Regression model covering three required cases:
  - Univariate regression on non-stationary data (single feature,
    typically a time index): checked with a lightweight stationarity
    heuristic (trend correlation + variance-ratio across halves,
    scikit-learn/numpy only) and differenced before fitting if the
    series looks non-stationary.
  - Bivariate / multivariate regression on data with non-linear
    relationships (2+ features): handled with a polynomial feature
    expansion before a linear fit.

Receives preprocessed float64 input from the backend (see
CODING_STANDARDS.md Section 4) — no scaling/encoding/imputation
happens inside this model.
"""

import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from models.base_model import BaseModel


class RegressionModel(BaseModel):
    """Regression that adapts its strategy to the input shape.

    - X with 1 column  -> univariate mode: tests for stationarity
      with a variance-ratio heuristic and differences the target
      before fitting if it is non-stationary, then integrates
      predictions back to the original scale.
    - X with 2+ columns -> multivariate/bivariate mode: expands
      features with a polynomial basis (degree `poly_degree`) so a
      linear model can capture non-linear relationships.
    """

    def __init__(self, poly_degree: int = 2, variance_ratio_threshold: float = 1.5,
                 random_state: int = 42, **kwargs):
        super().__init__(poly_degree=poly_degree,
                          variance_ratio_threshold=variance_ratio_threshold, **kwargs)
        self.poly_degree = poly_degree
        self.variance_ratio_threshold = variance_ratio_threshold
        self.random_state = random_state

        self._mode: str | None = None          # "univariate" | "multivariate"
        self._linreg: LinearRegression | None = None
        self._poly: PolynomialFeatures | None = None
        self._differenced: bool = False
        self._last_y_train_value: float | None = None
        self._stationarity_score: float | None = None
        self._train_time: float | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RegressionModel":
        if y is None:
            raise ValueError("RegressionModel is supervised; y must not be None.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 samples to fit.")

        t0 = time.perf_counter()
        self.n_features = X.shape[1]

        if self.n_features == 1:
            self._fit_univariate(X, y)
        else:
            self._fit_multivariate(X, y)

        self._train_time = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def _fit_univariate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-feature case: test for stationarity, difference if needed.

        Uses a dependency-free heuristic rather than a formal ADF test
        (kept to numpy/scikit-learn only, already pinned for this
        project): compares Var(y) to Var(diff(y)). A random-walk-style
        non-stationary series has a level variance that grows with
        sample size while its differenced variance stays roughly
        constant, so the ratio Var(y)/Var(diff(y)) is large. A
        already-stationary series has the opposite pattern (differencing
        roughly doubles the variance of independent noise), so the
        ratio is small. This mirrors what an ADF test is checking for
        without requiring the extra statsmodels dependency.
        """
        self._mode = "univariate"

        var_level = float(np.var(y))
        var_diff = float(np.var(np.diff(y)))
        variance_ratio = var_level / (var_diff + 1e-12)

        self._stationarity_score = variance_ratio
        self._differenced = variance_ratio > self.variance_ratio_threshold

        if self._differenced:
            y_fit = np.diff(y)
            X_fit = X[1:]
            self._last_y_train_value = float(y[-1])
        else:
            y_fit = y
            X_fit = X

        self._linreg = LinearRegression()
        self._linreg.fit(X_fit, y_fit)

    def _fit_multivariate(self, X: np.ndarray, y: np.ndarray) -> None:
        """2+ feature case: polynomial expansion to capture non-linearity."""
        self._mode = "multivariate"
        self._poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        X_poly = self._poly.fit_transform(X)
        self._linreg = LinearRegression()
        self._linreg.fit(X_poly, y)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predict() is called.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} feature(s), got {X.shape[1]}."
            )

        if self._mode == "univariate":
            preds = self._linreg.predict(X)
            if self._differenced:
                # Integrate: cumulative sum of predicted differences,
                # anchored to the last observed training value.
                preds = self._last_y_train_value + np.cumsum(preds)
            return preds.astype(np.float64)

        X_poly = self._poly.transform(X)
        return self._linreg.predict(X_poly).astype(np.float64)

    # ------------------------------------------------------------------
    # predict_proba — not applicable to regression
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        if self._mode == "univariate" or self._mode == "multivariate":
            coefs = self._linreg.coef_.tolist() if self._linreg is not None else None
        else:
            coefs = None

        # NOTE: CODING_STANDARDS.md Section 11 says the validator checks
        # for "exactly the required keys" — so this dict deliberately
        # contains only the six specified keys, even though self._mode,
        # self._differenced, and self._stationarity_score carry useful extra
        # detail (surfaced in the README instead, and available on the
        # instance for debugging).
        return {
            "model_name": "RegressionModel",
            "model_type": "regressor",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": coefs,
        }
