"""
models/group_06_regression/model.py

Regression model covering three required cases:
  - Univariate regression on non-stationary data (single feature,
    typically a time index): checked with a lightweight stationarity
    heuristic and differenced before fitting if the series looks
    non-stationary.
  - Bivariate / multivariate regression on data with non-linear
    relationships (2+ features): handled with a polynomial feature
    expansion before a linear fit.

Receives preprocessed float64 input from the backend — no scaling,
encoding, or imputation happens inside this model.
"""

import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from models.base_model import BaseModel


class RegressionModel(BaseModel):
    """Regression model that adapts its strategy to the input shape.

    - X with 1 column -> univariate mode.
    - X with 2+ columns -> multivariate mode using polynomial features.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        variance_ratio_threshold: float = 1.5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            poly_degree=poly_degree,
            variance_ratio_threshold=variance_ratio_threshold,
            random_state=random_state,
            **kwargs,
        )

        self.poly_degree = poly_degree
        self.variance_ratio_threshold = variance_ratio_threshold
        self.random_state = random_state

        self._mode: str | None = None
        self._linreg: LinearRegression | None = None
        self._poly: PolynomialFeatures | None = None
        self._differenced: bool = False
        self._last_y_train_value: float | None = None
        self._stationarity_score: float | None = None
        self._train_time: float | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RegressionModel":

        if y is None:
            raise ValueError(
                "RegressionModel is supervised; y must not be None."
            )

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X.shape}."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows, y has {y.shape[0]}."
            )

        if X.shape[0] < 3:
            raise ValueError(
                "Need at least 3 samples to fit."
            )

        t0 = time.perf_counter()

        self.n_features = X.shape[1]

        if self.n_features == 1:
            self._fit_univariate(X, y)
        else:
            self._fit_multivariate(X, y)

        self._train_time = time.perf_counter() - t0
        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Univariate regression
    # ------------------------------------------------------------------

    def _fit_univariate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Single-feature regression with a simple stationarity heuristic."""

        self._mode = "univariate"

        var_level = float(np.var(y))
        var_diff = float(np.var(np.diff(y)))

        variance_ratio = var_level / (var_diff + 1e-12)

        self._stationarity_score = variance_ratio

        self._differenced = (
            variance_ratio > self.variance_ratio_threshold
        )

        if self._differenced:
            y_fit = np.diff(y)
            X_fit = X[1:]

            self._last_y_train_value = float(y[-1])
        else:
            y_fit = y
            X_fit = X

        self._linreg = LinearRegression()
        self._linreg.fit(X_fit, y_fit)

    # ------------------------------------------------------------------
    # Multivariate regression
    # ------------------------------------------------------------------

    def _fit_multivariate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """2+ feature regression using polynomial feature expansion."""

        self._mode = "multivariate"

        self._poly = PolynomialFeatures(
            degree=self.poly_degree,
            include_bias=False,
        )

        X_poly = self._poly.fit_transform(X)

        self._linreg = LinearRegression()
        self._linreg.fit(X_poly, y)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fit before predict() is called."
            )

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X.shape}."
            )

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} feature(s), "
                f"got {X.shape[1]}."
            )

        if self._mode == "univariate":

            preds = self._linreg.predict(X)

            if self._differenced:
                preds = (
                    self._last_y_train_value
                    + np.cumsum(preds)
                )

            return preds.astype(np.float64)

        X_poly = self._poly.transform(X)

        return self._linreg.predict(
            X_poly
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray | None:
        """Probability prediction is not applicable to regression."""

        return None

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:

        coefs = None

        if self._linreg is not None:

            coefficients = np.asarray(
                self._linreg.coef_
            ).ravel()

            if self._mode == "univariate":

                coefs = {
                    "feature_1": float(coefficients[0])
                }

            elif self._mode == "multivariate":

                coefs = {
                    f"feature_{i + 1}": float(value)
                    for i, value in enumerate(coefficients)
                }

        return {
            "model_name": "RegressionModel",
            "model_type": "regressor",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": coefs,
        }