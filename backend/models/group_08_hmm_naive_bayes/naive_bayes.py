import time

import numpy as np
from sklearn.naive_bayes import GaussianNB

from models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """Gaussian Naive Bayes classifier.

    Wraps sklearn GaussianNB. Receives preprocessed float64 input from the
    backend (CODING_STANDARDS.md Section 4): scaled, encoded, no missing
    values. Gaussian variant specifically -- MultinomialNB and BernoulliNB
    assume non-negative counts / binary features respectively, and would
    reject or silently misinterpret StandardScaler output, which is
    zero-centred and can be negative.
    """

    def __init__(self, var_smoothing: float = 1e-9, random_state: int = 42, **kwargs):
        super().__init__(var_smoothing=var_smoothing, **kwargs)
        if var_smoothing <= 0:
            raise ValueError("var_smoothing must be > 0.")
        self.var_smoothing = var_smoothing
        # GaussianNB fits by closed-form per-class mean/variance -- there is
        # no stochastic component to seed. random_state is accepted and
        # stored only so this model's constructor matches the rest of the
        # platform's hyperparameter form; it is never passed to GaussianNB.
        self.random_state = random_state
        self._model = GaussianNB(var_smoothing=var_smoothing)
        self.classes_ = None
        self._train_time = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NaiveBayesModel":
        if y is None:
            raise ValueError("NaiveBayesModel is supervised; y must not be None.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or inf).")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")
        if len(np.unique(y)) < 2:
            raise ValueError("NaiveBayesModel requires at least 2 classes in y.")

        t0 = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
        self.classes_ = self._model.classes_
        return self

    def _validate_predict_input(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or inf).")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        self._validate_predict_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        self._validate_predict_input(X)
        return self._model.predict_proba(X)

    def get_metadata(self) -> dict:
        return {
            "model_name": "Gaussian Naive Bayes",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": self._feature_importance(),
        }

    def _feature_importance(self) -> dict | None:
        """Heuristic, not a native GaussianNB importance score.

        GaussianNB has no coefficient or split-gain analogue. `theta_` is
        the fitted per-class, per-feature mean, shape (n_classes,
        n_features). A feature whose class-conditional mean barely moves
        across classes contributes little to separating them; one whose
        mean varies a lot does the discriminating. The std-dev of each
        feature's per-class means, across classes, is used here as that
        signal. It is not comparable to a tree's gain or a linear model's
        coefficient magnitude -- documented as a heuristic in the README.
        """
        if self.n_features is None:
            return None
        theta = self._model.theta_
        if theta.shape[0] < 2:
            return None
        scores = theta.std(axis=0)
        return dict(zip(
            [f"feature_{i}" for i in range(self.n_features)],
            scores.tolist(),
        ))
