"""Support Vector Machine classifier for the ML integration platform.

Group 13. Wraps :class:`sklearn.svm.SVC` and exposes every kernel the
standards document requires (linear, poly, rbf, sigmoid) through a single
constructor hyperparameter, rather than one class per kernel. Section 6 of
the Coding Standards is explicit on this point: "Variants of one algorithm
that differ only by a hyperparameter (linkage criteria in hierarchical
clustering, kernel choice in SVM) are one class with a hyperparameter, not
separate classes."

This model receives input that the backend has already scaled, encoded,
and imputed (Section 4 of the Coding Standards). It performs no
preprocessing of its own.
"""

import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from models.base_model import BaseModel

# The only kernels the platform allows a group-13 instance to select.
# SVC itself also accepts "precomputed" and arbitrary callables; the
# standards document restricts this model to exactly these four.
SUPPORTED_KERNELS = ("linear", "poly", "rbf", "sigmoid")

# Number of cross-validation folds used internally to calibrate
# probability estimates. Not user-tunable: it is an implementation
# detail of how probabilities are produced, not a modelling choice.
_CALIBRATION_FOLDS = 5


class SVMModel(BaseModel):
    """Support Vector Machine classifier with a selectable kernel.

    Thin, validating wrapper around ``sklearn.svm.SVC``. Supports the
    linear, polynomial, RBF, and sigmoid kernels via the ``kernel``
    constructor argument; switching kernels does not require a different
    class.

    Parameters
    ----------
    kernel : str, default "rbf"
        One of ``"linear"``, ``"poly"``, ``"rbf"``, ``"sigmoid"``.
    C : float, default 1.0
        Regularization strength. Must be positive; smaller values
        specify stronger regularization.
    gamma : {"scale", "auto"} or float, default "scale"
        Kernel coefficient for ``"rbf"``, ``"poly"``, and ``"sigmoid"``.
        Ignored by the ``"linear"`` kernel.
    degree : int, default 3
        Degree of the polynomial kernel. Ignored by all other kernels.
    coef0 : float, default 0.0
        Independent term used by the ``"poly"`` and ``"sigmoid"``
        kernels. Ignored by ``"linear"`` and ``"rbf"``.
    random_state : int, default 42
        Seed controlling the internal probability-calibration routine,
        so that repeated fits on identical data are reproducible.

    Attributes
    ----------
    classes_ : np.ndarray or None
        Class labels seen during ``fit``, in the order used by the
        columns of ``predict_proba``'s output. ``None`` before fitting.
    is_fitted : bool
        Whether ``fit`` has been called successfully.
    n_features : int or None
        Number of features seen during ``fit``. ``None`` before fitting.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if kernel not in SUPPORTED_KERNELS:
            raise ValueError(
                f"kernel must be one of {SUPPORTED_KERNELS}, got {kernel!r}."
            )

        # Every argument forwarded to the estimator is recorded verbatim
        # so get_metadata() can report "every hyperparameter used"
        # (Section 8) without duplicating this list a second time.
        super().__init__(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            random_state=random_state,
            **kwargs,
        )

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

        # SVC's own probability=True flag is deprecated as of
        # scikit-learn 1.9 (removal slated for 1.11); scikit-learn's own
        # deprecation notice recommends CalibratedClassifierCV(SVC(),
        # ensemble=False) instead, which is what this wraps.
        base_svc = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            random_state=random_state,
            **kwargs,
        )
        self._estimator = CalibratedClassifierCV(
            estimator=base_svc,
            cv=_CALIBRATION_FOLDS,
            ensemble=False,
        )

        self.classes_: np.ndarray | None = None
        self._training_time_seconds: float | None = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_X(X: np.ndarray) -> None:
        """Validate a feature matrix against the platform's input contract.

        The backend guarantees float64, 2D, non-empty input (Section 4),
        but this model checks it anyway: a defect elsewhere in the
        pipeline should surface here as a clear exception rather than a
        confusing failure deep inside scikit-learn.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy ndarray, got {type(X).__name__}.")
        if X.dtype != np.float64:
            raise TypeError(f"X must have dtype float64, got {X.dtype}.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X has zero samples.")
        if X.shape[1] == 0:
            raise ValueError("X has zero features.")

    @staticmethod
    def _validate_y(y: np.ndarray | None, n_samples: int) -> None:
        """Validate training labels against the platform's input contract."""
        if y is None:
            raise ValueError("SVMModel is a supervised classifier; y must not be None.")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy ndarray, got {type(y).__name__}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if y.shape[0] == 0:
            raise ValueError("y has zero labels.")
        if y.shape[0] != n_samples:
            raise ValueError(f"X has {n_samples} rows, y has {y.shape[0]} labels.")

    def _require_fitted(self, method_name: str) -> None:
        """Raise if a caller reaches an inference method before fit()."""
        if not self.is_fitted:
            raise RuntimeError(f"Call fit() before {method_name}().")

    def _require_matching_features(self, X: np.ndarray) -> None:
        """Raise if X's feature count does not match the fitted model."""
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SVMModel":
        """Train the classifier and return self.

        Parameters
        ----------
        X : np.ndarray
            2D float64 array, shape (n_samples, n_features).
        y : np.ndarray
            1D array of class labels, shape (n_samples,).

        Returns
        -------
        SVMModel
            This instance, now fitted.

        Raises
        ------
        TypeError
            If X or y is not a numpy ndarray, or X is not float64.
        ValueError
            If y is None, dimensions are wrong, or sample counts mismatch.
        """
        self._validate_X(X)
        self._validate_y(y, n_samples=X.shape[0])

        start = time.perf_counter()
        self._estimator.fit(X, y)
        self._training_time_seconds = time.perf_counter() - start

        self.is_fitted = True
        self.n_features = X.shape[1]
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,), dtype matching the training
            labels, values drawn from ``self.classes_``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        TypeError, ValueError
            If X fails input validation or has the wrong feature count.
        """
        self._require_fitted("predict")
        self._validate_X(X)
        self._require_matching_features(X)
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, n_classes). Rows sum to 1.
            Columns are ordered to match ``self.classes_``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        TypeError, ValueError
            If X fails input validation or has the wrong feature count.
        """
        self._require_fitted("predict_proba")
        self._validate_X(X)
        self._require_matching_features(X)
        return self._estimator.predict_proba(X)

    def get_metadata(self) -> dict:
        """Return the platform's required metadata dict (Section 8).

        ``feature_importance`` is always ``None``: scikit-learn's SVC has
        no native, kernel-independent notion of per-feature importance
        (only the linear kernel exposes ``coef_``, and even that is not a
        true importance score once inputs are already scaled the same
        way for every kernel choice). Returning ``None`` uniformly keeps
        the contract's return type consistent regardless of which kernel
        was selected.
        """
        return {
            "model_name": "Support Vector Machine",
            "model_type": "classifier",
            "hyperparameters": dict(self.hyperparams),
            "training_time_seconds": self._training_time_seconds,
            "n_features": self.n_features,
            "feature_importance": None,
        }
