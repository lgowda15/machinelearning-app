import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier for the ML integration platform."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | None = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.classes_ = None
        self._train_time = None
        self._shap_X = None

    @staticmethod
    def _validate_X(X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if X.dtype != np.float64:
            raise TypeError(f"X must have dtype float64, got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "RandomForestModel":
        self._validate_X(X)

        if y is None:
            raise ValueError("RandomForestModel is supervised; y must not be None.")
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows, y has {y.shape[0]}."
            )
        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")
        if np.any(self._missing_target_mask(y)):
            raise ValueError("y must not contain missing target values.")

        if np.unique(y).size < 2:
            raise ValueError("Classification requires at least two target classes.")

        t0 = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
        self.classes_ = self._model.classes_
        self._shap_X = X.copy()
        return self

    @staticmethod
    def _missing_target_mask(y: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([value is None for value in y])
        except (TypeError, ValueError):
            return np.zeros(y.shape[0], dtype=bool)

    def _validate_predict_X(self, X: np.ndarray) -> None:
        self._validate_X(X)
        if not self.is_fitted:
            raise RuntimeError("Call fit() before prediction.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, "
                f"got {X.shape[1]}."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_predict_X(X)
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        self._validate_predict_X(X)
        probabilities = np.asarray(self._model.predict_proba(X), dtype=float)

        if probabilities.ndim != 2:
            raise RuntimeError("Random Forest returned invalid probability shape.")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise RuntimeError("Random Forest probabilities are not normalized.")

        return probabilities

    def get_metadata(self) -> dict:
        return {
            "model_name": "Random Forest",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": None if not self.is_fitted else float(self._train_time),
            "n_features": None if not self.is_fitted else int(self.n_features),
            "feature_importance": None if not self.is_fitted else {
                f"feature_{i}": float(score)
                for i, score in enumerate(self._model.feature_importances_)
            },
        }

    def get_visualization_data(self) -> dict | None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before get_visualization_data().")

        import shap

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(self._shap_X)
        return {"shap_values": _shap_to_nested_lists(shap_values)}


def _shap_to_nested_lists(shap_values):
    if isinstance(shap_values, list):
        return [np.asarray(value).tolist() for value in shap_values]
    return np.asarray(shap_values).tolist()
