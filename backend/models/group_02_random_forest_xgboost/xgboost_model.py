import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier for the ML integration platform."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in the interval (0, 1].")
        if not 0 < colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in the interval (0, 1].")

        self.random_state = random_state
        self._model = None
        self._label_encoder = LabelEncoder()
        self.classes_ = None
        self._train_time = None
        self._shap_X = None

        self._xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }

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
    ) -> "XGBoostModel":
        self._validate_X(X)

        if y is None:
            raise ValueError("XGBoostModel is supervised; y must not be None.")
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

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Classification requires at least two target classes.")

        encoded_y = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        n_classes = len(self.classes_)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

        params = dict(self._xgb_params)
        params["objective"] = objective
        if n_classes > 2:
            params["num_class"] = n_classes

        self._model = XGBClassifier(**params)

        t0 = time.perf_counter()
        self._model.fit(X, encoded_y)
        self._train_time = time.perf_counter() - t0

        self.is_fitted = True
        self.n_features = X.shape[1]
        self._shap_X = X.copy()
        return self

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
        encoded_predictions = np.asarray(self._model.predict(X), dtype=int)
        return self._label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        self._validate_predict_X(X)
        probabilities = np.asarray(self._model.predict_proba(X), dtype=float)

        if probabilities.ndim != 2:
            raise RuntimeError("XGBoost returned invalid probability shape.")
        if probabilities.shape[1] != len(self.classes_):
            raise RuntimeError("XGBoost probability columns do not match classes.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("XGBoost probabilities are not normalized.")

        return probabilities

    def get_metadata(self) -> dict:
        return {
            "model_name": "XGBoost",
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
