"""Deterministic cross-validated comparison of LDA and QDA."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from ._validation import to_python_scalar, validate_X, validate_y
from .lda import LDAModel
from .qda import QDAModel
from .suitability import TargetKind, analyze_suitability

ModelFactory = Callable[[], LDAModel | QDAModel]


def compare_lda_qda(
    X: NDArray[np.float64],
    y: NDArray[Any],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    lda_params: dict[str, Any] | None = None,
    qda_params: dict[str, Any] | None = None,
    target_kind: TargetKind = "auto",
    require_suitable: bool = True,
) -> dict[str, Any]:
    """Compare LDA and QDA using deterministic out-of-fold predictions."""
    X_valid = validate_X(X)
    y_valid = validate_y(y, n_samples=X_valid.shape[0])
    lda_config = dict(lda_params or {})
    qda_config = dict(qda_params or {})
    qda_reg_param = float(qda_config.get("reg_param", 0.0))
    suitability = analyze_suitability(
        X_valid,
        y_valid,
        target_kind=target_kind,
        cv_splits=n_splits,
        qda_reg_param=qda_reg_param,
    )
    if suitability["target"]["detected_kind"] != "classification":
        raise ValueError(
            "LDA/QDA comparison requires a categorical target; automatic detection "
            "found a regression-like target."
        )
    minimum_class_count = min(
        item["count"] for item in suitability["target"]["class_distribution"]
    )
    if minimum_class_count < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} samples for stratified "
            f"{n_splits}-fold comparison."
        )
    if require_suitable and not suitability["can_compare"]:
        issues = suitability["lda"]["warnings"] + suitability["qda"]["warnings"]
        detail = " ".join(dict.fromkeys(issues))
        raise ValueError(f"Dataset is not suitable for an LDA/QDA comparison. {detail}")

    classes = np.unique(y_valid)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    factories: dict[str, ModelFactory] = {
        "LDA": lambda: LDAModel(**lda_config),
        "QDA": lambda: QDAModel(**qda_config),
    }
    model_reports = {
        model_name: _cross_validate_model(
            model_factory,
            X_valid,
            y_valid,
            classes,
            splitter,
        )
        for model_name, model_factory in factories.items()
    }
    recommended_model, reason = _recommend_model(model_reports)
    return {
        "configuration": {
            "n_splits": n_splits,
            "random_state": random_state,
            "averaging": "weighted",
            "class_order": [to_python_scalar(label) for label in classes],
        },
        "suitability": suitability,
        "models": model_reports,
        "recommended_model": recommended_model,
        "recommendation_metric": "mean weighted F1, then ROC-AUC, then accuracy",
        "recommendation_reason": reason,
    }


def _cross_validate_model(
    model_factory: ModelFactory,
    X: NDArray[np.float64],
    y: NDArray[Any],
    classes: NDArray[Any],
    splitter: StratifiedKFold,
) -> dict[str, Any]:
    out_of_fold_predictions = np.empty_like(y)
    out_of_fold_probabilities = np.zeros((y.size, classes.size), dtype=np.float64)
    fold_metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }
    total_training_time = 0.0

    for train_indices, validation_indices in splitter.split(X, y):
        model = model_factory()
        model.fit(X[train_indices], y[train_indices])
        predictions = model.predict(X[validation_indices])
        probabilities = model.predict_proba(X[validation_indices])
        aligned_probabilities = _align_probabilities(
            probabilities,
            model.classes_,
            classes,
        )
        out_of_fold_predictions[validation_indices] = predictions
        out_of_fold_probabilities[validation_indices] = aligned_probabilities
        metrics = _calculate_metrics(
            y[validation_indices],
            predictions,
            aligned_probabilities,
            classes,
        )
        for metric_name, value in metrics.items():
            fold_metrics[metric_name].append(value)
        training_time = model.get_metadata()["training_time_seconds"]
        total_training_time += float(training_time)

    mean_metrics = {
        metric_name: float(np.mean(values))
        for metric_name, values in fold_metrics.items()
    }
    standard_deviations = {
        metric_name: float(np.std(values, ddof=0))
        for metric_name, values in fold_metrics.items()
    }
    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "standard_deviation": standard_deviations,
        "confusion_matrix": confusion_matrix(
            y,
            out_of_fold_predictions,
            labels=classes,
        ).tolist(),
        "out_of_fold_predictions": [
            to_python_scalar(label) for label in out_of_fold_predictions
        ],
        "total_training_time_seconds": total_training_time,
    }


def _align_probabilities(
    probabilities: NDArray[np.float64],
    model_classes: NDArray[Any] | None,
    expected_classes: NDArray[Any],
) -> NDArray[np.float64]:
    if model_classes is None:
        raise RuntimeError("Fitted classifier did not expose classes_.")
    aligned = np.zeros(
        (probabilities.shape[0], expected_classes.size), dtype=np.float64
    )
    for source_index, label in enumerate(model_classes):
        matches = np.flatnonzero(expected_classes == label)
        if matches.size != 1:
            raise RuntimeError(
                "Cross-validation fold produced inconsistent class labels."
            )
        aligned[:, int(matches[0])] = probabilities[:, source_index]
    return aligned


def _calculate_metrics(
    y_true: NDArray[Any],
    predictions: NDArray[Any],
    probabilities: NDArray[np.float64],
    classes: NDArray[Any],
) -> dict[str, float]:
    if classes.size == 2:
        binary_target = (y_true == classes[1]).astype(np.int8)
        roc_auc = roc_auc_score(binary_target, probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc),
    }


def _recommend_model(model_reports: dict[str, dict[str, Any]]) -> tuple[str, str]:
    metrics_in_order = ("f1", "roc_auc", "accuracy")
    for metric in metrics_in_order:
        lda_score = model_reports["LDA"]["mean_metrics"][metric]
        qda_score = model_reports["QDA"]["mean_metrics"][metric]
        if not np.isclose(lda_score, qda_score, atol=1e-12, rtol=0.0):
            winner = "LDA" if lda_score > qda_score else "QDA"
            return (
                winner,
                (
                    f"{winner} has the higher mean cross-validated {metric}: "
                    f"{max(lda_score, qda_score):.6f} versus "
                    f"{min(lda_score, qda_score):.6f}."
                ),
            )
    return "LDA", "All ranking metrics are tied; LDA is preferred as the simpler model."
