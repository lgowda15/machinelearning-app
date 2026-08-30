"""Dataset suitability analysis kept outside the LDA and QDA model classes."""

from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._validation import to_python_scalar, validate_X, validate_y

TargetKind = Literal["auto", "classification"]

# These two constants are informal diagnostic cutoffs, not results derived
# from a statistical test. They exist to flag a report line for a human
# reader, not to gate correctness, and the final LDA-vs-QDA recommendation
# in comparison.py always comes from held-out cross-validated performance,
# never from these heuristics alone. Tune them here if your course's
# grading rubric specifies different sensitivity.
#
# _COVARIANCE_DIFFERENCE_THRESHOLD: mean relative Frobenius-norm distance
# between each class covariance and the pooled covariance, above which
# class covariances are considered "meaningfully different" (favoring
# QDA's per-class covariance assumption over LDA's shared one).
_COVARIANCE_DIFFERENCE_THRESHOLD = 0.35

# _HIGH_CORRELATION_THRESHOLD: maximum pairwise absolute feature
# correlation, above which both models' covariance estimates are flagged
# as potentially unstable.
_HIGH_CORRELATION_THRESHOLD = 0.95


def analyze_suitability(
    X: NDArray[np.float64],
    y: NDArray[Any],
    *,
    target_kind: TargetKind = "auto",
    cv_splits: int = 5,
    qda_reg_param: float = 0.0,
) -> dict[str, Any]:
    """Assess whether clean tabular data can support LDA and QDA.

    This function only inspects data. It never mutates, scales, encodes, imputes,
    removes, or rebalances observations or features.
    """
    if target_kind not in {"auto", "classification"}:
        raise ValueError("target_kind must be 'auto' or 'classification'.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    if not 0.0 <= qda_reg_param <= 1.0:
        raise ValueError("qda_reg_param must be between 0 and 1.")

    X_valid = validate_X(X)
    y_valid = validate_y(y, n_samples=X_valid.shape[0])
    try:
        classes, counts = np.unique(y_valid, return_counts=True)
    except TypeError as exc:
        raise ValueError("y must contain mutually comparable class labels.") from exc

    n_samples, n_features = X_valid.shape
    n_classes = int(classes.size)
    detected_kind, detection_message = _detect_target_kind(
        y_valid,
        n_classes=n_classes,
        forced=target_kind == "classification",
    )
    problem_type = None
    if detected_kind == "classification":
        problem_type = "binary" if n_classes == 2 else "multiclass"

    proportions = counts / n_samples
    imbalance_detected = bool(np.any(proportions < 0.20) or np.any(proportions > 0.80))
    class_distribution = [
        {
            "label": to_python_scalar(label),
            "count": int(count),
            "proportion": float(proportion),
        }
        for label, count, proportion in zip(classes, counts, proportions, strict=True)
    ]

    base_report: dict[str, Any] = {
        "target": {
            "detected_kind": detected_kind,
            "problem_type": problem_type,
            "n_classes": n_classes,
            "classes": [to_python_scalar(label) for label in classes],
            "class_distribution": class_distribution,
            "detection_message": detection_message,
        },
        "data": {
            "n_samples": n_samples,
            "n_features": n_features,
            "cv_splits": cv_splits,
            "class_imbalance_detected": imbalance_detected,
        },
    }

    if detected_kind != "classification" or n_classes < 2:
        reason = (
            "The target is regression-like under automatic detection. "
            "Confirm categorical meaning with target_kind='classification' if needed."
            if n_classes >= 2
            else "At least two target classes are required."
        )
        base_report.update(
            {
                "diagnostics": _empty_diagnostics(),
                "lda": {"suitable": False, "reasons": [], "warnings": [reason]},
                "qda": {"suitable": False, "reasons": [], "warnings": [reason]},
                "can_compare": False,
                "structural_preference": None,
            }
        )
        return base_report

    diagnostics = _covariance_diagnostics(X_valid, y_valid, classes, counts)
    lda_result = _assess_lda(
        counts=counts,
        diagnostics=diagnostics,
        imbalance_detected=imbalance_detected,
    )
    qda_result = _assess_qda(
        counts=counts,
        n_features=n_features,
        cv_splits=cv_splits,
        qda_reg_param=qda_reg_param,
        diagnostics=diagnostics,
        imbalance_detected=imbalance_detected,
    )
    can_cross_validate = bool(np.min(counts) >= cv_splits)
    can_compare = bool(
        lda_result["suitable"] and qda_result["suitable"] and can_cross_validate
    )

    if not can_cross_validate:
        message = (
            f"Each class needs at least {cv_splits} samples for stratified "
            f"{cv_splits}-fold comparison."
        )
        lda_result["warnings"].append(message)
        qda_result["warnings"].append(message)

    difference = diagnostics["covariance_difference_score"]
    structural_preference = "LDA"
    if difference is not None and difference > _COVARIANCE_DIFFERENCE_THRESHOLD:
        structural_preference = "QDA" if qda_result["suitable"] else "LDA"

    base_report.update(
        {
            "diagnostics": diagnostics,
            "lda": lda_result,
            "qda": qda_result,
            "can_compare": can_compare,
            "structural_preference": structural_preference,
        }
    )
    return base_report


def _detect_target_kind(
    y: NDArray[Any],
    *,
    n_classes: int,
    forced: bool,
) -> tuple[str, str]:
    if forced:
        return (
            "classification",
            "Classification was explicitly confirmed by the caller.",
        )
    if n_classes < 2:
        return "invalid", "The target contains fewer than two distinct values."
    if y.dtype.kind in {"b", "S", "U"}:
        return "classification", "Non-continuous labels indicate classification."
    if y.dtype.kind == "O" and not all(
        isinstance(value, Real) and not isinstance(value, bool | np.bool_)
        for value in y
    ):
        return "classification", "Non-continuous labels indicate classification."
    classification_limit = max(20, int(np.sqrt(y.size)))
    if n_classes <= classification_limit:
        return (
            "classification",
            f"The numeric target has {n_classes} repeated discrete values.",
        )
    return (
        "regression_like",
        "The numeric target has many distinct values relative to sample size.",
    )


def _covariance_diagnostics(
    X: NDArray[np.float64],
    y: NDArray[Any],
    classes: NDArray[Any],
    counts: NDArray[np.intp],
) -> dict[str, Any]:
    variances = np.var(X, axis=0)
    constant_features = np.flatnonzero(variances <= np.finfo(np.float64).eps).tolist()
    max_abs_correlation = _max_abs_correlation(X, constant_features)

    class_covariances: list[NDArray[np.float64]] = []
    class_ranks: list[int | None] = []
    class_conditions: list[float | None] = []
    for label, count in zip(classes, counts, strict=True):
        if count < 2:
            class_ranks.append(None)
            class_conditions.append(None)
            continue
        covariance = np.atleast_2d(np.cov(X[y == label], rowvar=False, ddof=1))
        class_covariances.append(covariance)
        class_ranks.append(int(np.linalg.matrix_rank(covariance)))
        condition = float(np.linalg.cond(covariance))
        class_conditions.append(condition if np.isfinite(condition) else None)

    pooled_rank = None
    pooled_condition = None
    covariance_difference = None
    if len(class_covariances) == classes.size:
        denominator = int(np.sum(counts) - classes.size)
        if denominator > 0:
            pooled = (
                sum(
                    (int(count) - 1) * covariance
                    for count, covariance in zip(counts, class_covariances, strict=True)
                )
                / denominator
            )
            pooled_rank = int(np.linalg.matrix_rank(pooled))
            condition = float(np.linalg.cond(pooled))
            pooled_condition = condition if np.isfinite(condition) else None
            scale = max(float(np.linalg.norm(pooled, ord="fro")), np.finfo(float).eps)
            covariance_difference = float(
                np.mean(
                    [
                        np.linalg.norm(covariance - pooled, ord="fro") / scale
                        for covariance in class_covariances
                    ]
                )
            )

    return {
        "n_features": X.shape[1],
        "constant_feature_indices": constant_features,
        "max_absolute_feature_correlation": max_abs_correlation,
        "high_correlation_threshold": _HIGH_CORRELATION_THRESHOLD,
        "pooled_covariance_rank": pooled_rank,
        "pooled_covariance_condition_number": pooled_condition,
        "class_covariance_ranks": class_ranks,
        "class_covariance_condition_numbers": class_conditions,
        "covariance_difference_score": covariance_difference,
        "covariance_difference_threshold": _COVARIANCE_DIFFERENCE_THRESHOLD,
    }


def _max_abs_correlation(
    X: NDArray[np.float64],
    constant_features: list[int],
) -> float | None:
    if X.shape[1] < 2 or constant_features:
        return None
    correlation = np.corrcoef(X, rowvar=False)
    upper_triangle = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
    return float(np.max(upper_triangle)) if upper_triangle.size else 0.0


def _assess_lda(
    *,
    counts: NDArray[np.intp],
    diagnostics: dict[str, Any],
    imbalance_detected: bool,
) -> dict[str, Any]:
    reasons = [
        "The target is categorical with at least two classes.",
        "LDA can estimate class means from the available observations.",
    ]
    warnings: list[str] = []
    suitable = bool(np.min(counts) >= 2)
    if np.min(counts) < 2:
        warnings.append("Every class needs at least two observations for LDA.")
    if diagnostics["constant_feature_indices"]:
        suitable = False
        warnings.append(
            "Constant features were found; the backend should remove them "
            "before fitting."
        )
    pooled_rank = diagnostics["pooled_covariance_rank"]
    if pooled_rank is not None and pooled_rank < diagnostics["n_features"]:
        warnings.append(
            "The pooled covariance is rank-deficient; the default SVD solver "
            "may still fit."
        )
    correlation = diagnostics["max_absolute_feature_correlation"]
    if correlation is not None and correlation > _HIGH_CORRELATION_THRESHOLD:
        warnings.append(
            "Very high feature correlation may make LDA estimates unstable."
        )
    difference = diagnostics["covariance_difference_score"]
    if difference is not None and difference > _COVARIANCE_DIFFERENCE_THRESHOLD:
        warnings.append(
            "Class covariance matrices differ substantially from the pooled matrix."
        )
    if imbalance_detected:
        warnings.append(
            "Class imbalance was detected; report it, but do not rebalance "
            "inside the model."
        )
    return {"suitable": suitable, "reasons": reasons, "warnings": warnings}


def _assess_qda(
    *,
    counts: NDArray[np.intp],
    n_features: int,
    cv_splits: int,
    qda_reg_param: float,
    diagnostics: dict[str, Any],
    imbalance_detected: bool,
) -> dict[str, Any]:
    reasons = ["The target is categorical with at least two classes."]
    warnings: list[str] = []
    suitable = bool(np.min(counts) >= 2)
    minimum_training_count = int(
        np.min(counts - np.ceil(counts / cv_splits).astype(int))
    )

    if np.min(counts) < 2:
        warnings.append("Every class needs at least two observations for QDA.")

    if diagnostics["constant_feature_indices"]:
        suitable = False
        warnings.append(
            "Constant features were found; the backend should remove them "
            "before fitting."
        )
    singular_class_covariance = any(
        rank is None or rank < n_features
        for rank in diagnostics["class_covariance_ranks"]
    )
    if qda_reg_param == 0.0:
        if minimum_training_count <= n_features:
            suitable = False
            warnings.append(
                "Unregularized QDA needs more training samples per class than features "
                "inside every cross-validation fold."
            )
        else:
            reasons.append(
                "Every cross-validation training fold has more class samples "
                "than features."
            )
        if singular_class_covariance:
            suitable = False
            warnings.append("At least one class covariance matrix is rank-deficient.")
    else:
        reasons.append(
            f"QDA covariance regularization is enabled with reg_param={qda_reg_param}."
        )
        if minimum_training_count <= n_features or singular_class_covariance:
            warnings.append(
                "QDA relies on regularization because a class covariance "
                "estimate is weak."
            )
    correlation = diagnostics["max_absolute_feature_correlation"]
    if correlation is not None and correlation > _HIGH_CORRELATION_THRESHOLD:
        warnings.append(
            "Very high feature correlation may make QDA estimates unstable."
        )
    if imbalance_detected:
        warnings.append(
            "Class imbalance was detected; report it, but do not rebalance "
            "inside the model."
        )
    return {"suitable": suitable, "reasons": reasons, "warnings": warnings}


def _empty_diagnostics() -> dict[str, Any]:
    return {
        "n_features": None,
        "constant_feature_indices": [],
        "max_absolute_feature_correlation": None,
        "high_correlation_threshold": _HIGH_CORRELATION_THRESHOLD,
        "pooled_covariance_rank": None,
        "pooled_covariance_condition_number": None,
        "class_covariance_ranks": [],
        "class_covariance_condition_numbers": [],
        "covariance_difference_score": None,
        "covariance_difference_threshold": _COVARIANCE_DIFFERENCE_THRESHOLD,
    }
