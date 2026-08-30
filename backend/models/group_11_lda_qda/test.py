"""Unit tests for the LDA/QDA submission and its analysis layer."""

import json
import time

import numpy as np
import pytest
from sklearn.datasets import make_classification

from models.base_model import BaseModel
from models.group_11_lda_qda import (
    LDAModel,
    QDAModel,
    analyze_suitability,
    compare_lda_qda,
)

MODEL_CLASSES = (LDAModel, QDAModel)
METADATA_KEYS = {
    "model_name",
    "model_type",
    "hyperparameters",
    "training_time_seconds",
    "n_features",
    "feature_importance",
}


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, encoded_y = make_classification(
        n_samples=180,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=42,
    )
    y = np.where(encoded_y == 1, "yes", "no")
    return X.astype(np.float64), y, X[:20].astype(np.float64)


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=240,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    return X.astype(np.float64), y.astype(np.int16)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_models_subclass_base_model(model_class: type[BaseModel]) -> None:
    assert issubclass(model_class, BaseModel)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_fit_returns_self_and_records_state(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    model = model_class()
    started = time.perf_counter()
    assert model.fit(X, y) is model
    assert model.is_fitted is True
    assert model.n_features == X.shape[1]
    assert (
        model.get_metadata()["training_time_seconds"] <= time.perf_counter() - started
    )
    assert model.get_metadata()["training_time_seconds"] < 300.0


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_predict_preserves_training_label_dtype(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, X_test = binary_data
    predictions = model_class().fit(X, y).predict(X_test)
    assert predictions.shape == (20,)
    assert predictions.dtype == y.dtype
    assert set(predictions).issubset(set(y))


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_integer_label_dtype_is_preserved(
    model_class: type[BaseModel],
    multiclass_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = multiclass_data
    predictions = model_class().fit(X, y).predict(X[:25])
    assert predictions.dtype == np.dtype(np.int16)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_predict_proba_contract(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, X_test = binary_data
    model = model_class().fit(X, y)
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (20, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.classes_, np.unique(y))


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_metadata_contract(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    metadata = model_class().fit(X, y).get_metadata()
    assert set(metadata) == METADATA_KEYS
    assert metadata["model_type"] == "classifier"
    assert metadata["n_features"] == 4
    assert isinstance(metadata["training_time_seconds"], float)
    if model_class is LDAModel:
        assert set(metadata["feature_importance"]) == {
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
        }
    else:
        assert metadata["feature_importance"] is None


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_determinism(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, X_test = binary_data
    first = model_class().fit(X, y)
    second = model_class().fit(X, y)
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
    np.testing.assert_allclose(
        first.predict_proba(X_test), second.predict_proba(X_test)
    )


@pytest.mark.parametrize("operation", ["predict", "predict_proba"])
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_inference_before_fit_raises(
    operation: str,
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, _, X_test = binary_data
    with pytest.raises(RuntimeError, match=r"Call fit\(\)"):
        getattr(model_class(), operation)(X_test)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_feature_count_mismatch_raises(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    model = model_class().fit(X, y)
    with pytest.raises(ValueError, match="trained on 4 features"):
        model.predict(np.ones((3, 5), dtype=np.float64))


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    ("bad_X", "error_type", "message"),
    [
        (np.ones(10, dtype=np.float64), ValueError, "X must be 2D"),
        (np.ones((10, 2), dtype=np.float32), TypeError, "dtype float64"),
        (np.empty((0, 2), dtype=np.float64), ValueError, "at least one sample"),
        (np.empty((2, 0), dtype=np.float64), ValueError, "at least one feature"),
        (
            np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64),
            ValueError,
            "finite values",
        ),
    ],
)
def test_invalid_X_raises_specific_error(
    model_class: type[BaseModel],
    bad_X: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])[: bad_X.shape[0]]
    with pytest.raises(error_type, match=message):
        model_class().fit(bad_X, y)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_non_array_X_raises(model_class: type[BaseModel]) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        model_class().fit([[1.0], [2.0]], np.array([0, 1]))


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_invalid_y_raises_specific_errors(model_class: type[BaseModel]) -> None:
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(ValueError, match="must not be None"):
        model_class().fit(X)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        model_class().fit(X, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="y must be 1D"):
        model_class().fit(X, np.array([[0], [0], [0], [1], [1], [1]]))
    with pytest.raises(ValueError, match="X has 6 rows"):
        model_class().fit(X, np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="at least two target classes"):
        model_class().fit(X, np.zeros(6, dtype=int))
    with pytest.raises(ValueError, match="at least two samples"):
        model_class().fit(X, np.array([0, 0, 0, 0, 0, 1]))
    missing = np.array([0.0, 0.0, 0.0, 1.0, 1.0, np.nan])
    with pytest.raises(ValueError, match="missing or non-finite"):
        model_class().fit(X, missing)


def test_lda_hyperparameter_validation_and_metadata() -> None:
    with pytest.raises(ValueError, match="solver"):
        LDAModel(solver="invalid")
    with pytest.raises(ValueError, match="unsupported"):
        LDAModel(solver="svd", shrinkage="auto")
    with pytest.raises(ValueError, match="between 0 and 1"):
        LDAModel(solver="lsqr", shrinkage=1.5)
    with pytest.raises(ValueError, match="positive integer"):
        LDAModel(n_components=0)
    with pytest.raises(ValueError, match="tol must be positive"):
        LDAModel(tol=0.0)
    model = LDAModel(solver="lsqr", shrinkage="auto")
    assert model.get_metadata()["hyperparameters"]["shrinkage"] == "auto"
    assert model.get_metadata()["feature_importance"] is None


def test_qda_hyperparameter_validation_and_metadata() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        QDAModel(reg_param=-0.1)
    with pytest.raises(ValueError, match="tol must be positive"):
        QDAModel(tol=0.0)
    model = QDAModel(reg_param=0.2)
    assert model.get_metadata()["hyperparameters"]["reg_param"] == 0.2


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_custom_priors_shift_predicted_probabilities(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A strongly skewed prior must move predict_proba away from the
    default (empirical-frequency) prior, proving the parameter is wired
    through to the underlying estimator rather than only stored."""
    X, y, X_test = binary_data
    default_model = model_class().fit(X, y)
    skewed_model = model_class(priors=[0.99, 0.01]).fit(X, y)
    default_probabilities = default_model.predict_proba(X_test)
    skewed_probabilities = skewed_model.predict_proba(X_test)
    assert not np.allclose(default_probabilities, skewed_probabilities)
    assert skewed_model.get_metadata()["hyperparameters"]["priors"] == [0.99, 0.01]
    # The skewed prior favors class 0 ("no"), so its column-0 mass should
    # rise on average relative to the unskewed default.
    assert skewed_probabilities[:, 0].mean() > default_probabilities[:, 0].mean()


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_store_covariance_exposes_fitted_covariance(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """store_covariance=True must actually cause the wrapped estimator to
    retain covariance matrices, not just accept the flag silently."""
    X, y, _ = binary_data
    model = model_class(store_covariance=True).fit(X, y)
    # Both LDA's and QDA's sklearn estimators expose the fitted covariance
    # under the same attribute name.
    covariance_attribute = "covariance_"
    assert hasattr(model._model, covariance_attribute)
    stored = getattr(model._model, covariance_attribute)
    assert stored is not None
    default_model = model_class().fit(X, y)
    assert getattr(default_model._model, covariance_attribute, None) is None


def test_lda_n_components_restricts_transformed_dimensionality(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """n_components must actually be forwarded to the estimator and change
    its behavior, not merely pass hyperparameter validation."""
    X, y, X_test = binary_data
    model = LDAModel(solver="svd", n_components=1).fit(X, y)
    transformed = model._model.transform(X_test)
    assert transformed.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_custom_tol_is_accepted_and_recorded(
    model_class: type[BaseModel],
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A valid, non-default tol must be accepted by fit and preserved in
    the metadata's hyperparameters, not just rejected when invalid."""
    X, y, _ = binary_data
    model = model_class(tol=1e-3).fit(X, y)
    assert model.is_fitted is True
    assert model.get_metadata()["hyperparameters"]["tol"] == 1e-3


def test_unregularized_qda_rejects_insufficient_class_samples() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((12, 6)).astype(np.float64)
    y = np.repeat(np.array([0, 1]), 6)
    with pytest.raises(ValueError, match="more samples in every class than features"):
        QDAModel().fit(X, y)
    assert QDAModel(reg_param=0.2).fit(X, y).is_fitted is True


def test_good_dataset_is_suitable(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    report = analyze_suitability(X, y)
    assert report["target"]["problem_type"] == "binary"
    assert report["lda"]["suitable"] is True
    assert report["qda"]["suitable"] is True
    assert report["can_compare"] is True
    assert report["structural_preference"] in {"LDA", "QDA"}
    json.dumps(report)


def test_multiclass_target_detection(
    multiclass_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = multiclass_data
    report = analyze_suitability(X, y)
    assert report["target"]["problem_type"] == "multiclass"
    assert report["target"]["n_classes"] == 3


def test_regression_like_target_is_rejected() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3)).astype(np.float64)
    y = np.linspace(0.1, 9.9, 100)
    report = analyze_suitability(X, y)
    assert report["target"]["detected_kind"] == "regression_like"
    assert report["can_compare"] is False
    forced = analyze_suitability(X, y, target_kind="classification")
    assert forced["target"]["detected_kind"] == "classification"
    assert forced["can_compare"] is False


def test_numeric_object_target_is_not_mistaken_for_categories() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 3)).astype(np.float64)
    y = np.linspace(0.0, 1.0, 60).astype(object)
    report = analyze_suitability(X, y)
    assert report["target"]["detected_kind"] == "regression_like"


def test_constant_feature_is_reported(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    X_with_constant = np.column_stack((X, np.ones(X.shape[0]))).astype(np.float64)
    report = analyze_suitability(X_with_constant, y)
    assert report["diagnostics"]["constant_feature_indices"] == [4]
    assert report["lda"]["suitable"] is False
    assert report["qda"]["suitable"] is False


def test_qda_sample_size_and_regularization_are_reported() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 12)).astype(np.float64)
    y = np.repeat(np.array(["a", "b"]), 15)
    unregularized = analyze_suitability(X, y, qda_reg_param=0.0)
    regularized = analyze_suitability(X, y, qda_reg_param=0.2)
    assert unregularized["qda"]["suitable"] is False
    assert regularized["qda"]["suitable"] is True
    assert any("regularization" in item for item in regularized["qda"]["warnings"])


def test_imbalance_and_high_correlation_are_informational() -> None:
    rng = np.random.default_rng(42)
    first = rng.standard_normal(100)
    X = np.column_stack(
        (first, first + rng.normal(0, 1e-4, 100), rng.standard_normal(100))
    )
    X = X.astype(np.float64)
    y = np.array(["major"] * 90 + ["minor"] * 10)
    report = analyze_suitability(X, y, qda_reg_param=0.2)
    assert report["data"]["class_imbalance_detected"] is True
    assert report["diagnostics"]["max_absolute_feature_correlation"] > 0.95
    assert any("imbalance" in item for item in report["lda"]["warnings"])
    assert any("correlation" in item for item in report["qda"]["warnings"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_kind": "wrong"}, "target_kind"),
        ({"cv_splits": 1}, "at least 2"),
        ({"qda_reg_param": 1.1}, "between 0 and 1"),
    ],
)
def test_suitability_configuration_validation(
    kwargs: dict[str, object],
    message: str,
) -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match=message):
        analyze_suitability(X, y, **kwargs)


def test_binary_comparison_report_is_complete_and_json_serializable(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    report = compare_lda_qda(X, y)
    assert report["recommended_model"] in {"LDA", "QDA"}
    assert report["configuration"]["random_state"] == 42
    for model_report in report["models"].values():
        assert set(model_report["mean_metrics"]) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        }
        assert np.asarray(model_report["confusion_matrix"]).shape == (2, 2)
        assert len(model_report["out_of_fold_predictions"]) == y.size
    json.dumps(report)


def test_comparison_is_deterministic_except_for_timing(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = binary_data
    first = compare_lda_qda(X, y)
    second = compare_lda_qda(X, y)
    assert first["recommended_model"] == second["recommended_model"]
    for model_name in ("LDA", "QDA"):
        assert (
            first["models"][model_name]["mean_metrics"]
            == second["models"][model_name]["mean_metrics"]
        )
        assert (
            first["models"][model_name]["out_of_fold_predictions"]
            == second["models"][model_name]["out_of_fold_predictions"]
        )


def test_multiclass_comparison(multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = multiclass_data
    report = compare_lda_qda(X, y)
    assert report["configuration"]["class_order"] == [0, 1, 2]
    assert np.asarray(report["models"]["LDA"]["confusion_matrix"]).shape == (3, 3)
    assert 0.0 <= report["models"]["QDA"]["mean_metrics"]["roc_auc"] <= 1.0


def test_comparison_rejects_regression_like_target() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 3)).astype(np.float64)
    y = np.linspace(0.0, 1.0, 80)
    with pytest.raises(ValueError, match="categorical target"):
        compare_lda_qda(X, y)


def test_comparison_rejects_too_many_folds() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((12, 2)).astype(np.float64)
    y = np.repeat(np.array(["a", "b"]), 6)
    with pytest.raises(ValueError, match="at least 7 samples"):
        compare_lda_qda(X, y, n_splits=7, require_suitable=False)


def test_comparison_requires_suitability_unless_overridden() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 12)).astype(np.float64)
    y = np.repeat(np.array(["a", "b"]), 15)
    with pytest.raises(ValueError, match="not suitable"):
        compare_lda_qda(X, y)
    report = compare_lda_qda(X, y, qda_params={"reg_param": 0.2})
    assert report["models"]["QDA"]["mean_metrics"]["accuracy"] >= 0.0
