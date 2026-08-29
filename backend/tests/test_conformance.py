"""Conformance suite for the ML Integration Platform.

Parametrises over every model in app.core.registry.REGISTRY and asserts the
contract from CODING_STANDARDS.md SS4/7/8, branching the predict-shape and
predict_proba assertions by model_type -- CLAUDE.md: "model_type is a
switch, not documentation."

This suite is what replaces validate_submission.py as the merge gate. It
must pass against all four reference models before any group model is
registered.
"""
import numpy as np
import pytest

from app.core.registry import REGISTRY
from models.base_model import BaseModel

REQUIRED_METADATA_KEYS = {
    "model_name", "model_type", "hyperparameters",
    "training_time_seconds", "n_features", "feature_importance",
}
VALID_MODEL_TYPES = {"classifier", "clusterer", "regressor", "dimensionality_reducer"}


def _make_data(rng, n_samples, n_features, model_type):
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    if model_type == "classifier":
        y = rng.integers(0, 2, n_samples)
    elif model_type == "regressor":
        y = rng.standard_normal(n_samples)
    elif model_type in ("clusterer", "dimensionality_reducer"):
        y = None
    else:
        raise ValueError(f"Unrecognised model_type: {model_type!r}")
    return X, y


@pytest.fixture(params=list(REGISTRY.items()), ids=list(REGISTRY.keys()))
def reference(request):
    name, cls = request.param
    model_type = cls().get_metadata()["model_type"]
    rng = np.random.default_rng(42)
    X_train, y_train = _make_data(rng, 100, 4, model_type)
    X_test, _ = _make_data(rng, 20, 4, model_type)
    return {
        "name": name,
        "cls": cls,
        "model_type": model_type,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
    }


class TestConformance:
    def test_subclasses_base_model(self, reference):
        assert issubclass(reference["cls"], BaseModel)

    def test_fit_returns_self_and_sets_is_fitted(self, reference):
        m = reference["cls"]()
        result = m.fit(reference["X_train"], reference["y_train"])
        assert result is m
        assert m.is_fitted is True

    def test_predict_shape_by_model_type(self, reference):
        m = reference["cls"]().fit(reference["X_train"], reference["y_train"])
        preds = m.predict(reference["X_test"])
        n_samples = reference["X_test"].shape[0]

        if reference["model_type"] == "dimensionality_reducer":
            # Contract exception: 2D (n_samples, n_components), not 1D.
            assert preds.ndim == 2
            assert preds.shape[0] == n_samples
        else:
            assert preds.shape == (n_samples,)

    def test_predict_proba_contract(self, reference):
        m = reference["cls"]().fit(reference["X_train"], reference["y_train"])
        proba = m.predict_proba(reference["X_test"])

        if reference["model_type"] == "classifier":
            assert proba is not None
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        else:
            assert proba is None

    def test_metadata_contract(self, reference):
        m = reference["cls"]().fit(reference["X_train"], reference["y_train"])
        md = m.get_metadata()

        assert set(md.keys()) == REQUIRED_METADATA_KEYS
        assert md["model_type"] == reference["model_type"]
        assert md["model_type"] in VALID_MODEL_TYPES
        assert md["n_features"] == reference["X_train"].shape[1]

    def test_determinism(self, reference):
        a = (
            reference["cls"]()
            .fit(reference["X_train"], reference["y_train"])
            .predict(reference["X_test"])
        )
        b = (
            reference["cls"]()
            .fit(reference["X_train"], reference["y_train"])
            .predict(reference["X_test"])
        )
        np.testing.assert_array_equal(a, b)
