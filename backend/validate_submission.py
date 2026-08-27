"""Submission validator.

Usage:
    python validate_submission.py models/group_13_svm/

Checks a group's model against the interface contract. This is a working
skeleton; the integration team extends the checks marked TODO before the
first submission deadline.
"""
import sys
import importlib.util
import inspect
from pathlib import Path

import numpy as np


REQUIRED_METADATA_KEYS = {
    "model_name", "model_type", "hyperparameters",
    "training_time_seconds", "n_features", "feature_importance",
}
VALID_MODEL_TYPES = {
    "classifier", "clusterer", "regressor", "dimensionality_reducer",
}


def fail(msg):
    print(f"  FAIL: {msg}")
    return False


def load_model_class(folder: Path):
    model_file = folder / "model.py"
    if not model_file.exists():
        fail(f"no model.py in {folder}")
        return None
    spec = importlib.util.spec_from_file_location("submission_model", model_file)
    module = importlib.util.module_from_spec(spec)
    # base_model must be importable; add backend/ to path
    sys.path.insert(0, str(folder.parents[1]))
    spec.loader.exec_module(module)
    from models.base_model import BaseModel
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseModel) and obj is not BaseModel:
            return obj
    fail("no BaseModel subclass found in model.py")
    return None


def validate(folder: Path) -> bool:
    print(f"Validating {folder} ...")
    ok = True

    cls = load_model_class(folder)
    if cls is None:
        return False

    for method in ("fit", "predict", "predict_proba", "get_metadata"):
        if not hasattr(cls, method):
            ok = fail(f"missing method: {method}")

    # smoke test on synthetic data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 4))
    y = rng.integers(0, 2, 60)
    try:
        model = cls()
        returned = model.fit(X, y)
        if returned is not model:
            ok = fail("fit() must return self")
        if not getattr(model, "is_fitted", False):
            ok = fail("fit() must set is_fitted = True")
        preds = model.predict(X)
        if not isinstance(preds, np.ndarray) or preds.ndim != 1:
            ok = fail("predict() must return a 1D numpy array")
        elif preds.shape[0] != X.shape[0]:
            ok = fail("predict() length must equal n_samples")
        md = model.get_metadata()
        if set(md.keys()) != REQUIRED_METADATA_KEYS:
            ok = fail(f"metadata keys must be exactly {sorted(REQUIRED_METADATA_KEYS)}")
        elif md["model_type"] not in VALID_MODEL_TYPES:
            ok = fail(f"model_type must be one of {sorted(VALID_MODEL_TYPES)}")
    except Exception as e:  # noqa: BLE001
        ok = fail(f"smoke test raised: {type(e).__name__}: {e}")

    # TODO(integration team): determinism check (fit+predict twice, compare)
    # TODO(integration team): predict_proba shape + row-sum check for classifiers
    # TODO(integration team): training-time ceiling (5 min)
    # TODO(integration team): ruff + coverage gate (run in CI)

    print("  PASS" if ok else "  ---")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python validate_submission.py <group_folder>")
        sys.exit(2)
    sys.exit(0 if validate(Path(sys.argv[1])) else 1)
