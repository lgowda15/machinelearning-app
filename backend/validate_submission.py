"""
validate_submission.py — interface-contract validator for the ML Integration
Platform (course UM25MB653CA2, https://github.com/lgowda15/machinelearning-app).

Belongs at ``backend/validate_submission.py``. Run it from ``backend/`` before
opening a pull request:

    python validate_submission.py models/group_<NN>_<name>/

What it checks (CODING_STANDARDS.md, Section 11):

  1.  The folder imports as a package and every exported class subclasses
      BaseModel with all four required methods implemented.
  2.  fit(X, y) returns self and sets is_fitted = True.
  3.  predict(X) returns the correctly-shaped array for the model's type.
  4.  predict_proba(X) is None for non-classifiers, or a normalised
      (n_samples, n_classes) array (rows sum to 1) for classifiers.
  5.  get_metadata() returns exactly the required keys with a valid,
      JSON-friendly model_type and hyperparameters.
  6.  Determinism — two independent fit+predict cycles on identical input
      produce identical output.
  7.  Training completes within the 5-minute CPU ceiling.
  8.  No print() calls, no hardcoded absolute paths, no committed data
      files, no .ipynb files in the submission folder.
  9.  ruff reports no lint issues.
  10. pytest passes with >= 80% coverage on the submission folder.

Exit code 0 means every check passed. Any failure exits non-zero, and the
printed report says exactly which check failed and why — that report is
what should go back to the group, not a re-explanation of the standards doc.

Sequence/image groups (3, 7, 8): if a file
``fixtures/group_<NN>_fixture.py`` exists next to this script and defines
``make_fixture(seed) -> (X, y, X_test)``, it is used instead of the generic
tabular fixture below. Nothing under backend/fixtures/ exists yet for
groups 3/7/8 — that is separate outstanding work, not part of this script.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_METADATA_KEYS = {
    "model_name",
    "model_type",
    "hyperparameters",
    "training_time_seconds",
    "n_features",
    "feature_importance",
}
VALID_MODEL_TYPES = {"classifier", "clusterer", "regressor", "dimensionality_reducer"}
TRAINING_TIME_CEILING_SECONDS = 5 * 60
COVERAGE_THRESHOLD = 80
DISALLOWED_DATA_EXTENSIONS = {".csv", ".tsv", ".npy", ".npz", ".pkl", ".pickle", ".parquet", ".ipynb"}

# Heuristic patterns for hardcoded absolute paths in source. Deliberately
# conservative (string literals only) to avoid false positives on comments
# describing shapes like "(n_samples, n_features)".
ABS_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]{1,2}(Users|home|Documents|Desktop)", re.IGNORECASE),
    re.compile(r"/(home|Users|mnt|tmp)/[^\s\"']+"),
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True  # if False, a failure is reported but doesn't block the whole run


class SubmissionError(Exception):
    """Raised when the submission can't even be loaded far enough to check."""


# --------------------------------------------------------------------------
# Discovery: import the submission folder as a package, find BaseModel
# subclasses inside it.
# --------------------------------------------------------------------------

def load_base_model_class(backend_dir: Path):
    sys.path.insert(0, str(backend_dir))
    try:
        base_module = importlib.import_module("models.base_model")
    except ModuleNotFoundError as exc:
        raise SubmissionError(
            "Could not import models.base_model — run this script from the "
            "backend/ directory (or alongside it) with an intact models/ "
            f"package. Underlying error: {exc}"
        ) from exc
    base_cls = getattr(base_module, "BaseModel", None)
    if base_cls is None:
        raise SubmissionError("models/base_model.py does not define a BaseModel class.")
    return base_cls


def discover_model_classes(folder: Path, base_cls: type) -> list[type]:
    module_name = f"models.{folder.name}"
    try:
        pkg = importlib.import_module(module_name)
    except Exception as exc:  # surfaces any import-time error to the group
        raise SubmissionError(
            f"Could not import '{module_name}'. Your folder must be a valid "
            f"Python package (an __init__.py that imports without errors). "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    classes = [
        obj
        for _, obj in vars(pkg).items()
        if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls
    ]

    if not classes:
        # Fallback for single-algorithm folders whose __init__.py doesn't
        # re-export anything explicitly — try the conventional model.py.
        try:
            sub = importlib.import_module(f"{module_name}.model")
        except ModuleNotFoundError:
            sub = None
        if sub is not None:
            classes = [
                obj
                for _, obj in vars(sub).items()
                if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls
            ]

    if not classes:
        raise SubmissionError(
            f"No class subclassing BaseModel was found in '{module_name}'. "
            "Check that __init__.py re-exports your model class(es) "
            "(see Coding Standards Section 6)."
        )
    return classes


# --------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------

def load_custom_fixture(folder: Path, script_dir: Path):
    """Look for backend/fixtures/<group_folder>_fixture.py; return its
    make_fixture callable, or None if it doesn't exist yet."""
    fixture_path = script_dir / "fixtures" / f"{folder.name}_fixture.py"
    if not fixture_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"{folder.name}_fixture", fixture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "make_fixture", None)


def make_generic_fixture(model_type: str, seed: int = 42, n_samples: int = 120, n_features: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    X_test = rng.standard_normal((30, n_features))

    if model_type == "classifier":
        y = rng.integers(0, 3, n_samples)
    elif model_type == "regressor":
        y = rng.standard_normal(n_samples) * 10
    else:  # clusterer, dimensionality_reducer, or unknown
        y = None
    return X, y, X_test


# --------------------------------------------------------------------------
# Per-class validation
# --------------------------------------------------------------------------

def instantiate(cls: type):
    try:
        return cls()
    except Exception as exc:
        raise SubmissionError(
            f"{cls.__name__}() could not be instantiated with no arguments. "
            "Every constructor parameter needs a sensible default — the "
            f"integration backend constructs your model with none supplied. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc


def try_fit_and_detect_type(cls: type, fixture_fn, seed: int = 42) -> dict[str, Any]:
    """Fit the model against candidate fixtures until one succeeds, then
    trust the model's own post-fit get_metadata()['model_type'] as ground
    truth (matches the reference implementation, whose get_metadata() reads
    fitted attributes like self._model.coef_ and so cannot be called safely
    before fit — pre-fit type detection is not a valid strategy here).

    Attempt order: y=None first (correct call for clusterer/dimensionality
    reducer), then classifier-style y, then regressor-style y. Whichever
    fit() call doesn't raise wins; if a supervised model accepts y=None
    without raising, that's flagged separately as a missing input-validation
    defect (Coding Standards Section 10), not treated as "it's unsupervised".
    """
    if fixture_fn is not None:
        X, y, X_test = fixture_fn(seed=seed)
        attempts = [("custom group fixture", X, y, X_test)]
    else:
        Xu, _, Xtu = make_generic_fixture("clusterer", seed=seed)
        Xc, yc, Xtc = make_generic_fixture("classifier", seed=seed)
        Xr, yr, Xtr = make_generic_fixture("regressor", seed=seed)
        attempts = [
            ("y=None (unsupervised)", Xu, None, Xtu),
            ("classifier-style y", Xc, yc, Xtc),
            ("regressor-style y", Xr, yr, Xtr),
        ]

    last_exc: Exception | None = None
    for fixture_label, X, y, X_test in attempts:
        try:
            inst = instantiate(cls)
            t0 = time.perf_counter()
            ret = inst.fit(X, y)
            elapsed = time.perf_counter() - t0
            return {
                "ok": True, "instance": inst, "ret": ret, "X": X, "y": y, "X_test": X_test,
                "elapsed": elapsed, "fixture_label": fixture_label,
                "accepted_y_none": (y is None),
            }
        except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
            last_exc = exc
            continue
    return {"ok": False, "error": last_exc, "attempts": [a[0] for a in attempts]}


def validate_class(cls: type, folder: Path, fixture_fn, results: list[CheckResult]) -> None:
    label = cls.__name__

    fit_result = try_fit_and_detect_type(cls, fixture_fn)
    if not fit_result["ok"]:
        results.append(CheckResult(
            f"{label}: fit(X, y) succeeds on a matching fixture",
            False,
            f"fit() raised for every fixture tried ({', '.join(fit_result['attempts'])}). "
            f"Last error: {type(fit_result['error']).__name__}: {fit_result['error']}",
        ))
        return  # nothing further can be checked meaningfully

    inst1 = fit_result["instance"]
    ret = fit_result["ret"]
    X, y, X_test = fit_result["X"], fit_result["y"], fit_result["X_test"]
    elapsed = fit_result["elapsed"]

    results.append(CheckResult(
        f"{label}: fit() returns self",
        ret is inst1,
        "" if ret is inst1 else "fit() must `return self`.",
    ))
    results.append(CheckResult(
        f"{label}: fit() sets is_fitted = True",
        getattr(inst1, "is_fitted", False) is True,
        "" if getattr(inst1, "is_fitted", False) is True else
        "is_fitted must be True on the instance after fit() completes.",
    ))
    results.append(CheckResult(
        f"{label}: training completes within {TRAINING_TIME_CEILING_SECONDS}s (CPU ceiling)",
        elapsed <= TRAINING_TIME_CEILING_SECONDS,
        "" if elapsed <= TRAINING_TIME_CEILING_SECONDS else
        f"fit() took {elapsed:.1f}s on a {X.shape[0]}-row synthetic fixture, "
        f"over the {TRAINING_TIME_CEILING_SECONDS}s ceiling. Note: this fixture "
        "is small — a real failure here on toy data usually means an "
        "unbounded loop or missing max_iter, not a borderline timing issue.",
    ))

    # Ground truth for what shape/proba checks to apply — the model's own
    # post-fit metadata, not a guess from which fixture happened to work.
    try:
        pre_md = inst1.get_metadata()
        effective_type = pre_md.get("model_type") if isinstance(pre_md, dict) else None
    except Exception:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
        effective_type = None
    if effective_type not in VALID_MODEL_TYPES:
        effective_type = None

    if fit_result["accepted_y_none"] and effective_type in {"classifier", "regressor"}:
        results.append(CheckResult(
            f"{label}: rejects y=None for a supervised model_type",
            False,
            f"model_type='{effective_type}' but fit(X, y=None) completed without "
            "raising. Supervised models must validate that y is provided — "
            "raise a specific exception (e.g. ValueError) when it isn't, "
            "per Coding Standards Section 10.",
        ))

    # --- predict shape ---
    try:
        pred = inst1.predict(X_test)
        if effective_type == "dimensionality_reducer":
            shape_ok = pred.ndim == 2 and pred.shape[0] == X_test.shape[0]
            shape_msg = "predict() for a dimensionality reducer must return shape (n_samples, n_components)."
        else:
            shape_ok = pred.ndim == 1 and pred.shape[0] == X_test.shape[0]
            shape_msg = f"predict() must return a 1D array of shape ({X_test.shape[0]},); got shape {pred.shape}."
        results.append(CheckResult(f"{label}: predict() output shape", shape_ok, "" if shape_ok else shape_msg))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
        pred = None
        results.append(CheckResult(f"{label}: predict() output shape", False, f"predict() raised: {type(exc).__name__}: {exc}"))

    # --- predict_proba ---
    try:
        proba = inst1.predict_proba(X_test)
        if effective_type == "classifier":
            proba_ok = (
                proba is not None
                and proba.ndim == 2
                and proba.shape[0] == X_test.shape[0]
                and np.all(proba >= -1e-9)
                and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
            )
            proba_msg = (
                "Classifiers must return a 2D (n_samples, n_classes) array "
                "of non-negative probabilities whose rows sum to 1. Got "
                f"{'None' if proba is None else f'shape {proba.shape}, row sums {proba.sum(axis=1)[:3]}...'}"
            )
        else:
            proba_ok = proba is None
            proba_msg = f"Non-classifiers must return None from predict_proba(); got {type(proba)}."
        results.append(CheckResult(f"{label}: predict_proba() contract", proba_ok, "" if proba_ok else proba_msg))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
        results.append(CheckResult(f"{label}: predict_proba() contract", False, f"predict_proba() raised: {type(exc).__name__}: {exc}"))

    # --- determinism: independent instance, same fixture, same seed ---
    try:
        inst2 = instantiate(cls)
        inst2.fit(X, y)
        pred2 = inst2.predict(X_test)
        deterministic = pred is not None and np.array_equal(pred, pred2)
        results.append(CheckResult(
            f"{label}: determinism (two fit+predict cycles match)",
            deterministic,
            "" if deterministic else
            "Two independently fitted instances on identical input produced "
            "different predict() output. Set random_state=42 on every "
            "stochastic component (and, for PyTorch groups, torch.manual_seed(42) "
            "+ torch.use_deterministic_algorithms(True) at import time).",
        ))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
        results.append(CheckResult(f"{label}: determinism (two fit+predict cycles match)", False, f"Second fit/predict cycle raised: {type(exc).__name__}: {exc}"))

    # --- metadata (post-fit, exact keys) ---
    try:
        md = inst1.get_metadata()
        keys_ok = isinstance(md, dict) and set(md.keys()) == REQUIRED_METADATA_KEYS
        keys_msg = "" if keys_ok else (
            f"get_metadata() must return exactly these keys: "
            f"{sorted(REQUIRED_METADATA_KEYS)}. Got: {sorted(md.keys()) if isinstance(md, dict) else type(md)}"
        )
        results.append(CheckResult(f"{label}: get_metadata() exact keys", keys_ok, keys_msg))

        if keys_ok:
            type_ok = md["model_type"] in VALID_MODEL_TYPES
            results.append(CheckResult(
                f"{label}: model_type is valid",
                type_ok,
                "" if type_ok else f"model_type must be one of {sorted(VALID_MODEL_TYPES)}; got {md['model_type']!r}.",
            ))
            nfeat_ok = md["n_features"] == X.shape[1]
            results.append(CheckResult(
                f"{label}: n_features matches training data",
                nfeat_ok,
                "" if nfeat_ok else f"n_features={md['n_features']!r} but the model was fit on {X.shape[1]} features.",
            ))
            try:
                json.dumps(md["hyperparameters"])
                json.dumps(md["feature_importance"])
                json_ok = True
                json_msg = ""
            except (TypeError, ValueError) as exc:
                json_ok = False
                json_msg = f"hyperparameters/feature_importance must be JSON-serialisable (no numpy types, no callables): {exc}"
            results.append(CheckResult(f"{label}: hyperparameters/feature_importance are JSON-safe", json_ok, json_msg))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
        results.append(CheckResult(f"{label}: get_metadata() exact keys", False, f"get_metadata() raised post-fit: {type(exc).__name__}: {exc}"))

    # --- optional get_visualization_data ---
    if hasattr(cls, "get_visualization_data"):
        try:
            viz = inst1.get_visualization_data()
            viz_ok = viz is None or (json.dumps(viz) is not None)
            results.append(CheckResult(f"{label}: get_visualization_data() JSON-safe", viz_ok, "", fatal=False))
        except Exception as exc:  # noqa: BLE001 — deliberately broad: reports any failure in third-party submission code as a result line instead of crashing the validator
            results.append(CheckResult(
                f"{label}: get_visualization_data() JSON-safe", False,
                f"get_visualization_data() raised or returned non-JSON-safe data: {type(exc).__name__}: {exc}",
                fatal=False,
            ))


# --------------------------------------------------------------------------
# File-level checks: no print(), no hardcoded paths, no committed data files
# --------------------------------------------------------------------------

def check_source_hygiene(folder: Path, results: list[CheckResult]) -> None:
    py_files = [p for p in folder.rglob("*.py") if "__pycache__" not in p.parts]

    print_offenders: list[str] = []
    path_offenders: list[str] = []

    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    print_offenders.append(f"{path.relative_to(folder)}:{node.lineno}")
        except SyntaxError as exc:
            results.append(CheckResult(f"parseable Python: {path.name}", False, f"SyntaxError: {exc}"))
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in ABS_PATH_PATTERNS:
                if pattern.search(line):
                    path_offenders.append(f"{path.relative_to(folder)}:{lineno}")
                    break

    results.append(CheckResult(
        "no print() calls in submitted source",
        not print_offenders,
        "" if not print_offenders else f"print() found at: {', '.join(print_offenders)}",
    ))
    results.append(CheckResult(
        "no hardcoded absolute paths in submitted source",
        not path_offenders,
        "" if not path_offenders else f"Path-like literal found at: {', '.join(path_offenders)}",
    ))

    data_files = [
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.suffix.lower() in DISALLOWED_DATA_EXTENSIONS and "__pycache__" not in p.parts
    ]
    results.append(CheckResult(
        "no committed data files or notebooks",
        not data_files,
        "" if not data_files else f"Found: {', '.join(data_files)}. Generate synthetic data in tests instead (Section 12).",
    ))


# --------------------------------------------------------------------------
# Tooling: ruff, pytest + coverage
# --------------------------------------------------------------------------

def run_ruff(folder: Path, backend_dir: Path, results: list[CheckResult]) -> None:
    try:
        proc = subprocess.run(
            ["ruff", "check", str(folder)],
            cwd=backend_dir, capture_output=True, text=True, timeout=60, check=False,
        )
    except FileNotFoundError:
        results.append(CheckResult("ruff lint", False, "ruff is not installed/on PATH — install it and rerun (`pip install ruff`).", fatal=False))
        return
    ok = proc.returncode == 0
    results.append(CheckResult("ruff lint", ok, "" if ok else proc.stdout.strip()[-2000:] or proc.stderr.strip()[-2000:]))


def run_pytest_coverage(folder: Path, backend_dir: Path, results: list[CheckResult]) -> None:
    # Section 12/13 mandates the submission's test file be named exactly
    # `test.py` and run as `pytest test.py`. Pointing pytest at the folder
    # instead relies on its default discovery (test_*.py / *_test.py), which
    # never matches a bare `test.py` — that silently collects zero tests and
    # reports 0% coverage no matter how good the group's tests actually are.
    test_file = folder / "test.py"
    if not test_file.is_file():
        results.append(CheckResult(
            f"pytest with >={COVERAGE_THRESHOLD}% coverage", False,
            f"Expected {test_file} per Section 13 — no test.py found in the submission folder.",
        ))
        return
    # Section 12's example imports the model bare (`from model import ...`),
    # which only resolves with the submission folder itself as cwd (matching
    # the documented `cd` + `python -m pytest test.py` workflow). Several
    # existing submissions instead import fully-qualified
    # (`from models.group_XX.model import ...`), which only resolves with
    # backend/ on sys.path. Support both: run with cwd=folder (satisfies the
    # bare-import convention via python -m's own cwd insertion) and add
    # backend_dir to PYTHONPATH (satisfies the qualified-import convention),
    # rather than silently favoring one style over the other.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_dir) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", "test.py",
                "--cov=.", "--cov-report=term-missing",
                f"--cov-fail-under={COVERAGE_THRESHOLD}", "-q",
            ],
            cwd=folder, env=env, capture_output=True, text=True,
            timeout=TRAINING_TIME_CEILING_SECONDS + 60, check=False,
        )
    except FileNotFoundError:
        results.append(CheckResult(
            "pytest with >=80% coverage", False,
            "pytest/pytest-cov not installed — install them and rerun (`pip install pytest pytest-cov`).",
            fatal=False,
        ))
        return
    ok = proc.returncode == 0
    tail = (proc.stdout.strip() or proc.stderr.strip())[-3000:]
    results.append(CheckResult(f"pytest with >={COVERAGE_THRESHOLD}% coverage", ok, "" if ok else tail))


# --------------------------------------------------------------------------
# Report + main
# --------------------------------------------------------------------------

def print_report(results: list[CheckResult]) -> bool:
    all_fatal_ok = True
    width = max((len(r.name) for r in results), default=40)
    for r in results:
        status = "PASS" if r.ok else ("WARN" if not r.fatal else "FAIL")
        print(f"[{status:4}] {r.name.ljust(width)}")
        if not r.ok and r.detail:
            for line in r.detail.splitlines():
                print(f"         {line}")
        if not r.ok and r.fatal:
            all_fatal_ok = False
    print()
    print("ALL CHECKS PASSED" if all_fatal_ok else "VALIDATION FAILED — fix the FAIL items above before opening a pull request.")
    return all_fatal_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a model submission against the BaseModel interface contract.")
    parser.add_argument("folder", type=str, help="Path to the submission folder, e.g. models/group_05_knn_kmeans_gmm/")
    parser.add_argument("--skip-lint", action="store_true", help="Skip the ruff check (useful before ruff is installed locally).")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip the pytest/coverage check.")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"'{folder}' is not a directory.")
        return 2

    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir if folder.parent == script_dir / "models" else folder.parent.parent

    results: list[CheckResult] = []

    try:
        base_cls = load_base_model_class(backend_dir)
        classes = discover_model_classes(folder, base_cls)
    except SubmissionError as exc:
        print(f"[FAIL] could not load submission\n         {exc}")
        return 1

    fixture_fn = load_custom_fixture(folder, script_dir)

    for cls in classes:
        validate_class(cls, folder, fixture_fn, results)

    check_source_hygiene(folder, results)

    if not args.skip_lint:
        run_ruff(folder, backend_dir, results)
    if not args.skip_coverage:
        run_pytest_coverage(folder, backend_dir, results)

    ok = print_report(results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())