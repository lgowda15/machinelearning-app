"""Stage 2: split -> impute -> encode -> scale -> cast float64
(DATA_FLOW_GUIDE.md SS3, .claude/rules/backend.md).
"""
import numpy as np
import pandas as pd
import pytest

from app.core.errors import DataValidationError
from app.core.preprocessing import (
    ONE_HOT_MAX_CARDINALITY,
    fit_transform_split,
    transform_new,
)


def _mixed_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num1": rng.standard_normal(n),
        "num2": rng.standard_normal(n) * 10 + 5,
        "low_card": rng.choice(["a", "b", "c"], size=n),
        "high_card": rng.choice([f"cat_{i}" for i in range(15)], size=n),
        "label": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 5, replace=False), "num1"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "low_card"] = np.nan
    return df


class TestClassificationSplit:
    def test_output_is_2d_float64_no_missing(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        assert result.X_train.dtype == np.float64
        assert result.X_test.dtype == np.float64
        assert result.X_train.ndim == 2
        assert not np.isnan(result.X_train).any()
        assert not np.isnan(result.X_test).any()

    def test_split_is_80_20_by_default(self):
        df = _mixed_df(n=200)
        result = fit_transform_split(df, "label", "classification")
        assert result.X_train.shape[0] == 160
        assert result.X_test.shape[0] == 40

    def test_stratified_split_preserves_class_proportions(self):
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "x": rng.standard_normal(n),
            "y": ["neg"] * 450 + ["pos"] * 50,
        })
        result = fit_transform_split(df, "y", "classification")
        train_pos_ratio = (result.y_train == "pos").mean()
        test_pos_ratio = (result.y_test == "pos").mean()
        assert abs(train_pos_ratio - test_pos_ratio) < 0.05

    def test_low_cardinality_is_one_hot_encoded(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        assert "low_card=a" in result.feature_names
        assert "low_card=b" in result.feature_names
        assert "low_card=c" in result.feature_names

    def test_high_cardinality_is_label_encoded(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        assert "high_card" in result.feature_names
        assert not any(f.startswith("high_card=") for f in result.feature_names)

    def test_cardinality_cutoff_is_ten(self):
        assert ONE_HOT_MAX_CARDINALITY == 10

    def test_scaler_only_touches_numeric_columns(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        num1_idx = result.feature_names.index("num1")
        assert abs(result.X_train[:, num1_idx].mean()) < 1e-6
        assert abs(result.X_train[:, num1_idx].std() - 1.0) < 1e-6
        # One-hot columns must remain pure 0/1, not standardized.
        low_a_idx = result.feature_names.index("low_card=a")
        assert set(np.unique(result.X_train[:, low_a_idx])) <= {0.0, 1.0}

    def test_y_train_y_test_are_1d(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        assert result.y_train.ndim == 1
        assert result.y_test.ndim == 1


class TestRegressionSplit:
    def test_not_stratified_continuous_target(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"x": rng.standard_normal(n), "y": rng.standard_normal(n)})
        result = fit_transform_split(df, "y", "regression")
        assert result.y_train.dtype.kind == "f"


class TestClusteringSplit:
    def test_y_is_none_when_no_target(self):
        df = _mixed_df().drop(columns=["label"])
        result = fit_transform_split(df, target_column=None, data_type="clustering")
        assert result.y_train is None
        assert result.y_test is None

    def test_all_columns_become_features(self):
        df = _mixed_df().drop(columns=["label"])
        result = fit_transform_split(df, target_column=None, data_type="clustering")
        assert result.X_train.shape[0] + result.X_test.shape[0] == len(df)


class TestTransformNew:
    def test_reuses_fitted_objects_without_refitting(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        new_df = df.drop(columns=["label"]).iloc[:20]
        X_new = transform_new(new_df, result.fitted)
        np.testing.assert_allclose(X_new, result.fitted.transform(new_df))
        assert X_new.shape == (20, len(result.feature_names))

    def test_unseen_high_cardinality_category_does_not_crash(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        new_df = df.drop(columns=["label"]).iloc[:5].copy()
        new_df.loc[new_df.index[0], "high_card"] = "never_seen_at_fit_time"
        X_new = transform_new(new_df, result.fitted)  # does not raise
        assert X_new.shape[0] == 5

    def test_missing_column_is_rejected_before_reaching_a_model(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        new_df = df.drop(columns=["label", "num2"]).iloc[:5]
        with pytest.raises(DataValidationError) as exc_info:
            transform_new(new_df, result.fitted)
        assert "num2" in str(exc_info.value.details)

    def test_extra_column_is_rejected(self):
        df = _mixed_df()
        result = fit_transform_split(df, "label", "classification")
        new_df = df.drop(columns=["label"]).iloc[:5].copy()
        new_df["surprise_column"] = 1
        with pytest.raises(DataValidationError):
            transform_new(new_df, result.fitted)


def test_no_columns_raises_explicit_error():
    df = pd.DataFrame({"only_target": [0, 1] * 30})
    with pytest.raises(DataValidationError):
        fit_transform_split(df, "only_target", "classification")


def test_determinism_same_seed_same_output():
    df = _mixed_df()
    a = fit_transform_split(df, "label", "classification")
    b = fit_transform_split(df, "label", "classification")
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.X_test, b.X_test)
