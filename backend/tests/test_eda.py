"""Stage 1: raw table validation, target resolution, data-type inference,
and EDA profiling (DATA_FLOW_GUIDE.md SS2, CODING_STANDARDS.md SS4)."""
import numpy as np
import pandas as pd
import pytest

from app.core.eda import (
    infer_data_type,
    profile_dataset,
    resolve_target_column,
    validate_raw_dataframe,
)
from app.core.errors import DataValidationError


def _numeric_df(n_rows=60, n_cols=3):
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"c{i}" for i in range(n_cols)])


class TestValidateRawDataframe:
    def test_accepts_valid_dataframe(self):
        validate_raw_dataframe(_numeric_df())  # does not raise

    def test_rejects_fewer_than_50_rows(self):
        with pytest.raises(DataValidationError):
            validate_raw_dataframe(_numeric_df(n_rows=49))

    def test_rejects_more_than_100_columns(self):
        with pytest.raises(DataValidationError):
            validate_raw_dataframe(_numeric_df(n_cols=101))

    def test_rejects_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x"] * 60, "b": ["y"] * 60})
        with pytest.raises(DataValidationError):
            validate_raw_dataframe(df)

    def test_accepts_exactly_50_rows_and_100_columns(self):
        validate_raw_dataframe(_numeric_df(n_rows=50, n_cols=100))  # does not raise


class TestResolveTargetColumn:
    def test_no_target_returns_none(self):
        df = _numeric_df()
        assert resolve_target_column(df, target_column=None, has_target=False) is None

    def test_named_target_used(self):
        df = _numeric_df()
        assert resolve_target_column(df, target_column="c1", has_target=True) == "c1"

    def test_defaults_to_last_column(self):
        df = _numeric_df()
        assert resolve_target_column(df, target_column=None, has_target=True) == "c2"

    def test_unknown_named_target_raises(self):
        df = _numeric_df()
        with pytest.raises(DataValidationError):
            resolve_target_column(df, target_column="nope", has_target=True)


class TestInferDataType:
    def test_no_target_is_clustering(self):
        assert infer_data_type(_numeric_df(), None) == "clustering"

    def test_two_to_ten_uniques_is_classification(self):
        df = pd.DataFrame({"y": [0, 1, 2] * 20})
        assert infer_data_type(df, "y") == "classification"

    def test_more_than_ten_uniques_is_regression(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"y": rng.standard_normal(60)})
        assert infer_data_type(df, "y") == "regression"

    def test_single_unique_value_raises(self):
        df = pd.DataFrame({"y": [1] * 60})
        with pytest.raises(DataValidationError):
            infer_data_type(df, "y")


class TestProfileDataset:
    def test_profile_shape_matches_dataframe(self):
        df = _numeric_df(n_rows=60, n_cols=3)
        profile = profile_dataset("id1", df, "clustering", None, "test-source")
        assert profile.n_rows == 60
        assert profile.n_columns == 3
        assert len(profile.columns) == 3
        assert profile.class_imbalance is None

    def test_classification_carries_imbalance_info(self):
        df = pd.DataFrame({"x": range(100), "y": ["neg"] * 95 + ["pos"] * 5})
        profile = profile_dataset("id2", df, "classification", "y", "test-source")
        assert profile.class_imbalance is not None
        assert profile.class_imbalance.is_imbalanced is True

    def test_missing_values_counted(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0] * 20})
        profile = profile_dataset("id3", df, "clustering", None, "test-source")
        col = profile.columns[0]
        assert col.missing_count == 20
        assert col.dtype == "numeric"

    def test_categorical_column_gets_categorical_distribution(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"] * 20})
        profile = profile_dataset("id4", df, "clustering", None, "test-source")
        col = profile.columns[0]
        assert col.dtype == "categorical"
        assert col.distribution.kind == "categorical"
        assert col.top_value == "a"

    def test_target_column_flagged_is_target(self):
        df = pd.DataFrame({"x": range(60), "y": [0, 1] * 30})
        profile = profile_dataset("id5", df, "classification", "y", "test-source")
        by_name = {c.name: c for c in profile.columns}
        assert by_name["y"].is_target is True
        assert by_name["x"].is_target is False
