"""Sample datasets ship scikit-learn built-ins covering all three Stage-1
data types (BUILD_SESSIONS.md Session 2)."""
import pytest

from app.core.errors import NotFoundError
from app.core.samples import get_sample, list_samples


def test_lists_all_three_data_types():
    data_types = {s.data_type for s in list_samples()}
    assert data_types == {"classification", "regression", "clustering"}


def test_unknown_sample_raises_not_found():
    with pytest.raises(NotFoundError):
        get_sample("does-not-exist")


def test_iris_loads_as_classification_with_target():
    sample = get_sample("iris")
    df, target_column = sample.loader()
    assert target_column == "species"
    assert df[target_column].nunique() == 3
    assert len(df) >= 50


def test_diabetes_loads_as_regression_with_continuous_target():
    sample = get_sample("diabetes")
    df, target_column = sample.loader()
    assert target_column == "target"
    assert df[target_column].nunique() > 10


def test_blobs_loads_with_no_target():
    sample = get_sample("blobs")
    df, target_column = sample.loader()
    assert target_column is None
    assert len(df) >= 50


def test_every_sample_satisfies_upload_row_column_minimums():
    from app.core.eda import validate_raw_dataframe

    for sample in list_samples():
        df, _ = sample.loader()
        validate_raw_dataframe(df)  # does not raise
