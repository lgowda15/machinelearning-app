"""ARCHITECTURE.md SS8: flag any class below 20% or above 80% of samples."""
import pandas as pd

from app.core.imbalance import detect_imbalance


def test_balanced_two_class_not_flagged():
    y = pd.Series(["a"] * 50 + ["b"] * 50)
    result = detect_imbalance(y)
    assert result.is_imbalanced is False
    assert result.message is None
    assert result.class_counts == {"a": 50, "b": 50}


def test_skewed_two_class_flagged():
    y = pd.Series(["neg"] * 95 + ["pos"] * 5)
    result = detect_imbalance(y)
    assert result.is_imbalanced is True
    assert result.message == "Class imbalance present; predictions may be biased."
    assert result.class_counts == {"neg": 95, "pos": 5}


def test_boundary_exactly_20_percent_not_flagged():
    # Exactly at the threshold: not below 20%, not above 80%.
    y = pd.Series(["a"] * 80 + ["b"] * 20)
    result = detect_imbalance(y)
    assert result.is_imbalanced is False


def test_just_under_20_percent_flagged():
    y = pd.Series(["a"] * 81 + ["b"] * 19)
    result = detect_imbalance(y)
    assert result.is_imbalanced is True


def test_multiclass_one_minority_flags_whole_dataset():
    y = pd.Series(["a"] * 45 + ["b"] * 45 + ["c"] * 10)
    result = detect_imbalance(y)
    assert result.is_imbalanced is True


def test_missing_values_excluded_from_counts():
    y = pd.Series(["a"] * 50 + ["b"] * 50 + [None] * 10)
    result = detect_imbalance(y)
    assert sum(result.class_counts.values()) == 100
    assert result.is_imbalanced is False
