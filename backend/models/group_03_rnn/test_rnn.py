import numpy as np
import pytest

from models.group_03_rnn.rnn import RNNModel


def make_fixture(seed=42, n_samples=160, n_features=6):
    rng = np.random.default_rng(seed)
    lookback = RNNModel.EXPECTED_LOOKBACK
    assert n_features % lookback == 0

    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    X_seq = X.reshape(n_samples, lookback, n_features // lookback)

    # Deterministic synthetic regression target with sequential dependency.
    y = (
        0.7 * X_seq[:, 0, 0]
        - 0.4 * X_seq[:, 0, 1]
        + 1.1 * X_seq[:, 1, 0]
        - 0.2 * X_seq[:, 1, 1]
        + 1.4 * X_seq[:, 2, 0]
        + 0.3 * np.tanh(X_seq[:, 0, 0] + X_seq[:, 1, 1])
    ).astype(float)
    y += 0.01 * rng.standard_normal(n_samples)

    X_test = rng.standard_normal((40, n_features)).astype(np.float64)
    return X, y, X_test


def test_fit_returns_self():
    X, y, _ = make_fixture()
    m = RNNModel(max_epochs=80, patience=8)
    assert m.fit(X, y) is m


def test_predict_before_fit_raises():
    _, _, X_test = make_fixture()
    m = RNNModel()
    with pytest.raises(RuntimeError):
        m.predict(X_test)


def test_predict_shape_and_proba_none():
    X, y, X_test = make_fixture()
    m = RNNModel(max_epochs=80, patience=8).fit(X, y)
    pred = m.predict(X_test)
    assert pred.shape == (40,)
    assert np.issubdtype(pred.dtype, np.floating)
    # Regressor: predict_proba must be None
    assert m.predict_proba(X_test) is None


def test_exactly_six_features_is_accepted():
    X, y, X_test = make_fixture(n_features=6)
    m = RNNModel(max_epochs=30, patience=5).fit(X, y)
    pred = m.predict(X_test)
    assert pred.shape == (40,)


def test_model_actually_trains_and_predictions_are_not_identical():
    X, y, X_test = make_fixture()
    m = RNNModel(max_epochs=120, patience=12).fit(X, y)

    train_pred = m.predict(X)
    mse_model = float(np.mean((train_pred - y) ** 2))
    mse_baseline = float(np.mean((y.mean() - y) ** 2))

    assert mse_model < mse_baseline
    test_pred = m.predict(X_test)
    assert np.std(test_pred) > 1e-8


def test_feature_count_mismatch():
    X, y, _ = make_fixture()
    m = RNNModel(max_epochs=40, patience=6).fit(X, y)
    bad = np.zeros((5, m.n_features + 1), dtype=np.float64)
    with pytest.raises(ValueError):
        m.predict(bad)


def test_non_six_feature_counts_are_rejected():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(20).astype(float)
    for n_features in [5, 7, 9]:
        X_bad = rng.standard_normal((20, n_features)).astype(np.float64)
        with pytest.raises(ValueError, match="exactly 6 features"):
            RNNModel().fit(X_bad, y)


def test_reshape_layout_mapping():
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float64,
    )
    m = RNNModel()
    X_seq = m._reshape_input(X)
    expected = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(X_seq, expected)


def test_invalid_lookback_config_raises():
    with pytest.raises(ValueError, match="lookback must be 3"):
        RNNModel(lookback=4)


def test_invalid_dtype_rejected():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    y = rng.standard_normal(20).astype(float)
    with pytest.raises(ValueError):
        RNNModel().fit(X, y)


def test_non_finite_values_rejected():
    X, y, _ = make_fixture()
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RNNModel().fit(X_nan, y)

    y_inf = y.copy()
    y_inf[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        RNNModel().fit(X, y_inf)


def test_metadata_keys_and_type():
    X, y, _ = make_fixture()
    m = RNNModel(max_epochs=50, patience=6).fit(X, y)
    md = m.get_metadata()
    assert set(md.keys()) == {
        "model_name",
        "model_type",
        "hyperparameters",
        "training_time_seconds",
        "n_features",
        "feature_importance",
    }
    assert md["model_type"] == "regressor"
    assert md["training_time_seconds"] is not None
    assert md["training_time_seconds"] < 300
    assert md["feature_importance"] is None


def test_invalid_input_shape_handling_in_fit_and_predict():
    rng = np.random.default_rng(42)
    X_bad = rng.standard_normal(100).astype(np.float64)
    y = rng.standard_normal(100)
    m = RNNModel()
    with pytest.raises(ValueError):
        m.fit(X_bad, y)

    X, y, _ = make_fixture()
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.ones(10, dtype=np.float64))
    with pytest.raises(ValueError):
        m.predict(np.ones((10, 6), dtype=np.float32))
    X_non_finite = np.ones((10, 6), dtype=np.float64)
    X_non_finite[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        m.predict(X_non_finite)


def test_deterministic_training_predictions():
    X, y, X_test = make_fixture()
    a = RNNModel(max_epochs=80, patience=8).fit(X, y).predict(X_test)
    b = RNNModel(max_epochs=80, patience=8).fit(X, y).predict(X_test)
    assert np.array_equal(a, b)
