import numpy as np
import pytest

from models.group_08_hmm_naive_bayes.hmm import HMMModel


def make_fixture(seed=42, n_samples=120, n_test=30):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, HMMModel.EXPECTED_N_FEATURES)).astype(np.float64)
    y = rng.integers(0, 2, n_samples)
    X_test = rng.standard_normal((n_test, HMMModel.EXPECTED_N_FEATURES)).astype(np.float64)
    return X, y, X_test


class TestHMMModelFitPredict:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_fit_returns_self_and_sets_is_fitted(self):
        m = HMMModel()
        result = m.fit(self.X, self.y)
        assert result is m
        assert m.is_fitted is True

    def test_predict_shape(self):
        m = HMMModel().fit(self.X, self.y)
        preds = m.predict(self.X_test)
        assert preds.shape == (self.X_test.shape[0],)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            HMMModel().predict(self.X_test)

    def test_predict_proba_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            HMMModel().predict_proba(self.X_test)

    def test_proba_rows_sum_to_one(self):
        m = HMMModel().fit(self.X, self.y)
        proba = m.predict_proba(self.X_test)
        assert proba is not None
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_proba_shape_matches_classes(self):
        m = HMMModel().fit(self.X, self.y)
        proba = m.predict_proba(self.X_test)
        assert proba.shape == (self.X_test.shape[0], len(m.classes_))

    def test_determinism(self):
        a = HMMModel().fit(self.X, self.y).predict(self.X_test)
        b = HMMModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_predict_labels_come_from_training_classes(self):
        y_labels = np.where(self.y == 0, 5, 9)  # non-0/1 label set
        m = HMMModel().fit(self.X, y_labels)
        preds = m.predict(self.X_test)
        assert set(np.unique(preds)).issubset({5, 9})

    def test_classes_attribute_set(self):
        m = HMMModel().fit(self.X, self.y)
        np.testing.assert_array_equal(m.classes_, np.array([0, 1]))


class TestHMMModelMetadata:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_metadata_keys(self):
        md = HMMModel().fit(self.X, self.y).get_metadata()
        expected = {
            "model_name", "model_type", "hyperparameters",
            "training_time_seconds", "n_features", "feature_importance",
        }
        assert set(md.keys()) == expected

    def test_metadata_model_type_and_name(self):
        md = HMMModel().fit(self.X, self.y).get_metadata()
        assert md["model_type"] == "classifier"
        assert md["model_name"] == "Hidden Markov Model Classifier"
        assert md["n_features"] == HMMModel.EXPECTED_N_FEATURES

    def test_feature_importance_is_none(self):
        md = HMMModel().fit(self.X, self.y).get_metadata()
        assert md["feature_importance"] is None

    def test_training_time_recorded(self):
        md = HMMModel().fit(self.X, self.y).get_metadata()
        assert md["training_time_seconds"] is not None
        assert md["training_time_seconds"] >= 0


class TestHMMModelConstructorValidation:
    def test_wrong_lookback_raises(self):
        with pytest.raises(ValueError):
            HMMModel(lookback=3)

    def test_invalid_n_components_raises(self):
        with pytest.raises(ValueError):
            HMMModel(n_components=0)

    def test_invalid_n_iter_raises(self):
        with pytest.raises(ValueError):
            HMMModel(n_iter=0)

    def test_invalid_covariance_type_raises(self):
        with pytest.raises(ValueError):
            HMMModel(covariance_type="bogus")


class TestHMMModelFitValidation:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_fit_with_y_none_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X, None)

    def test_fit_with_non_ndarray_X_raises(self):
        with pytest.raises(TypeError):
            HMMModel().fit(self.X.tolist(), self.y)

    def test_fit_with_non_ndarray_y_raises(self):
        with pytest.raises(TypeError):
            HMMModel().fit(self.X, self.y.tolist())

    def test_fit_with_wrong_y_ndim_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X, self.y.reshape(-1, 1))

    def test_fit_with_non_2d_X_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X.reshape(-1), self.y)

    def test_fit_with_wrong_dtype_X_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X.astype(np.float32), self.y)

    def test_fit_with_non_finite_X_raises(self):
        X_bad = self.X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            HMMModel().fit(X_bad, self.y)

    def test_fit_with_wrong_column_count_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X[:, :-1], self.y)

    def test_fit_with_mismatched_rows_raises(self):
        with pytest.raises(ValueError):
            HMMModel().fit(self.X, self.y[:-1])

    def test_fit_with_single_class_raises(self):
        y_single = np.zeros_like(self.y)
        with pytest.raises(ValueError):
            HMMModel().fit(self.X, y_single)

    def test_fit_with_too_few_sequences_for_a_class_raises(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, HMMModel.EXPECTED_N_FEATURES)).astype(np.float64)
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # class 1 has 2 rows
        with pytest.raises(ValueError):
            HMMModel(n_components=3).fit(X, y)  # needs >= 3 per class

    def test_predict_wrong_feature_count_raises(self):
        m = HMMModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict(self.X_test[:, :-1])

    def test_predict_proba_wrong_feature_count_raises(self):
        m = HMMModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict_proba(self.X_test[:, :-1])


class TestHMMModelReshape:
    def test_reshape_produces_expected_shape(self):
        m = HMMModel()
        X = np.zeros((5, HMMModel.EXPECTED_N_FEATURES), dtype=np.float64)
        X_seq = m._reshape_input(X)
        assert X_seq.shape == (
            5, HMMModel.EXPECTED_LOOKBACK, HMMModel.EXPECTED_FEATURES_PER_STEP,
        )

    def test_reshape_preserves_column_order(self):
        m = HMMModel()
        row = np.arange(HMMModel.EXPECTED_N_FEATURES, dtype=np.float64)
        X_seq = m._reshape_input(row.reshape(1, -1))
        # [t0_f0, t0_f1, t1_f0, t1_f1, ...] -> timestep 1, feature 0 is
        # column index 2.
        assert X_seq[0, 1, 0] == 2.0
        assert X_seq[0, 0, 1] == 1.0
