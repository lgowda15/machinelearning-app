import numpy as np
import pytest

from models.group_08_hmm_naive_bayes.naive_bayes import NaiveBayesModel


def make_fixture(seed=42, n_samples=100, n_features=4, n_test=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    y = rng.integers(0, 2, n_samples)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float64)
    return X, y, X_test


class TestNaiveBayesModelFitPredict:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_fit_returns_self_and_sets_is_fitted(self):
        m = NaiveBayesModel()
        result = m.fit(self.X, self.y)
        assert result is m
        assert m.is_fitted is True

    def test_predict_shape(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        preds = m.predict(self.X_test)
        assert preds.shape == (self.X_test.shape[0],)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            NaiveBayesModel().predict(self.X_test)

    def test_predict_proba_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            NaiveBayesModel().predict_proba(self.X_test)

    def test_proba_rows_sum_to_one(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        proba = m.predict_proba(self.X_test)
        assert proba is not None
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_proba_shape_matches_classes(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        proba = m.predict_proba(self.X_test)
        assert proba.shape == (self.X_test.shape[0], len(m.classes_))

    def test_determinism(self):
        a = NaiveBayesModel().fit(self.X, self.y).predict(self.X_test)
        b = NaiveBayesModel().fit(self.X, self.y).predict(self.X_test)
        np.testing.assert_array_equal(a, b)

    def test_predict_dtype_matches_training_labels(self):
        y_str_like = np.where(self.y == 0, 3, 7)  # non-0/1 label set
        m = NaiveBayesModel().fit(self.X, y_str_like)
        preds = m.predict(self.X_test)
        assert set(np.unique(preds)).issubset({3, 7})

    def test_classes_attribute_set(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        np.testing.assert_array_equal(m.classes_, np.array([0, 1]))


class TestNaiveBayesModelMetadata:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_metadata_keys(self):
        md = NaiveBayesModel().fit(self.X, self.y).get_metadata()
        expected = {
            "model_name", "model_type", "hyperparameters",
            "training_time_seconds", "n_features", "feature_importance",
        }
        assert set(md.keys()) == expected

    def test_metadata_model_type_and_name(self):
        md = NaiveBayesModel().fit(self.X, self.y).get_metadata()
        assert md["model_type"] == "classifier"
        assert md["model_name"] == "Gaussian Naive Bayes"
        assert md["n_features"] == self.X.shape[1]

    def test_feature_importance_has_one_entry_per_feature(self):
        md = NaiveBayesModel().fit(self.X, self.y).get_metadata()
        assert len(md["feature_importance"]) == self.X.shape[1]
        for v in md["feature_importance"].values():
            assert v >= 0

    def test_training_time_recorded(self):
        md = NaiveBayesModel().fit(self.X, self.y).get_metadata()
        assert md["training_time_seconds"] is not None
        assert md["training_time_seconds"] >= 0

    def test_random_state_not_forwarded_to_gaussian_nb(self):
        # random_state has no effect on GaussianNB (no stochastic
        # component) -- confirm it's stored but not part of the sklearn
        # estimator's own params.
        m = NaiveBayesModel(random_state=123)
        assert "random_state" not in m._model.get_params()


class TestNaiveBayesModelValidation:
    def setup_method(self):
        self.X, self.y, self.X_test = make_fixture()

    def test_fit_with_y_none_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X, None)

    def test_fit_with_non_ndarray_X_raises(self):
        with pytest.raises(TypeError):
            NaiveBayesModel().fit(self.X.tolist(), self.y)

    def test_fit_with_non_ndarray_y_raises(self):
        with pytest.raises(TypeError):
            NaiveBayesModel().fit(self.X, self.y.tolist())

    def test_fit_with_wrong_ndim_X_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X.reshape(-1), self.y)

    def test_fit_with_wrong_dtype_X_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X.astype(np.float32), self.y)

    def test_fit_with_non_finite_X_raises(self):
        X_bad = self.X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(X_bad, self.y)

    def test_fit_with_mismatched_rows_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X, self.y[:-1])

    def test_fit_with_single_class_raises(self):
        y_single = np.zeros_like(self.y)
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X, y_single)

    def test_fit_with_wrong_y_ndim_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel().fit(self.X, self.y.reshape(-1, 1))

    def test_predict_wrong_feature_count_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict(self.X_test[:, :-1])

    def test_predict_proba_wrong_feature_count_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict_proba(self.X_test[:, :-1])

    def test_predict_non_ndarray_X_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(TypeError):
            m.predict(self.X_test.tolist())

    def test_predict_wrong_ndim_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict(self.X_test.reshape(-1))

    def test_predict_wrong_dtype_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(ValueError):
            m.predict(self.X_test.astype(np.float32))

    def test_predict_proba_non_ndarray_X_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        with pytest.raises(TypeError):
            m.predict_proba(self.X_test.tolist())

    def test_predict_proba_non_finite_X_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        X_bad = self.X_test.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            m.predict_proba(X_bad)

    def test_predict_non_finite_X_raises(self):
        m = NaiveBayesModel().fit(self.X, self.y)
        X_bad = self.X_test.copy()
        X_bad[0, 0] = np.inf
        with pytest.raises(ValueError):
            m.predict(X_bad)

    def test_negative_var_smoothing_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesModel(var_smoothing=-1.0)
