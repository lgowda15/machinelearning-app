"""API integration tests for the training endpoints (ARCHITECTURE.md SS6,
BUILD_SESSIONS.md Session 3).

Done-when criterion: all four reference models train through the API on
an appropriate dataset and return the right metric set for their type. A
clusterer requested on labelled data comes back as incompatible with a
reason, not an error.
"""
import pytest

from app.core import storage, training_store


@pytest.fixture(autouse=True)
def _clean_stores():
    storage.clear()
    training_store.clear()
    yield
    storage.clear()
    training_store.clear()


def _load_sample(client, sample_id: str) -> str:
    response = client.post(f"/api/data/samples/{sample_id}")
    assert response.status_code == 200
    return response.json()["data_id"]


def _train(client, data_id, model_key, hyperparameters=None, test_size=None):
    body = {"data_id": data_id, "models": [{"model_key": model_key, "hyperparameters": hyperparameters or {}}]}
    if test_size is not None:
        body["test_size"] = test_size
    return client.post("/api/training/train", json=body)


class TestTrainClassifier:
    def test_logistic_regression_on_iris_returns_classifier_metrics(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "logistic_regression")
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["model_type"] == "classifier"
        assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1", "confusion_matrix", "labels"}
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
        # Confusion matrix chart is fed entirely by `metrics` -- no plot_data needed.
        assert result["plot_data"] is None


class TestTrainClusterer:
    def test_kmeans_on_blobs_returns_clusterer_metrics(self, client):
        data_id = _load_sample(client, "blobs")
        response = _train(client, data_id, "kmeans")
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["model_type"] == "clusterer"
        assert set(result["metrics"]) == {"silhouette_score", "davies_bouldin_score", "inertia"}
        # blobs is three well-separated synthetic clusters and kmeans
        # defaults to n_clusters=3, so these are definable, not None.
        assert result["metrics"]["silhouette_score"] > 0.5
        # blobs ships 4 features, so this exercises the real PCA-projection
        # path (app.core.metrics._cluster_scatter_points), not the
        # degenerate <=2-feature fallback.
        plot_data = result["plot_data"]
        n_test_rows = round(200 * 0.2)
        assert len(plot_data["points"]) == n_test_rows
        assert all(len(point) == 2 for point in plot_data["points"])
        assert len(plot_data["labels"]) == n_test_rows

    def test_hyperparameter_override_is_echoed_back(self, client):
        data_id = _load_sample(client, "blobs")
        response = _train(client, data_id, "kmeans", hyperparameters={"n_clusters": 2})
        result = response.json()["results"][0]
        assert result["hyperparameters"]["n_clusters"] == 2

    def test_clusterer_on_labelled_data_is_rejected_before_fitting_not_a_500(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "kmeans")
        assert response.status_code == 400
        body = response.json()
        assert body["error"] is True
        assert body["code"] == "data_validation_error"
        assert "kmeans" in body["message"]


class TestTrainRegressor:
    def test_regression_on_diabetes_returns_regressor_metrics(self, client):
        data_id = _load_sample(client, "diabetes")
        response = _train(client, data_id, "regression")
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["model_type"] == "regressor"
        assert set(result["metrics"]) == {"mse", "rmse", "r2", "mae"}
        plot_data = result["plot_data"]
        assert len(plot_data["y_true"]) == len(plot_data["y_pred"])
        assert len(plot_data["y_true"]) > 0


class TestTrainDimensionalityReducer:
    def test_pca_on_iris_returns_2d_predict_and_explained_variance(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "pca")
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["model_type"] == "dimensionality_reducer"
        assert len(result["metrics"]["explained_variance_ratio"]) == 2  # default n_components
        # PCA's own get_visualization_data() is present too.
        assert "explained_variance_ratio" in result["visualization_data"]
        # Variance plot chart is fed entirely by `metrics` -- no plot_data needed.
        assert result["plot_data"] is None

    def test_pca_compatible_with_labelled_and_unlabelled_data(self, client):
        for sample_id in ("iris", "diabetes", "blobs"):
            data_id = _load_sample(client, sample_id)
            response = _train(client, data_id, "pca")
            assert response.status_code == 200, sample_id


class TestTrainRequestValidation:
    def test_unknown_model_key_returns_404(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "does_not_exist")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_unknown_data_id_returns_404(self, client):
        response = _train(client, "does-not-exist", "logistic_regression")
        assert response.status_code == 404

    def test_bad_hyperparameter_fails_cleanly_not_a_500(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "logistic_regression", hyperparameters={"C": -1})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] is True
        assert body["code"] == "data_validation_error"

    def test_custom_test_size_is_reflected_in_response(self, client):
        data_id = _load_sample(client, "iris")
        response = _train(client, data_id, "logistic_regression", test_size=0.3)
        assert response.json()["test_size"] == 0.3

    def test_multiple_models_train_in_one_request(self, client):
        data_id = _load_sample(client, "iris")
        body = {
            "data_id": data_id,
            "models": [
                {"model_key": "logistic_regression"},
                {"model_key": "pca"},
            ],
        }
        response = client.post("/api/training/train", json=body)
        assert response.status_code == 200
        keys = {r["model_key"] for r in response.json()["results"]}
        assert keys == {"logistic_regression", "pca"}


class TestGetResults:
    def test_round_trips_through_training_store(self, client):
        data_id = _load_sample(client, "iris")
        train_response = _train(client, data_id, "logistic_regression")
        training_id = train_response.json()["training_id"]

        get_response = client.get(f"/api/training/{training_id}/results")
        assert get_response.status_code == 200
        assert get_response.json() == train_response.json()

    def test_unknown_training_id_returns_404(self, client):
        response = client.get("/api/training/does-not-exist/results")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
