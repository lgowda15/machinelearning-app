"""API integration tests for the registry and compatibility endpoints
(ARCHITECTURE.md SS6, BUILD_SESSIONS.md Session 3).

Done-when criterion (compatibility half): a clusterer requested on
labelled data comes back as incompatible with a reason, not an error.
"""
import pytest

from app.core import storage

EXPECTED_TYPES = {
    "logistic_regression": "classifier",
    "kmeans": "clusterer",
    "linear_regression": "regressor",
    "pca": "dimensionality_reducer",
}


@pytest.fixture(autouse=True)
def _clean_storage():
    storage.clear()
    yield
    storage.clear()


def _load_sample(client, sample_id: str) -> str:
    response = client.post(f"/api/data/samples/{sample_id}")
    assert response.status_code == 200
    return response.json()["data_id"]


class TestRegistry:
    def test_lists_all_four_reference_models(self, client):
        response = client.get("/api/models/registry")
        assert response.status_code == 200
        models = {m["key"]: m["model_type"] for m in response.json()["models"]}
        assert models == EXPECTED_TYPES


class TestCompatibility:
    def test_classification_data_only_classifier_and_reducer_compatible(self, client):
        data_id = _load_sample(client, "iris")
        response = client.post("/api/models/compatible", json={"data_id": data_id})
        assert response.status_code == 200
        body = response.json()
        compatible_keys = {m["key"] for m in body["compatible"]}
        incompatible = {m["key"]: m["reason"] for m in body["incompatible"]}

        assert compatible_keys == {"logistic_regression", "pca"}
        assert set(incompatible) == {"kmeans", "linear_regression"}
        assert "classification" in incompatible["kmeans"]

    def test_regression_data_only_regressor_and_reducer_compatible(self, client):
        data_id = _load_sample(client, "diabetes")
        response = client.post("/api/models/compatible", json={"data_id": data_id})
        body = response.json()
        compatible_keys = {m["key"] for m in body["compatible"]}
        assert compatible_keys == {"linear_regression", "pca"}

    def test_clustering_data_clusterer_and_reducer_compatible(self, client):
        data_id = _load_sample(client, "blobs")
        response = client.post("/api/models/compatible", json={"data_id": data_id})
        body = response.json()
        compatible_keys = {m["key"] for m in body["compatible"]}
        assert compatible_keys == {"kmeans", "pca"}

    def test_clusterer_on_labelled_data_is_incompatible_with_reason_not_an_error(self, client):
        data_id = _load_sample(client, "iris")
        response = client.post("/api/models/compatible", json={"data_id": data_id})
        assert response.status_code == 200  # not an error response
        incompatible = {m["key"]: m for m in response.json()["incompatible"]}
        assert "kmeans" in incompatible
        assert isinstance(incompatible["kmeans"]["reason"], str) and incompatible["kmeans"]["reason"]

    def test_unknown_data_id_returns_404(self, client):
        response = client.post("/api/models/compatible", json={"data_id": "does-not-exist"})
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
