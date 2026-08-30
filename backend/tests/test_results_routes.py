"""API integration tests for the results-comparison endpoint
(ARCHITECTURE.md SS6, BUILD_SESSIONS.md Session 4).

Comparison is cross-run by design: pairs may name the same training run
or two different ones.
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


def _train(client, data_id: str, model_key: str) -> str:
    body = {"data_id": data_id, "models": [{"model_key": model_key}]}
    response = client.post("/api/training/train", json=body)
    assert response.status_code == 200
    return response.json()["training_id"]


class TestComparison:
    def test_same_model_type_across_two_training_runs_shares_all_metrics(self, client):
        iris = _load_sample(client, "iris")
        training_id_1 = _train(client, iris, "logistic_regression")
        training_id_2 = _train(client, _load_sample(client, "iris"), "logistic_regression")

        response = client.post(
            "/api/results/comparison",
            json={
                "models": [
                    {"training_id": training_id_1, "model_key": "logistic_regression"},
                    {"training_id": training_id_2, "model_key": "logistic_regression"},
                ]
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body["common_metrics"]) == {
            "accuracy", "precision", "recall", "f1", "confusion_matrix", "labels",
        }
        assert [m["training_id"] for m in body["models"]] == [training_id_1, training_id_2]

    def test_different_model_types_compare_with_empty_common_metrics(self, client):
        iris = _load_sample(client, "iris")
        training_id_1 = _train(client, iris, "logistic_regression")
        training_id_2 = _train(client, iris, "pca")

        response = client.post(
            "/api/results/comparison",
            json={
                "models": [
                    {"training_id": training_id_1, "model_key": "logistic_regression"},
                    {"training_id": training_id_2, "model_key": "pca"},
                ]
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["common_metrics"] == []
        assert {m["model_type"] for m in body["models"]} == {"classifier", "dimensionality_reducer"}

    def test_unknown_training_id_returns_404(self, client):
        response = client.post(
            "/api/results/comparison",
            json={
                "models": [
                    {"training_id": "does-not-exist", "model_key": "logistic_regression"},
                    {"training_id": "also-does-not-exist", "model_key": "logistic_regression"},
                ]
            },
        )
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_fewer_than_two_models_is_rejected(self, client):
        training_id = _train(client, _load_sample(client, "iris"), "logistic_regression")
        response = client.post(
            "/api/results/comparison",
            json={"models": [{"training_id": training_id, "model_key": "logistic_regression"}]},
        )
        assert response.status_code == 422
