"""API integration tests for the prediction endpoint (ARCHITECTURE.md SS6,
BUILD_SESSIONS.md Session 4, DATA_FLOW_GUIDE.md SS7).

Done-when criterion: train -> predict on a fresh CSV works, and a
mismatched CSV fails with a message a user could act on.
"""
import io

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from app.core import storage, training_store


@pytest.fixture(autouse=True)
def _clean_stores():
    storage.clear()
    training_store.clear()
    yield
    storage.clear()
    training_store.clear()


def _csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _iris_features(n: int = 10) -> pd.DataFrame:
    """The four iris feature columns only, no target -- exactly what a
    trained model expects new prediction data to look like."""
    return load_iris(as_frame=True).frame.drop(columns=["target"]).head(n)


def _load_sample(client, sample_id: str) -> str:
    response = client.post(f"/api/data/samples/{sample_id}")
    assert response.status_code == 200
    return response.json()["data_id"]


def _train(client, data_id: str, model_key: str) -> str:
    body = {"data_id": data_id, "models": [{"model_key": model_key}]}
    response = client.post("/api/training/train", json=body)
    assert response.status_code == 200
    return response.json()["training_id"]


def _predict(client, training_id: str, model_key: str, df: pd.DataFrame):
    return client.post(
        "/api/prediction/predict",
        data={"training_id": training_id, "model_key": model_key},
        files={"file": ("new.csv", _csv_bytes(df), "text/csv")},
    )


class TestPredictClassifier:
    def test_train_then_predict_on_fresh_csv_returns_labels_and_probabilities(self, client):
        training_id = _train(client, _load_sample(client, "iris"), "logistic_regression")
        response = _predict(client, training_id, "logistic_regression", _iris_features())

        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "classifier"
        assert body["n_samples"] == 10
        assert len(body["predictions"]) == 10
        assert len(body["probabilities"]) == 10
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in body["probabilities"])


class TestPredictDimensionalityReducer:
    def test_predict_returns_2d_output_and_no_probabilities(self, client):
        training_id = _train(client, _load_sample(client, "iris"), "pca")
        response = _predict(client, training_id, "pca", _iris_features())

        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "dimensionality_reducer"
        assert body["probabilities"] is None
        assert len(body["predictions"]) == 10
        assert len(body["predictions"][0]) == 2  # default n_components


class TestPredictionValidation:
    def test_mismatched_columns_rejected_with_actionable_message(self, client):
        training_id = _train(client, _load_sample(client, "iris"), "logistic_regression")
        bad_df = _iris_features().drop(columns=["sepal length (cm)"])
        bad_df["unexpected column"] = 1

        response = _predict(client, training_id, "logistic_regression", bad_df)

        assert response.status_code == 400
        body = response.json()
        assert body["code"] == "data_validation_error"
        assert body["details"]["missing_columns"] == ["sepal length (cm)"]
        assert body["details"]["unexpected_columns"] == ["unexpected column"]

    def test_unknown_training_id_returns_404(self, client):
        response = _predict(client, "does-not-exist", "logistic_regression", _iris_features())
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_unknown_model_key_in_training_run_returns_404(self, client):
        training_id = _train(client, _load_sample(client, "iris"), "logistic_regression")
        response = _predict(client, training_id, "pca", _iris_features())
        assert response.status_code == 404
        body = response.json()
        assert body["code"] == "not_found"
        assert "pca" in body["message"]
