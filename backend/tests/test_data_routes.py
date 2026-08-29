"""API integration tests for the upload/EDA endpoints
(ARCHITECTURE.md SS6, BUILD_SESSIONS.md Session 2).

Done-when criterion: a real CSV and a sample dataset both return an EDA
payload with the imbalance flag correct on a deliberately skewed target.
"""
import io

import numpy as np
import pandas as pd
import pytest

from app.core import storage


@pytest.fixture(autouse=True)
def _clean_storage():
    storage.clear()
    yield
    storage.clear()


def _csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _skewed_df(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "cat": rng.choice(["x", "y", "z"], size=n),
        "label": ["pos"] * 5 + ["neg"] * (n - 5),
    })


class TestUpload:
    def test_real_csv_returns_eda_payload_with_correct_imbalance_flag(self, client):
        df = _skewed_df()
        response = client.post(
            "/api/data/upload",
            files={"file": ("skewed.csv", _csv_bytes(df), "text/csv")},
            data={"target_column": "label"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data_type"] == "classification"
        assert body["class_imbalance"]["is_imbalanced"] is True
        assert body["class_imbalance"]["message"] == "Class imbalance present; predictions may be biased."
        assert len(body["columns"]) == 4

    def test_defaults_target_to_last_column(self, client):
        df = pd.DataFrame({"a": range(60), "b": [0, 1] * 30})
        response = client.post(
            "/api/data/upload", files={"file": ("d.csv", _csv_bytes(df), "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["target_column"] == "b"

    def test_explicit_no_target_is_clustering(self, client):
        df = pd.DataFrame({"a": range(60), "b": range(60)})
        response = client.post(
            "/api/data/upload",
            files={"file": ("d.csv", _csv_bytes(df), "text/csv")},
            data={"has_target": "false"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data_type"] == "clustering"
        assert body["target_column"] is None

    def test_rejects_wrong_format(self, client):
        response = client.post(
            "/api/data/upload",
            files={"file": ("bad.csv", io.BytesIO(b"\x00\x01\x02garbage"), "text/csv")},
        )
        assert response.status_code == 400
        assert response.json()["error"] is True
        assert response.json()["code"] == "data_validation_error"

    def test_rejects_fewer_than_50_rows(self, client):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        response = client.post(
            "/api/data/upload", files={"file": ("small.csv", _csv_bytes(df), "text/csv")},
        )
        assert response.status_code == 400

    def test_rejects_more_than_100_columns(self, client):
        df = pd.DataFrame(np.zeros((60, 101)))
        response = client.post(
            "/api/data/upload", files={"file": ("wide.csv", _csv_bytes(df), "text/csv")},
        )
        assert response.status_code == 400

    def test_rejects_no_numeric_columns(self, client):
        df = pd.DataFrame({"a": ["x"] * 60, "b": ["y"] * 60})
        response = client.post(
            "/api/data/upload", files={"file": ("str.csv", _csv_bytes(df), "text/csv")},
        )
        assert response.status_code == 400

    def test_error_shape_matches_contract(self, client):
        df = pd.DataFrame({"a": range(10)})
        response = client.post(
            "/api/data/upload", files={"file": ("small.csv", _csv_bytes(df), "text/csv")},
        )
        body = response.json()
        assert set(body.keys()) == {"error", "code", "message", "details"}


class TestGetData:
    def test_round_trips_through_storage(self, client):
        df = _skewed_df()
        upload_response = client.post(
            "/api/data/upload",
            files={"file": ("skewed.csv", _csv_bytes(df), "text/csv")},
            data={"target_column": "label"},
        )
        data_id = upload_response.json()["data_id"]

        get_response = client.get(f"/api/data/{data_id}")
        assert get_response.status_code == 200
        assert get_response.json() == upload_response.json()

    def test_unknown_id_returns_404_with_error_shape(self, client):
        response = client.get("/api/data/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] is True
        assert body["code"] == "not_found"


class TestSamples:
    def test_lists_samples(self, client):
        response = client.get("/api/data/samples")
        assert response.status_code == 200
        ids = {s["id"] for s in response.json()["samples"]}
        assert ids == {"iris", "diabetes", "blobs"}

    def test_load_sample_returns_eda_payload(self, client):
        response = client.post("/api/data/samples/iris")
        assert response.status_code == 200
        body = response.json()
        assert body["data_type"] == "classification"
        assert body["target_column"] == "species"
        assert body["class_imbalance"]["is_imbalanced"] is False

    def test_loaded_sample_is_retrievable_by_id(self, client):
        load_response = client.post("/api/data/samples/diabetes")
        data_id = load_response.json()["data_id"]
        get_response = client.get(f"/api/data/{data_id}")
        assert get_response.status_code == 200
        assert get_response.json()["data_type"] == "regression"

    def test_clustering_sample_has_no_target(self, client):
        response = client.post("/api/data/samples/blobs")
        body = response.json()
        assert body["data_type"] == "clustering"
        assert body["target_column"] is None
        assert body["class_imbalance"] is None

    def test_unknown_sample_id_returns_404(self, client):
        response = client.post("/api/data/samples/does-not-exist")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
