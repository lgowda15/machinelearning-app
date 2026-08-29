"""The OpenAPI schema is what Session 5's typed frontend client generates
from -- confirm the error shape and data endpoints are actually documented,
not just implemented."""


def test_error_response_schema_is_documented(client):
    schema = client.get("/openapi.json").json()
    assert "ErrorResponse" in schema["components"]["schemas"]
    error_props = schema["components"]["schemas"]["ErrorResponse"]["properties"]
    assert set(error_props.keys()) == {"error", "code", "message", "details"}


def test_data_endpoints_are_documented(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/data/upload" in paths
    assert "/api/data/{data_id}" in paths
    assert "/api/data/samples" in paths
    assert "/api/data/samples/{sample_id}" in paths
