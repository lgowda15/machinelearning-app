"""ARCHITECTURE.md SS6: the single error shape."""
from app.core.errors import AppError, DataValidationError, NotFoundError


def test_app_error_to_dict_shape():
    err = AppError("something broke", details={"x": 1})
    assert err.to_dict() == {
        "error": True,
        "code": "app_error",
        "message": "something broke",
        "details": {"x": 1},
    }


def test_data_validation_error_defaults():
    err = DataValidationError("bad data")
    assert err.status_code == 400
    assert err.code == "data_validation_error"
    assert err.to_dict()["details"] == {}


def test_not_found_error_defaults():
    err = NotFoundError("missing")
    assert err.status_code == 404
    assert err.code == "not_found"
