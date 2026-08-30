"""Application error types.

Every error the API returns uses one shape (ARCHITECTURE.md SS6):

    {"error": true, "code": "...", "message": "...", "details": {...}}

Routes raise a specific AppError subclass naming what went wrong; the
FastAPI exception handlers registered in app.main translate it into that
shape with the matching HTTP status. Never return None or an empty
result to signal failure (CLAUDE.md "Hard rules").
"""
from typing import Any


class AppError(Exception):
    """Base for application errors that map onto the single error shape."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class DataValidationError(AppError):
    """The uploaded or requested data fails a sanity check (SS4 rejects,
    Stage 2 column mismatches, etc.)."""

    status_code = 400
    code = "data_validation_error"


class NotFoundError(AppError):
    """A referenced resource (data_id, sample id, ...) does not exist."""

    status_code = 404
    code = "not_found"
