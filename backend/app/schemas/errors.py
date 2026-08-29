"""The single error shape used across the whole API (ARCHITECTURE.md SS6)."""
from typing import Any, Literal

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: Literal[True] = True
    code: str
    message: str
    details: dict[str, Any] = {}
