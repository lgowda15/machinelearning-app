"""Registry and compatibility endpoints (ARCHITECTURE.md SS6).

GET  /api/models/registry     -- list all registered models and metadata
POST /api/models/compatible   -- given a dataset, compatible/incompatible
                                  models with a reason per incompatible one
"""
from fastapi import APIRouter

from app.core import storage
from app.core.compatibility import incompatibility_reason
from app.core.registry import REGISTRY
from app.schemas.errors import ErrorResponse
from app.schemas.models import (
    CompatibilityRequest,
    CompatibilityResponse,
    IncompatibleModelSummary,
    ModelSummary,
    RegistryResponse,
)

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)


def _summarise(key: str) -> tuple[ModelSummary, str]:
    """A fresh instance's metadata and its own model_type -- same pattern
    test_conformance.py uses to parametrise over the registry."""
    instance = REGISTRY[key]()
    metadata = instance.get_metadata()
    summary = ModelSummary(
        key=key,
        model_name=metadata["model_name"],
        model_type=metadata["model_type"],
        default_hyperparameters=instance.hyperparams,
    )
    return summary, metadata["model_type"]


@router.get("/registry", response_model=RegistryResponse)
def get_registry() -> RegistryResponse:
    return RegistryResponse(models=[_summarise(key)[0] for key in REGISTRY])


@router.post("/compatible", response_model=CompatibilityResponse)
def get_compatible(request: CompatibilityRequest) -> CompatibilityResponse:
    record = storage.get_dataset(request.data_id)

    compatible: list[ModelSummary] = []
    incompatible: list[IncompatibleModelSummary] = []
    for key in REGISTRY:
        summary, model_type = _summarise(key)
        reason = incompatibility_reason(model_type, record.data_type)
        if reason is None:
            compatible.append(summary)
        else:
            incompatible.append(IncompatibleModelSummary(**summary.model_dump(), reason=reason))

    return CompatibilityResponse(
        data_type=record.data_type, compatible=compatible, incompatible=incompatible,
    )
