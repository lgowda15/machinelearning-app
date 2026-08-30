"""Model comparison endpoint (ARCHITECTURE.md SS6, BUILD_SESSIONS.md
Session 4).

POST /api/results/comparison -- side-by-side metrics for trained models,
named by {training_id, model_key} pairs. The pairs may point at the same
training run or different ones, so models trained separately can still be
compared without retraining them together in one /api/training/train call.
"""
from fastapi import APIRouter

from app.core import training_store
from app.core.errors import NotFoundError
from app.schemas.errors import ErrorResponse
from app.schemas.results import (
    ComparisonEntry,
    ComparisonModelRef,
    ComparisonRequest,
    ComparisonResponse,
)

router = APIRouter(
    prefix="/api/results",
    tags=["results"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)


def _entry(ref: ComparisonModelRef) -> ComparisonEntry:
    record = training_store.get_training(ref.training_id)
    for result in record.results:
        if result.model_key == ref.model_key:
            return ComparisonEntry(
                training_id=ref.training_id,
                model_key=ref.model_key,
                model_name=result.metadata["model_name"],
                model_type=result.metadata["model_type"],
                metrics=result.metrics,
            )
    raise NotFoundError(
        f"No model '{ref.model_key}' was trained in training run '{ref.training_id}'.",
        details={
            "training_id": ref.training_id,
            "model_key": ref.model_key,
            "available": [r.model_key for r in record.results],
        },
    )


@router.post("/comparison", response_model=ComparisonResponse)
def compare(request: ComparisonRequest) -> ComparisonResponse:
    entries = [_entry(ref) for ref in request.models]

    common_metrics = set(entries[0].metrics)
    for entry in entries[1:]:
        common_metrics &= set(entry.metrics)

    return ComparisonResponse(common_metrics=sorted(common_metrics), models=entries)
