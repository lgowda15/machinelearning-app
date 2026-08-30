"""Training endpoints (ARCHITECTURE.md SS6, DATA_FLOW_GUIDE.md Stages 2-4).

POST /api/training/train                       -- train selected models,
                                                    synchronous, returns metrics
GET  /api/training/{training_id}/results        -- retrieve a past training run
"""
from typing import Any

from fastapi import APIRouter

from app.core import storage, training_store
from app.core.compatibility import incompatibility_reason
from app.core.errors import DataValidationError, NotFoundError
from app.core.metrics import compute_metrics, compute_plot_data
from app.core.preprocessing import fit_transform_split
from app.core.registry import REGISTRY
from app.core.training_store import TrainedModelResult
from app.schemas.errors import ErrorResponse
from app.schemas.training import (
    ModelTrainSpec,
    TrainedModelResponse,
    TrainRequest,
    TrainResponse,
)
from models.base_model import BaseModel

router = APIRouter(
    prefix="/api/training",
    tags=["training"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)

# Contract exception (CLAUDE.md): clusterers and reducers are always fitted
# with y=None, whether or not the dataset actually carries a target.
_UNSUPERVISED_MODEL_TYPES = {"clusterer", "dimensionality_reducer"}


def _instantiate(model_key: str, hyperparameters: dict[str, Any]) -> BaseModel:
    cls = REGISTRY[model_key]
    try:
        return cls(**hyperparameters)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Could not construct model '{model_key}' with the given hyperparameters: {exc}",
            details={"model_key": model_key, "hyperparameters": hyperparameters},
        ) from exc


def _check_known_and_compatible(spec: ModelTrainSpec, data_type: str) -> str:
    """Returns the model's model_type. Raises before any fit() is called if
    the key is unknown or the model is incompatible with this dataset --
    the same rule POST /api/models/compatible reports, never a raw
    exception from inside fit() (BUILD_SESSIONS.md Session 3 'Done when')."""
    if spec.model_key not in REGISTRY:
        raise NotFoundError(
            f"No registered model '{spec.model_key}'.",
            details={"model_key": spec.model_key, "available": list(REGISTRY)},
        )
    model_type = REGISTRY[spec.model_key]().get_metadata()["model_type"]
    reason = incompatibility_reason(model_type, data_type)
    if reason is not None:
        raise DataValidationError(
            f"Model '{spec.model_key}' is incompatible with this dataset: {reason}",
            details={"model_key": spec.model_key, "reason": reason},
        )
    return model_type


def _train_one(
    spec: ModelTrainSpec,
    model_type: str,
    X_train,
    X_test,
    y_train,
    y_test,
) -> TrainedModelResult:
    model = _instantiate(spec.model_key, spec.hyperparameters)
    fit_y = None if model_type in _UNSUPERVISED_MODEL_TYPES else y_train

    try:
        model.fit(X_train, fit_y)
        predictions = model.predict(X_test)
        model.predict_proba(X_test)
        metadata = model.get_metadata()
        visualization_data = model.get_visualization_data()
    except Exception as exc:
        # A model's own exception is caught at the route boundary and
        # returned as a clean message -- it never surfaces as a 500
        # (.claude/rules/backend.md "Errors").
        raise DataValidationError(
            f"Model '{spec.model_key}' failed during training: {exc}",
            details={"model_key": spec.model_key},
        ) from exc

    y_true = y_test if model_type in ("classifier", "regressor") else None
    metrics = compute_metrics(model_type, X_test, y_true, predictions)
    plot_data = compute_plot_data(model_type, X_test, y_true, predictions)

    return TrainedModelResult(
        model_key=spec.model_key,
        model=model,
        metrics=metrics,
        metadata=metadata,
        visualization_data=visualization_data,
        plot_data=plot_data,
    )


def _to_response(result: TrainedModelResult) -> TrainedModelResponse:
    md = result.metadata
    return TrainedModelResponse(
        model_key=result.model_key,
        model_name=md["model_name"],
        model_type=md["model_type"],
        metrics=result.metrics,
        hyperparameters=md["hyperparameters"],
        training_time_seconds=md["training_time_seconds"],
        n_features=md["n_features"],
        feature_importance=md["feature_importance"],
        visualization_data=result.visualization_data,
        plot_data=result.plot_data,
    )


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    record = storage.get_dataset(request.data_id)

    # Validate every requested model before fitting any of them.
    model_types = {
        spec.model_key: _check_known_and_compatible(spec, record.data_type)
        for spec in request.models
    }

    split = fit_transform_split(
        record.df,
        target_column=record.target_column,
        data_type=record.data_type,
        test_size=request.test_size,
    )

    results = [
        _train_one(
            spec,
            model_types[spec.model_key],
            split.X_train,
            split.X_test,
            split.y_train,
            split.y_test,
        )
        for spec in request.models
    ]

    training_record = training_store.save_training(
        data_id=request.data_id,
        data_type=record.data_type,
        target_column=record.target_column,
        test_size=request.test_size,
        fitted=split.fitted,
        results=results,
    )

    return TrainResponse(
        training_id=training_record.training_id,
        data_id=training_record.data_id,
        test_size=training_record.test_size,
        results=[_to_response(r) for r in results],
    )


@router.get("/{training_id}/results", response_model=TrainResponse)
def get_results(training_id: str) -> TrainResponse:
    record = training_store.get_training(training_id)
    return TrainResponse(
        training_id=record.training_id,
        data_id=record.data_id,
        test_size=record.test_size,
        results=[_to_response(r) for r in record.results],
    )
