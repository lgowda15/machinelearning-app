"""Upload and EDA endpoints (ARCHITECTURE.md SS6, BUILD_SESSIONS.md Session 2).

POST /api/data/upload            -- upload a CSV, get back its EDA profile
GET  /api/data/{data_id}         -- retrieve a previously stored profile
GET  /api/data/samples           -- list the shipped sample datasets
POST /api/data/samples/{id}      -- load a sample dataset as if it were uploaded
"""
import io

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from app.core import eda, storage
from app.core.errors import DataValidationError
from app.core.samples import SampleDataset
from app.core.samples import get_sample as get_sample_dataset
from app.core.samples import list_samples as list_sample_datasets
from app.core.storage import DatasetRecord
from app.schemas.data import DataProfileResponse, SampleDatasetInfo, SampleListResponse
from app.schemas.errors import ErrorResponse

router = APIRouter(
    prefix="/api/data",
    tags=["data"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)


def _parse_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise DataValidationError(f"Could not parse file as CSV: {exc}") from exc
    if len(df.columns) == 0:
        raise DataValidationError("Could not parse file as CSV: no columns found.")
    return df


def _ingest(
    df: pd.DataFrame, target_column: str | None, has_target: bool, source: str
) -> DataProfileResponse:
    eda.validate_raw_dataframe(df)
    resolved_target = eda.resolve_target_column(df, target_column, has_target)
    data_type = eda.infer_data_type(df, resolved_target)
    record = storage.save_dataset(df, data_type, resolved_target, source)
    return _profile(record)


def _profile(record: DatasetRecord) -> DataProfileResponse:
    return eda.profile_dataset(
        record.data_id, record.df, record.data_type, record.target_column, record.source,
    )


def _sample_info(sample: SampleDataset) -> SampleDatasetInfo:
    df, _ = sample.loader()
    return SampleDatasetInfo(
        id=sample.id,
        name=sample.name,
        description=sample.description,
        data_type=sample.data_type,
        n_rows=len(df),
        n_columns=len(df.columns),
    )


@router.post("/upload", response_model=DataProfileResponse)
async def upload_data(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI's parameter-marker pattern requires the call in the default
    target_column: str | None = Form(None),
    has_target: bool = Form(True),
) -> DataProfileResponse:
    raw = await file.read()
    df = _parse_csv(raw)
    return _ingest(df, target_column, has_target, source=file.filename or "upload.csv")


@router.get("/samples", response_model=SampleListResponse)
def get_samples() -> SampleListResponse:
    return SampleListResponse(samples=[_sample_info(s) for s in list_sample_datasets()])


@router.post("/samples/{sample_id}", response_model=DataProfileResponse)
def load_sample(sample_id: str) -> DataProfileResponse:
    sample = get_sample_dataset(sample_id)
    df, target_column = sample.loader()
    return _ingest(df, target_column, has_target=target_column is not None, source=f"sample:{sample.id}")


@router.get("/{data_id}", response_model=DataProfileResponse)
def get_data(data_id: str) -> DataProfileResponse:
    return _profile(storage.get_dataset(data_id))
