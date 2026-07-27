"""Single-query and dataset benchmark evaluation endpoints."""

from pathlib import Path
import tempfile
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.evaluation import BenchmarkRequest, BenchmarkResponse, EvaluationQueryRequest, EvaluationResponse
from app.evaluation.dataset import DatasetLoader


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

_BENCHMARK_SUFFIXES = {".csv", ".json", ".yaml", ".yml"}


@router.post("/query", response_model=EvaluationResponse)
def evaluate_query(request: EvaluationQueryRequest, services: ApiServices = Depends(get_services)) -> EvaluationResponse:
    """Evaluate supplied pipeline artifacts and persist the configured report."""
    start = perf_counter()
    logger.bind(event="evaluation_requested").info("Single-query evaluation requested")
    result = services.evaluation.evaluate_and_report(request.evaluation_input, request.report_name)
    processing_time = perf_counter() - start
    logger.bind(event="evaluation_completed").info("Single-query evaluation completed in {:.3f}s", processing_time)
    return EvaluationResponse(result=result, processing_time=processing_time)


@router.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(request: BenchmarkRequest, services: ApiServices = Depends(get_services)) -> BenchmarkResponse:
    """Execute a local benchmark dataset through the full generation workflow."""
    start = perf_counter()
    logger.bind(event="benchmark_started").info("Benchmark requested")
    path = (
        (Path.cwd() / request.dataset_path).resolve()
        if request.dataset_path
        else services.settings.evaluation_dataset_path
    )
    if path is None:
        raise HTTPException(status_code=400, detail="dataset_path is required")
    result = services.benchmark(DatasetLoader().load(path))
    processing_time = perf_counter() - start
    logger.bind(event="benchmark_completed").info("Benchmark completed in {:.3f}s", processing_time)
    return BenchmarkResponse(**result.model_dump(), processing_time=processing_time)


@router.post("/benchmark/upload", response_model=BenchmarkResponse)
async def run_uploaded_benchmark(
    file: UploadFile = File(...), services: ApiServices = Depends(get_services)
) -> BenchmarkResponse:
    """Run a JSON, YAML, or CSV benchmark uploaded by the browser without persisting it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _BENCHMARK_SUFFIXES:
        raise HTTPException(status_code=415, detail="Benchmark uploads must be JSON, YAML, or CSV")
    try:
        content = await file.read(services.settings.max_upload_size_bytes + 1)
        if not content or len(content) > services.settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="Benchmark upload exceeds the configured size limit")
        start = perf_counter()
        logger.bind(event="benchmark_upload_started").info("Uploaded benchmark requested")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"benchmark{suffix}"
            path.write_bytes(content)
            result = services.benchmark(DatasetLoader().load(path))
        processing_time = perf_counter() - start
        logger.bind(event="benchmark_completed").info("Uploaded benchmark completed in {:.3f}s", processing_time)
        return BenchmarkResponse(**result.model_dump(), processing_time=processing_time)
    finally:
        await file.close()
