"""FastAPI entrypoint for the Pulse 109 ML boundary."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .candidate_runtime import CandidateArtifactError
from .schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ClassifyRequest,
    ClassifyResponse,
    EmbedRequest,
    EmbedResponse,
    EvaluationRequest,
    EvaluationResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelManifestResponse,
    ModelMetadata,
    TrainingRequest,
    TrainingResponse,
)
from .services import make_services


class JSONFormatter(logging.Formatter):
    """Emit operational logs as JSON without request text or other PII."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "service": "pulse109-ml",
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "n/a"),
            "trace_id": getattr(record, "trace_id", "n/a"),
            "endpoint": getattr(record, "endpoint", "n/a"),
            "latency_ms": getattr(record, "latency_ms", 0.0),
            "model_version": getattr(record, "model_version", "n/a"),
            "status": getattr(record, "status", 0),
            "error_code": getattr(record, "error_code", "none"),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


logger = logging.getLogger("pulse109.ml")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


registry, classifier, embedder, forecaster, anomaly_detector, trainer, evaluator = make_services()

app = FastAPI(
    title="Pulse 109 ML Service",
    version=__version__,
    description="Local deterministic baseline for classification, retrieval embeddings, forecast and anomaly APIs.",
    docs_url="/docs",
    redoc_url="/redoc",
)
router = APIRouter(prefix="/internal/v1")


def require_ready() -> None:
    if not registry.ready:
        raise HTTPException(status_code=503, detail=registry.load_error or "model runtime is not ready")


def require_model_runtime(model_type: str) -> ModelMetadata:
    require_ready()
    try:
        return registry.get(model_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown model type: {model_type}") from exc


@app.middleware("http")
async def request_logging(request: Request, call_next: Any) -> JSONResponse:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={"request_id": request_id, "endpoint": request.url.path, "status": 500, "error_code": "internal_error"},
        )
        raise
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "trace_id": request.headers.get("x-trace-id", request_id),
            "endpoint": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "model_version": "n/a",
            "error_code": "none",
        },
    )
    return response


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", service="pulse109-ml", version=__version__, model_versions=registry.model_versions())


@app.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readyz() -> HealthResponse:
    require_ready()
    return HealthResponse(status="ready", service="pulse109-ml", version=__version__, model_versions=registry.model_versions())


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "pulse109-ml", "version": __version__, "docs": "/docs"}


@router.post("/classify", response_model=ClassifyResponse, tags=["inference"])
async def classify(
    request: ClassifyRequest,
    x_request_id: str | None = Header(default=None, alias="x-request-id"),
    x_trace_id: str | None = Header(default=None, alias="x-trace-id"),
) -> ClassifyResponse:
    configured = require_model_runtime("classifier")
    if (
        request.model_version is not None
        and request.model_version != configured.model_version
        and request.expected_artifact_checksum is None
    ):
        raise HTTPException(status_code=404, detail="requested model version is not loaded")
    try:
        predictions = [
            classifier.classify(
                text,
                request.language,
                request.top_k,
                request.model_version,
                request.expected_artifact_checksum,
            )
            for text in request.get_texts()
        ]
    except CandidateArtifactError as error:
        status_code = 404 if str(error) == "CANDIDATE_MODEL_NOT_FOUND" else 503
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    first = predictions[0] if len(predictions) == 1 else None
    return ClassifyResponse(
        model_version=predictions[0].model_version,
        predictions=predictions,
        language=first.language if first else None,
        topic_id=first.topic_id if first else None,
        topic=first.topic if first else None,
        label=first.label if first else None,
        confidence=first.confidence if first else None,
        confidence_state=first.confidence_state if first else None,
        needs_review=first.needs_review if first else None,
        alternatives=first.alternatives if first else [],
    )


@router.post("/embed", response_model=EmbedResponse, tags=["inference"])
async def embed(
    request: EmbedRequest,
    x_request_id: str | None = Header(default=None, alias="x-request-id"),
    x_trace_id: str | None = Header(default=None, alias="x-trace-id"),
) -> EmbedResponse:
    configured = require_model_runtime("embedder")
    if request.model_version is not None and request.model_version != configured.model_version:
        raise HTTPException(status_code=404, detail="requested model version is not loaded")
    embeddings = [embedder.embed(text, request.dimension, request.normalize) for text in request.get_texts()]
    return EmbedResponse(
        model_version=embedder.model_version,
        dimension=request.dimension,
        normalized=request.normalize,
        embeddings=embeddings,
        embedding=embeddings[0] if len(embeddings) == 1 else None,
    )


@router.post("/forecast", response_model=ForecastResponse, tags=["inference"])
async def forecast(request: ForecastRequest) -> ForecastResponse:
    configured = require_model_runtime("forecast")
    if request.model_version is not None and request.model_version != configured.model_version:
        raise HTTPException(status_code=404, detail="requested model version is not loaded")
    return forecaster.forecast(request.get_values(), request.horizon, request.season_length, request.get_timestamps())


@router.post("/anomaly", response_model=AnomalyResponse, tags=["inference"])
async def anomaly(request: AnomalyRequest) -> AnomalyResponse:
    configured = require_model_runtime("anomaly")
    if request.model_version is not None and request.model_version != configured.model_version:
        raise HTTPException(status_code=404, detail="requested model version is not loaded")
    return anomaly_detector.detect(request.get_values(), request.window, request.threshold, request.get_timestamps())


@router.get("/models", response_model=ModelManifestResponse, tags=["models"])
async def models() -> ModelManifestResponse:
    require_ready()
    assert registry.manifest is not None
    return registry.manifest


@router.get("/models/{model_type}", response_model=ModelMetadata, tags=["models"])
async def model(model_type: str) -> ModelMetadata:
    return require_model_runtime(model_type)


@router.get("/metadata", response_model=ModelManifestResponse, tags=["models"])
async def metadata() -> ModelManifestResponse:
    require_ready()
    assert registry.manifest is not None
    return registry.manifest


@router.post("/training", response_model=TrainingResponse, tags=["training"])
async def train(request: TrainingRequest) -> TrainingResponse:
    require_ready()
    return trainer.train(request)


@router.post("/training/{model_type}", response_model=TrainingResponse, tags=["training"])
async def train_for_model(model_type: str, request: TrainingRequest) -> TrainingResponse:
    require_model_runtime(model_type)
    return trainer.train(request, model_type=model_type)


@router.get("/training/jobs/{job_id}", response_model=TrainingResponse, tags=["training"])
async def training_job(job_id: str) -> TrainingResponse:
    result = trainer.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown training job: {job_id}")
    return result


@router.post("/evaluation", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    require_ready()
    return evaluator.evaluate(request)


@router.post("/evaluation/{model_type}", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluate_for_model(model_type: str, request: EvaluationRequest) -> EvaluationResponse:
    require_model_runtime(model_type)
    return evaluator.evaluate(request, model_type=model_type)


@router.get("/evaluation/jobs/{evaluation_id}", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluation_job(evaluation_id: str) -> EvaluationResponse:
    result = evaluator.get(evaluation_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown evaluation: {evaluation_id}")
    return result


app.include_router(router)
