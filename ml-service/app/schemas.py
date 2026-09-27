"""Pydantic API schemas.

Schemas intentionally accept both singular and batch forms.  This keeps the
internal API convenient for Rust Core while allowing a single smoke request to
be used from the operator demo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class HealthResponse(APIModel):
    status: Literal["ok", "ready"]
    service: str
    version: str
    model_versions: dict[str, str] = Field(default_factory=dict)


class Alternative(APIModel):
    topic_id: str
    topic: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class Classification(APIModel):
    language: str
    topic_id: str
    topic: str
    label: str
    confidence: float = Field(ge=0, le=1)
    confidence_state: Literal["CONFIDENT", "UNCERTAIN", "LOW_CONFIDENCE"]
    needs_review: bool
    alternatives: list[Alternative] = Field(default_factory=list)
    model_version: str


class ClassifyRequest(APIModel):
    text: str | None = None
    texts: list[str] | None = None
    inputs: list[str] | None = None
    language: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    model_version: str | None = None
    expected_artifact_checksum: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "ClassifyRequest":
        values = [self.text is not None, bool(self.texts), bool(self.inputs)]
        if sum(values) != 1:
            raise ValueError("provide exactly one of text, texts, or inputs")
        if self.expected_artifact_checksum is not None and not self.model_version:
            raise ValueError("expected_artifact_checksum requires model_version")
        if self.text is not None and not self.text.strip():
            raise ValueError("text must not be empty")
        if self.texts is not None and any(not item.strip() for item in self.texts):
            raise ValueError("texts must not contain empty values")
        if self.inputs is not None and any(not item.strip() for item in self.inputs):
            raise ValueError("inputs must not contain empty values")
        return self

    def get_texts(self) -> list[str]:
        if self.text is not None:
            return [self.text]
        return self.texts if self.texts is not None else self.inputs or []


class ClassifyResponse(APIModel):
    model_version: str
    predictions: list[Classification]
    language: str | None = None
    topic_id: str | None = None
    topic: str | None = None
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_state: str | None = None
    needs_review: bool | None = None
    alternatives: list[Alternative] = Field(default_factory=list)


class EmbedRequest(APIModel):
    text: str | None = None
    texts: list[str] | None = None
    inputs: list[str] | None = None
    dimension: int = Field(default=32, ge=8, le=1024)
    normalize: bool = True
    model_version: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "EmbedRequest":
        values = [self.text is not None, bool(self.texts), bool(self.inputs)]
        if sum(values) != 1:
            raise ValueError("provide exactly one of text, texts, or inputs")
        if self.text is not None and not self.text.strip():
            raise ValueError("text must not be empty")
        if self.texts is not None and any(not item.strip() for item in self.texts):
            raise ValueError("texts must not contain empty values")
        if self.inputs is not None and any(not item.strip() for item in self.inputs):
            raise ValueError("inputs must not contain empty values")
        return self

    def get_texts(self) -> list[str]:
        if self.text is not None:
            return [self.text]
        return self.texts if self.texts is not None else self.inputs or []


class EmbedResponse(APIModel):
    model_version: str
    dimension: int
    normalized: bool
    embeddings: list[list[float]]
    embedding: list[float] | None = None


class TimeSeriesPoint(APIModel):
    timestamp: datetime | str | None = None
    value: float


class ForecastRequest(APIModel):
    values: list[float] | None = None
    history: list[TimeSeriesPoint] | None = None
    series: list[TimeSeriesPoint] | None = None
    observations: list[float] | None = None
    horizon: int = Field(default=30, ge=1, le=366)
    season_length: int = Field(default=7, ge=1, le=366)
    model_version: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "ForecastRequest":
        supplied = [
            self.values is not None,
            self.history is not None,
            self.series is not None,
            self.observations is not None,
        ]
        if sum(supplied) != 1:
            raise ValueError("provide exactly one of values, observations, history, or series")
        data = self.get_values()
        if not data:
            raise ValueError("forecast history must not be empty")
        return self

    def get_values(self) -> list[float]:
        if self.values is not None:
            return self.values
        if self.observations is not None:
            return self.observations
        points = self.history if self.history is not None else self.series or []
        return [point.value for point in points]

    def get_timestamps(self) -> list[datetime | str | None] | None:
        points = self.history if self.history is not None else self.series
        return [point.timestamp for point in points] if points is not None else None


class ForecastPoint(APIModel):
    index: int
    value: float
    timestamp: datetime | str | None = None


class ForecastResponse(APIModel):
    model_version: str
    model: Literal["seasonal_naive"]
    status: Literal["OK", "INSUFFICIENT_HISTORY"]
    insufficient_history: bool
    horizon: int
    season_length: int
    forecast: list[float]
    points: list[ForecastPoint]
    expected_peaks: list[int] = Field(default_factory=list)
    backtest: dict[str, float | int | None] = Field(default_factory=dict)


class AnomalyRequest(APIModel):
    values: list[float] | None = None
    history: list[TimeSeriesPoint] | None = None
    series: list[TimeSeriesPoint] | None = None
    window: int = Field(default=7, ge=2, le=366)
    threshold: float = Field(default=3.0, gt=0, le=100)
    model_version: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "AnomalyRequest":
        if sum(item is not None for item in (self.values, self.history, self.series)) != 1:
            raise ValueError("provide exactly one of values, history, or series")
        if len(self.get_values()) < 2:
            raise ValueError("anomaly history must contain at least two values")
        return self

    def get_values(self) -> list[float]:
        if self.values is not None:
            return self.values
        points = self.history if self.history is not None else self.series or []
        return [point.value for point in points]

    def get_timestamps(self) -> list[datetime | str | None] | None:
        points = self.history if self.history is not None else self.series
        return [point.timestamp for point in points] if points is not None else None


class AnomalyPoint(APIModel):
    index: int
    value: float
    expected: float
    score: float
    is_anomaly: bool
    timestamp: datetime | str | None = None


class AnomalyResponse(APIModel):
    model_version: str
    threshold: float
    window: int
    is_anomaly: bool
    latest: AnomalyPoint
    anomalies: list[AnomalyPoint]
    points: list[AnomalyPoint]


class TrainingSample(APIModel):
    text: str
    label: str | None = None
    topic_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_label(self) -> str | None:
        return self.label or self.topic_id


class TrainingRequest(APIModel):
    model_type: str = "classifier"
    dataset_version: str = "demo-feedback-v1"
    samples: list[TrainingSample] = Field(default_factory=list)
    min_samples: int = Field(default=1, ge=0)
    base_model: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingResponse(APIModel):
    job_id: str
    state: Literal["COMPLETED", "INSUFFICIENT_FEEDBACK", "TRAINER_NOT_CONFIGURED", "FAILED"]
    model_type: str
    dataset_version: str
    candidate_model_version: str | None = None
    sample_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)


class CandidateTrainingJob(APIModel):
    schema_version: Literal["candidate-training-job.v1"]
    cycle_id: str = Field(min_length=1, max_length=200)
    candidate_model_version: str = Field(min_length=1, max_length=200)
    candidate_dataset_version: str = Field(min_length=1, max_length=200)
    production_model_version: str = Field(min_length=1, max_length=200)
    training_config_version: Literal["classifier-training.v1"]
    dataset_uri: str = Field(min_length=1)
    dataset_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_uri: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_artifact_uri: str = Field(min_length=1)
    min_samples: int = Field(ge=1)


class CandidateTrainingResult(APIModel):
    schema_version: Literal["candidate-training-result.v1"]
    status: Literal["COMPLETED"]
    cycle_id: str
    candidate_model_version: str
    candidate_dataset_version: str
    production_model_version: str
    training_config_version: Literal["classifier-training.v1"]
    artifact_uri: str
    artifact_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    manifest_uri: str
    manifest_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    sample_count: int = Field(ge=1)
    synthetic: bool
    metrics: dict[str, int]
    manifest: dict[str, Any]


class EvaluationRequest(APIModel):
    model_type: str = "classifier"
    dataset_version: str = "demo-eval-v1"
    samples: list[TrainingSample] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    predictions: list[float] = Field(default_factory=list)
    season_length: int = Field(default=7, ge=1, le=366)


class EvaluationResponse(APIModel):
    evaluation_id: str
    model_type: str
    dataset_version: str
    status: Literal["COMPLETED", "INSUFFICIENT_DATA"]
    sample_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_version: str


class CandidateOfflineSample(APIModel):
    ticket_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1, max_length=200)
    language: str = "UNKNOWN"


class CandidateShadowSample(APIModel):
    production_topic_id: str | None = None
    candidate_topic_id: str | None = None
    confirmed_topic_id: str = Field(min_length=1, max_length=200)
    candidate_inference_status: Literal["COMPLETED", "FAILED"]


class CandidateEvaluationRequest(APIModel):
    cycle_id: str = Field(min_length=1, max_length=200)
    candidate_model_version: str = Field(min_length=1, max_length=200)
    candidate_artifact_checksum: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )
    production_model_version: str = Field(min_length=1, max_length=200)
    production_baseline_available: bool = True
    production_baseline_is_synthetic: bool = False
    production_artifact_checksum: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )
    candidate_dataset_version: str = Field(min_length=1, max_length=200)
    frozen_evaluation_dataset_version: str = Field(min_length=1, max_length=200)
    promotion_policy_version: str = Field(min_length=1, max_length=200)
    offline_samples: list[CandidateOfflineSample] = Field(default_factory=list)
    shadow_samples: list[CandidateShadowSample] = Field(default_factory=list)
    shadow_inference_failures: int = Field(default=0, ge=0)
    synthetic: bool = False
    blind_ab_enabled: bool = False


class ModelMetadata(APIModel):
    schema_version: Literal["model-metadata.v1", "classifier-manifest.v1", "embedder-manifest.v1"]
    model_version: str
    model_family: str
    base_model: str | None
    dataset_version: str
    created_at: str
    status: Literal["CANDIDATE", "PRODUCTION", "REJECTED", "ARCHIVED", "DEMO_BASELINE"]
    artifact_kind: Literal["TRAINED_ARTIFACT", "DETERMINISTIC_BASELINE"]
    artifact_uri: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    languages: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    training_config: dict[str, Any] = Field(default_factory=dict)
    artifact_checksum: str | None
    demo_artifact_id: str | None = None
    evaluation_version: str | None = None
    synthetic: bool
    implementation: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    dimension: int | None = Field(default=None, ge=1, le=65536)
    distance_metric: Literal["cosine", "dot", "euclidean"] | None = None
    normalized: bool | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)


class ModelManifestResponse(APIModel):
    schema_version: Literal["model-manifest.v1"]
    manifest_version: str
    service: str
    models: dict[str, ModelMetadata]
