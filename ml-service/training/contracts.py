"""Versioned Data/ML manifests shared by offline builders and evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


Sha256 = str


class StrictManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @classmethod
    def read(cls, path: Path):
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _checksum(value: str) -> str:
    if not value.startswith("sha256:") or len(value) != 71 or any(ch not in "0123456789abcdef" for ch in value[7:]):
        raise ValueError("checksum must be sha256:<64 lowercase hex characters>")
    return value


class DatasetManifest(StrictManifest):
    manifest_version: Literal["dataset-manifest.v1"] = "dataset-manifest.v1"
    dataset_version: str = Field(min_length=1)
    schema_version: Literal["unified-ticket.v1"]
    created_at: AwareDatetime
    synthetic: bool
    seed: int = Field(ge=0)
    sources: list[str] = Field(min_length=1)
    source_checksums: dict[str, Sha256]
    record_count: int = Field(ge=0)
    quarantine_count: int = Field(ge=0)
    languages: dict[str, int]
    topics: dict[str, int]
    regions: dict[str, int]
    split_policy: str = Field(min_length=1)
    split_group_key: str = Field(min_length=1)
    frozen_evaluation_version: str = Field(min_length=1)
    pii_policy_version: str = Field(min_length=1)
    content_sha256: Sha256

    @field_validator("content_sha256")
    @classmethod
    def valid_content_checksum(cls, value: str) -> str:
        return _checksum(value)

    @field_validator("source_checksums")
    @classmethod
    def valid_source_checksums(cls, values: dict[str, str]) -> dict[str, str]:
        return {source: _checksum(checksum) for source, checksum in values.items()}

    @model_validator(mode="after")
    def valid_counts_and_sources(self):
        if len(set(self.sources)) != len(self.sources) or set(self.sources) != set(self.source_checksums):
            raise ValueError("sources and source_checksums must identify the same unique sources")
        for name, counts in (("languages", self.languages), ("topics", self.topics), ("regions", self.regions)):
            if any(count < 0 for count in counts.values()) or sum(counts.values()) != self.record_count:
                raise ValueError(f"{name} counts must sum to record_count")
        return self


class ClassifierManifest(StrictManifest):
    manifest_version: Literal["model-manifest.v1"] = "model-manifest.v1"
    model_version: str = Field(min_length=1)
    model_type: Literal["classifier"] = "classifier"
    model_family: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    synthetic: bool
    seed: int = Field(ge=0)
    labels: list[str] = Field(min_length=2)
    languages: list[str] = Field(min_length=1)
    training_config: dict[str, Any] = Field(min_length=1)
    input_length_strategy: str = Field(min_length=1)
    calibration: dict[str, Any] = Field(min_length=1)
    confidence_thresholds: dict[str, float] = Field(min_length=1)
    held_out_metrics: dict[str, Any] = Field(min_length=1)
    artifact_checksum: Sha256
    runtime_requirements: dict[str, str] = Field(min_length=1)

    @field_validator("artifact_checksum")
    @classmethod
    def valid_artifact_checksum(cls, value: str) -> str:
        return _checksum(value)

    @model_validator(mode="after")
    def valid_labels_and_thresholds(self):
        if len(set(self.labels)) != len(self.labels) or len(set(self.languages)) != len(self.languages):
            raise ValueError("labels and languages must be unique")
        if any(not 0 <= value <= 1 for value in self.confidence_thresholds.values()):
            raise ValueError("confidence thresholds must be in [0, 1]")
        return self


class EmbedderManifest(StrictManifest):
    manifest_version: Literal["model-manifest.v1"] = "model-manifest.v1"
    model_version: str = Field(min_length=1)
    model_type: Literal["embedder"] = "embedder"
    model_family: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    synthetic: bool
    seed: int = Field(ge=0)
    embedding_dimension: int = Field(gt=0)
    pooling: str = Field(min_length=1)
    normalization: str = Field(min_length=1)
    training_config: dict[str, Any] = Field(min_length=1)
    retrieval_metrics: dict[str, Any] = Field(min_length=1)
    artifact_checksum: Sha256
    runtime_requirements: dict[str, str] = Field(min_length=1)

    @field_validator("artifact_checksum")
    @classmethod
    def valid_artifact_checksum(cls, value: str) -> str:
        return _checksum(value)


class EvaluationReport(StrictManifest):
    report_version: Literal["evaluation-report.v1"] = "evaluation-report.v1"
    evaluated_at: AwareDatetime
    task_type: Literal["classifier", "retrieval", "duplicate_repeat", "forecast", "spike", "controlled_learning"]
    dataset_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    synthetic: bool
    sample_count: int = Field(ge=0)
    metrics: dict[str, Any] = Field(min_length=1)
    evaluation_data_checksum: Sha256

    @field_validator("evaluation_data_checksum")
    @classmethod
    def valid_evaluation_checksum(cls, value: str) -> str:
        return _checksum(value)
