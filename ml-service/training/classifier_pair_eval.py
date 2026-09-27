"""Compare production and candidate classifiers on one frozen evaluation set."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import AutoTokenizer

from app.schemas import ModelMetadata
from app.trained_classifier import TrainedClassifierService
from training.classifier_baselines import evaluate_predictions, load_verified_classifier_package
from training.dataset_builder import checksum


class CriticalRegressionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    policy_version: str = Field(pattern=r"^classifier-critical-regression\.v1$")
    critical_topics: list[str] = Field(min_length=1)
    max_f1_drop: float = Field(ge=0, le=1)
    min_topic_support: int = Field(ge=1)
    min_total_samples: int = Field(ge=1)

    @field_validator("critical_topics")
    @classmethod
    def unique_topics(cls, topics: list[str]) -> list[str]:
        if len(topics) != len(set(topics)):
            raise ValueError("critical topics must be unique")
        return topics


def compare_classifiers(package: Path, production_dir: Path, candidate_dir: Path,
                        policy_path: Path) -> dict:
    dataset, splits = load_verified_classifier_package(package)
    policy = CriticalRegressionPolicy.model_validate_json(policy_path.read_text(encoding="utf-8"))
    labels = sorted({row["topic_id"] for row in splits["train"]})
    if {row["topic_id"] for row in splits["test"]} != set(labels):
        raise ValueError("frozen test does not cover all trained topics")
    if not set(policy.critical_topics).issubset(labels):
        raise ValueError("critical regression policy contains unknown topics")

    models = {}
    for name, path in (("production", production_dir), ("candidate", candidate_dir)):
        metadata = ModelMetadata.model_validate_json((path / "manifest.json").read_text(encoding="utf-8"))
        extra = metadata.model_extra or {}
        if (extra.get("frozen_evaluation_version") != dataset.frozen_evaluation_version or
                set(metadata.labels) != set(labels)):
            raise ValueError(f"{name} classifier does not match frozen evaluation version or labels")
        models[name] = metadata
    production = models["production"]
    candidate = models["candidate"]
    candidate_extra = candidate.model_extra or {}
    if production.model_version == candidate.model_version:
        raise ValueError("production and candidate model versions must differ")
    if (candidate.base_model != production.model_version or
            candidate_extra.get("base_model_artifact_checksum") != production.artifact_checksum or
            candidate_extra.get("base_model_manifest_sha256") != checksum(production_dir / "manifest.json") or
            candidate_extra.get("frozen_evaluation_sha256") != dataset.frozen_evaluation_sha256):
        raise ValueError("candidate classifier does not match production or frozen lineage")
    for field in ("max_length", "input_length_strategy"):
        value = production.training_config.get(field)
        if value is None or candidate.training_config.get(field) != value:
            raise ValueError("classifier preprocessing differs between production and candidate")
    rows = splits["test"]
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in (("production", production_dir), ("candidate", candidate_dir))
    }
    max_length = production.training_config["max_length"]
    for row in rows:
        encoded = [tokenizer(row["text"], truncation=True, max_length=max_length)
                   for tokenizer in tokenizers.values()]
        if dict(encoded[0]) != dict(encoded[1]):
            raise ValueError("classifier tokenization differs on frozen evaluation text")
    del tokenizers

    predictions = {}
    for name, path in (("production", production_dir), ("candidate", candidate_dir)):
        classifier = TrainedClassifierService(path)
        predictions[name] = [
            classifier.classify(row["text"], language=row["language"] if row["language"] != "MIXED" else None).topic_id
            for row in rows
        ]
        device = classifier.device
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()
    production_metrics = evaluate_predictions(rows, predictions["production"], labels)
    candidate_metrics = evaluate_predictions(rows, predictions["candidate"], labels)
    critical = {}
    insufficient = []
    regressions = []
    for topic in policy.critical_topics:
        support = production_metrics["per_class"][topic]["support"]
        production_f1 = production_metrics["per_class"][topic]["f1"]
        candidate_f1 = candidate_metrics["per_class"][topic]["f1"]
        drop = round(production_f1 - candidate_f1, 6)
        critical[topic] = {"support": support, "production_f1": production_f1,
                           "candidate_f1": candidate_f1, "f1_drop": drop}
        if support < policy.min_topic_support:
            insufficient.append(topic)
        elif drop > policy.max_f1_drop:
            regressions.append(topic)
    if len(rows) < policy.min_total_samples or insufficient:
        decision = "INSUFFICIENT_EVIDENCE"
    elif regressions:
        decision = "CRITICAL_REGRESSION"
    else:
        decision = "PENDING_HUMAN_REVIEW"
    sample_ids = sorted(row["variant_id"] for row in rows)
    ids_digest = hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "report_version": "classifier-pair-evaluation.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_content_sha256": dataset.content_sha256,
        "frozen_evaluation_version": dataset.frozen_evaluation_version,
        "frozen_evaluation_sha256": dataset.frozen_evaluation_sha256,
        "synthetic": dataset.synthetic,
        "sample_count": len(rows),
        "sample_ids_sha256": f"sha256:{ids_digest}",
        "labels": labels,
        "policy_version": policy.policy_version,
        "policy_sha256": checksum(policy_path),
        "policy": policy.model_dump(),
        "production": {"model_version": production.model_version,
                       "dataset_version": production.dataset_version,
                       "artifact_checksum": production.artifact_checksum,
                       "metrics": production_metrics},
        "candidate": {"model_version": candidate.model_version,
                      "dataset_version": candidate.dataset_version,
                      "artifact_checksum": candidate.artifact_checksum,
                      "metrics": candidate_metrics},
        "macro_f1_delta": round(candidate_metrics["macro_f1"] - production_metrics["macro_f1"], 6),
        "critical_topics": critical,
        "insufficient_critical_topics": insufficient,
        "regressed_critical_topics": regressions,
        "decision": decision,
    }
