"""Train versioned classifier artifacts from immutable candidate datasets."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

from contracts import validate_document

from ..schemas import CandidateTrainingJob, CandidateTrainingResult


_TOKEN_RE = re.compile(r"[\wа-яёәғқңөұүһі]+", flags=re.IGNORECASE | re.UNICODE)
_SUPPORTED_TRAINING_CONFIG = "classifier-training.v1"
_NAIVE_BAYES_ALPHA = 1.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _feature_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise RuntimeError("TRAINING_ARTIFACT_URI_NOT_LOCAL")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise RuntimeError("TRAINING_ARTIFACT_URI_NOT_LOCAL")
    return path


def _artifact_path(uri: str, relative_directory: str) -> Path:
    path = _file_path(uri).resolve()
    allowed_root = (
        Path(os.environ.get("MODEL_DIR", "/app/trained-artifacts")).resolve()
        / relative_directory
    )
    if not path.is_relative_to(allowed_root):
        raise RuntimeError("CANDIDATE_ARTIFACT_URI_NOT_ALLOWED")
    return path


def _write_atomically(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    except OSError as error:
        raise RuntimeError("CANDIDATE_ARTIFACT_WRITE_FAILED") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_candidate_dataset(job: CandidateTrainingJob) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        manifest_path = _artifact_path(job.dataset_manifest_uri, "datasets")
        manifest_bytes = manifest_path.read_bytes()
        dataset_path = _artifact_path(job.dataset_uri, "datasets")
        dataset_bytes = dataset_path.read_bytes()
    except OSError as error:
        raise RuntimeError("CANDIDATE_DATASET_ARTIFACT_UNAVAILABLE") from error

    if _sha256(manifest_bytes) != job.dataset_manifest_sha256:
        raise RuntimeError("CANDIDATE_DATASET_MANIFEST_CHECKSUM_MISMATCH")
    if _sha256(dataset_bytes) != job.dataset_checksum:
        raise RuntimeError("CANDIDATE_DATASET_CHECKSUM_MISMATCH")

    try:
        manifest = json.loads(manifest_bytes)
        validate_document(manifest, "CandidateDatasetManifest")
        samples = [json.loads(line) for line in dataset_bytes.splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as error:
        raise RuntimeError("CANDIDATE_DATASET_ARTIFACT_INVALID") from error

    if (
        manifest["dataset_version"] != job.candidate_dataset_version
        or manifest["artifact_uri"] != job.dataset_uri
        or manifest["content_sha256"] != job.dataset_checksum
        or manifest["candidate_model_version"] != job.candidate_model_version
        or manifest["production_model_version"] != job.production_model_version
        or manifest["record_count"] != len(samples)
    ):
        raise RuntimeError("CANDIDATE_DATASET_LINEAGE_MISMATCH")

    if len(samples) < job.min_samples:
        raise RuntimeError("INSUFFICIENT_CANDIDATE_DATASET")

    validated_samples: list[dict[str, str]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise RuntimeError("CANDIDATE_DATASET_ARTIFACT_INVALID")
        text = sample.get("text")
        label = sample.get("label") or sample.get("topic_id")
        language = str(sample.get("language") or "UNKNOWN").upper()
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str) or not label.strip():
            raise RuntimeError("CANDIDATE_DATASET_ARTIFACT_INVALID")
        if language not in {"RU", "KZ", "MIXED", "UNKNOWN"}:
            language = "UNKNOWN"
        validated_samples.append(
            {"text": text.strip(), "label": label.strip(), "language": language}
        )

    return validated_samples, manifest


def _train_multinomial_naive_bayes(
    samples: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, int], list[str], list[str]]:
    class_counts: Counter[str] = Counter()
    token_counts: dict[str, Counter[str]] = {}
    token_totals: Counter[str] = Counter()
    vocabulary: set[str] = set()
    languages: set[str] = set()

    for sample in samples:
        label = sample["label"]
        tokens = [
            _feature_key(token)
            for token in _TOKEN_RE.findall(sample["text"].lower())
        ]
        if not tokens:
            raise RuntimeError("CANDIDATE_DATASET_TEXT_HAS_NO_TOKENS")
        class_counts[label] += 1
        languages.add(sample["language"])
        counts = token_counts.setdefault(label, Counter())
        counts.update(tokens)
        token_totals[label] += len(tokens)
        vocabulary.update(tokens)

    labels = sorted(class_counts)
    ordered_vocabulary = sorted(vocabulary)
    if not labels or not ordered_vocabulary:
        raise RuntimeError("CANDIDATE_DATASET_ARTIFACT_INVALID")

    class_log_prior = {
        label: math.log(class_counts[label] / len(samples))
        for label in labels
    }
    token_log_probability = {
        label: {
            token: math.log(
                (token_counts[label][token] + _NAIVE_BAYES_ALPHA)
                / (token_totals[label] + _NAIVE_BAYES_ALPHA * len(ordered_vocabulary))
            )
            for token in ordered_vocabulary
        }
        for label in labels
    }
    artifact = {
        "schema_version": "trained-classifier-artifact.v1",
        "model_version": "pending",
        "candidate_dataset_version": "pending",
        "production_model_version": "pending",
        "training_config_version": _SUPPORTED_TRAINING_CONFIG,
        "algorithm": "multinomial_naive_bayes",
        "feature_encoding": "sha256",
        "alpha": _NAIVE_BAYES_ALPHA,
        "labels": labels,
        "vocabulary": ordered_vocabulary,
        "class_log_prior": class_log_prior,
        "token_log_probability": token_log_probability,
    }
    metrics = {
        "sample_count": len(samples),
        "class_count": len(labels),
        "vocabulary_size": len(ordered_vocabulary),
    }
    return artifact, metrics, labels, sorted(languages)


def train_candidate_classifier(job: CandidateTrainingJob) -> CandidateTrainingResult:
    """Validate immutable inputs, fit the versioned NB model, and write artifacts."""

    if job.training_config_version != _SUPPORTED_TRAINING_CONFIG:
        raise RuntimeError("CANDIDATE_TRAINING_CONFIG_UNSUPPORTED")
    if job.candidate_model_version == job.production_model_version:
        raise RuntimeError("CANDIDATE_MODEL_VERSION_CONFLICT")

    samples, dataset_manifest = _read_candidate_dataset(job)
    artifact, metrics, labels, languages = _train_multinomial_naive_bayes(samples)
    artifact["model_version"] = job.candidate_model_version
    artifact["candidate_dataset_version"] = job.candidate_dataset_version
    artifact["production_model_version"] = job.production_model_version
    validate_document(artifact, "TrainedClassifierArtifact")

    artifact_path = _artifact_path(job.output_artifact_uri, "candidates")
    artifact_uri = artifact_path.resolve().as_uri()
    artifact_bytes = _canonical_json(artifact)
    artifact_checksum = f"sha256:{_sha256(artifact_bytes)}"
    _write_atomically(artifact_path, artifact_bytes)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": "classifier-manifest.v1",
        "model_version": job.candidate_model_version,
        "model_family": "multinomial-naive-bayes",
        "base_model": None,
        "dataset_version": job.candidate_dataset_version,
        "created_at": created_at,
        "status": "CANDIDATE",
        "artifact_kind": "TRAINED_ARTIFACT",
        "artifact_uri": artifact_uri,
        "metrics": metrics,
        "languages": languages,
        "labels": labels,
        "training_config": {
            "version": job.training_config_version,
            "algorithm": "multinomial_naive_bayes",
            "feature_encoding": "sha256",
            "alpha": _NAIVE_BAYES_ALPHA,
        },
        "artifact_checksum": artifact_checksum,
        "demo_artifact_id": None,
        "evaluation_version": None,
        "synthetic": bool(dataset_manifest["synthetic"]),
        "implementation": "pulse109.multinomial-naive-bayes.v1",
        "provenance": {
            "cycle_id": job.cycle_id,
            "production_model_version": job.production_model_version,
            "candidate_dataset_checksum": job.dataset_checksum,
            "candidate_dataset_manifest_sha256": job.dataset_manifest_sha256,
        },
    }
    validate_document(manifest, "ClassifierManifest")
    manifest_path = artifact_path.with_name("manifest.json")
    manifest_bytes = _canonical_json(manifest)
    _write_atomically(manifest_path, manifest_bytes)

    return CandidateTrainingResult(
        schema_version="candidate-training-result.v1",
        status="COMPLETED",
        cycle_id=job.cycle_id,
        candidate_model_version=job.candidate_model_version,
        candidate_dataset_version=job.candidate_dataset_version,
        production_model_version=job.production_model_version,
        training_config_version=job.training_config_version,
        artifact_uri=artifact_uri,
        artifact_checksum=artifact_checksum,
        manifest_uri=manifest_path.resolve().as_uri(),
        manifest_checksum=_sha256(manifest_bytes),
        sample_count=len(samples),
        synthetic=bool(dataset_manifest["synthetic"]),
        metrics=metrics,
        manifest=manifest,
    )
