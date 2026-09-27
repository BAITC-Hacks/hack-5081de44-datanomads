"""Load and serve checksum-verified candidate classifier artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .constants import TOPIC_BY_ID
from .schemas import Alternative, Classification
from contracts import ContractValidationError, validate_document


_TOKEN_RE = re.compile(r"[\wа-яёәғқңөұүһі]+", flags=re.IGNORECASE | re.UNICODE)
_CONFIDENT_THRESHOLD = 0.78
_UNCERTAIN_THRESHOLD = 0.58
_TRAINED_IMPLEMENTATION = "pulse109.multinomial-naive-bayes.v1"
_CANDIDATE_CACHE_SIZE = 8


class CandidateArtifactError(RuntimeError):
    """A candidate artifact could not be safely resolved for inference."""


def _file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise CandidateArtifactError("CANDIDATE_ARTIFACT_URI_INVALID")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise CandidateArtifactError("CANDIDATE_ARTIFACT_URI_INVALID")
    return path.resolve()


def _candidate_root() -> Path:
    return Path(os.environ.get("MODEL_DIR", "/app/trained-artifacts")).resolve() / "candidates"


def _read_artifact(model_version: str, expected_checksum: str) -> tuple[dict[str, Any], bool]:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_checksum):
        raise CandidateArtifactError("CANDIDATE_ARTIFACT_CHECKSUM_REQUIRED")

    production_environment = os.environ.get("PULSE_ENV", "demo").strip().lower() in {
        "prod",
        "production",
    }
    return _read_verified_artifact(
        model_version,
        expected_checksum,
        str(_candidate_root()),
        production_environment,
    )


@lru_cache(maxsize=_CANDIDATE_CACHE_SIZE)
def _read_verified_artifact(
    model_version: str,
    expected_checksum: str,
    candidate_root_value: str,
    production_environment: bool,
) -> tuple[dict[str, Any], bool]:
    """Load immutable artifacts once per version/checksum/runtime root."""

    candidate_root = Path(candidate_root_value).resolve()
    if not candidate_root.is_dir():
        raise CandidateArtifactError("CANDIDATE_MODEL_NOT_FOUND")

    for manifest_path in candidate_root.glob("*/manifest.json"):
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            validate_document(manifest, "ClassifierManifest")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, ContractValidationError, ValueError, TypeError):
            continue

        if manifest.get("model_version") != model_version:
            continue
        if manifest.get("implementation") != _TRAINED_IMPLEMENTATION:
            raise CandidateArtifactError("CANDIDATE_RUNTIME_UNSUPPORTED")
        if manifest.get("artifact_kind") != "TRAINED_ARTIFACT":
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_KIND_UNSUPPORTED")
        if manifest.get("status") not in {"CANDIDATE", "SHADOW"}:
            raise CandidateArtifactError("CANDIDATE_MODEL_NOT_SHADOWABLE")
        if manifest.get("artifact_checksum") != expected_checksum:
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_CHECKSUM_MISMATCH")
        if manifest.get("synthetic") and production_environment:
            raise CandidateArtifactError("CANDIDATE_SYNTHETIC_ARTIFACT_FORBIDDEN")

        artifact_path = _file_path(str(manifest.get("artifact_uri", "")))
        if not artifact_path.is_relative_to(candidate_root):
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_URI_INVALID")
        if artifact_path.parent != manifest_path.parent.resolve():
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_LINEAGE_MISMATCH")
        try:
            artifact_bytes = artifact_path.read_bytes()
        except OSError as error:
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_UNAVAILABLE") from error

        actual_checksum = f"sha256:{hashlib.sha256(artifact_bytes).hexdigest()}"
        if actual_checksum != expected_checksum:
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_CHECKSUM_MISMATCH")
        try:
            artifact = json.loads(artifact_bytes)
            validate_document(artifact, "TrainedClassifierArtifact")
        except (json.JSONDecodeError, UnicodeDecodeError, ContractValidationError, ValueError, TypeError) as error:
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_INVALID") from error

        if (
            artifact.get("model_version") != model_version
            or artifact.get("training_config_version") != manifest.get("training_config", {}).get("version")
            or set(artifact.get("labels", [])) != set(artifact.get("class_log_prior", {}))
            or set(artifact.get("labels", [])) != set(artifact.get("token_log_probability", {}))
        ):
            raise CandidateArtifactError("CANDIDATE_ARTIFACT_LINEAGE_MISMATCH")
        return artifact, bool(manifest.get("synthetic"))

    raise CandidateArtifactError("CANDIDATE_MODEL_NOT_FOUND")


def classify_candidate(
    model_version: str,
    expected_checksum: str,
    text: str,
    language: str,
    top_k: int,
) -> Classification:
    artifact, _ = _read_artifact(model_version, expected_checksum)
    tokens = [
        hashlib.sha256(token.encode("utf-8")).hexdigest()
        for token in _TOKEN_RE.findall(text.lower())
    ]
    if not tokens:
        raise CandidateArtifactError("CANDIDATE_INPUT_HAS_NO_TOKENS")

    labels: list[str] = artifact["labels"]
    log_scores: dict[str, float] = {}
    for label in labels:
        score = float(artifact["class_log_prior"][label])
        token_probabilities: dict[str, float] = artifact["token_log_probability"][label]
        score += sum(float(token_probabilities[token]) for token in tokens if token in token_probabilities)
        log_scores[label] = score

    maximum = max(log_scores.values())
    weights = {label: math.exp(score - maximum) for label, score in log_scores.items()}
    weight_sum = sum(weights.values())
    probabilities = {label: weights[label] / weight_sum for label in labels}
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    topic_id, confidence = ordered[0]
    if confidence >= _CONFIDENT_THRESHOLD:
        confidence_state = "CONFIDENT"
    elif confidence >= _UNCERTAIN_THRESHOLD:
        confidence_state = "UNCERTAIN"
    else:
        confidence_state = "LOW_CONFIDENCE"

    topic = TOPIC_BY_ID.get(topic_id)
    topic_label = topic.name_kz if topic is not None and language == "KZ" else (
        topic.name_ru if topic is not None else topic_id
    )
    alternatives = [
        Alternative(
            topic_id=label,
            topic=(
                TOPIC_BY_ID[label].name_kz
                if label in TOPIC_BY_ID and language == "KZ"
                else TOPIC_BY_ID[label].name_ru
                if label in TOPIC_BY_ID
                else label
            ),
            score=round(probability, 6),
            confidence=round(probability, 6),
        )
        for label, probability in ordered[:top_k]
    ]
    return Classification(
        language=language,
        topic_id=topic_id,
        topic=topic_label,
        label=topic_id,
        confidence=round(confidence, 6),
        confidence_state=confidence_state,
        needs_review=confidence_state != "CONFIDENT",
        alternatives=alternatives,
        model_version=model_version,
    )
