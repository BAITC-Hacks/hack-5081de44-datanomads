"""Score fresh tickets with one local candidate before operator confirmation."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
from typing import Any

import torch

from app.trained_classifier import TrainedClassifierService
from training.contracts import _checksum
from training.feedback_dataset import ID_RE


class ShadowJobError(RuntimeError):
    """Stable failure code without ticket text."""


_cached_model: tuple[str, str, TrainedClassifierService] | None = None


def _candidate_model(root: Path, version: str, expected_checksum: str,
                     manifest_uri: str) -> TrainedClassifierService:
    global _cached_model
    model_path = root / "models" / version
    manifest_path = model_path / "manifest.json"
    if (model_path.is_symlink() or not model_path.is_dir() or
            any(path.is_symlink() for path in model_path.rglob("*")) or
            str(manifest_path.resolve()) != manifest_uri):
        raise ShadowJobError("SHADOW_ARTIFACT_INVALID")
    if _cached_model is not None and _cached_model[:2] == (version, expected_checksum):
        return _cached_model[2]
    previous = _cached_model
    _cached_model = None
    del previous
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        classifier = TrainedClassifierService(model_path)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        raise ShadowJobError("SHADOW_ARTIFACT_INVALID") from None
    if (classifier.model_version != version or
            classifier.metadata.artifact_checksum != expected_checksum or
            (classifier.metadata.model_extra or {}).get("status") != "CANDIDATE"):
        raise ShadowJobError("SHADOW_ARTIFACT_INVALID")
    _cached_model = (version, expected_checksum, classifier)
    return classifier


async def score_shadow_ticket(pool: Any, payload: dict[str, Any]) -> dict:
    try:
        raw_ids = (payload["cycle_id"], payload["ticket_id"], payload["production_prediction_id"])
        if any(not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value) for value in raw_ids):
            raise ValueError("invalid shadow identity")
        cycle_id, ticket_id, prediction_id = (int(value) for value in raw_ids)
        production_version = payload["production_model_version"]
        candidate_version = payload["candidate_model_version"]
        expected_checksum = _checksum(payload["candidate_artifact_checksum"])
        if any(not isinstance(value, str) or not ID_RE.fullmatch(value)
               for value in (production_version, candidate_version)):
            raise ValueError("invalid shadow identity")
    except (KeyError, TypeError, ValueError):
        raise ShadowJobError("INVALID_SHADOW_JOB") from None
    root_value = os.environ.get("PULSE_TRAINING_ROOT")
    if not root_value:
        raise ShadowJobError("TRAINER_NOT_CONFIGURED")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ShadowJobError("SHADOW_ARTIFACT_INVALID")
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT t.original_text, t.language, t.created_at AS ticket_created_at, lc.state, lc.production_model_version, lc.candidate_model_version, lc.evaluation_started_at, lc.evaluation_ends_at, mv.status AS candidate_status, mv.artifact_checksum, mv.manifest_uri, tp.model_version AS ticket_production_model_version, tp.created_at AS production_predicted_at, EXISTS (SELECT 1 FROM operator_decisions od WHERE od.ticket_id = t.id) AS decided FROM tickets t JOIN learning_cycles lc ON lc.id = $1 JOIN model_versions mv ON mv.model_version = lc.candidate_model_version JOIN ticket_predictions tp ON tp.id = $3 AND tp.ticket_id = t.id WHERE t.id = $2",
            cycle_id, ticket_id, prediction_id,
        )
    if row is None:
        raise ShadowJobError("SHADOW_CONTEXT_CHANGED")
    row = dict(row)
    if (row["state"] != "EVALUATE" or row["production_model_version"] != production_version or
            row["candidate_model_version"] != candidate_version or
            row["ticket_production_model_version"] != production_version or
            row["candidate_status"] not in {"CANDIDATE", "SHADOW"} or
            row["artifact_checksum"] != expected_checksum or
            row["evaluation_started_at"] is None or
            row["ticket_created_at"] < row["evaluation_started_at"] or
            (row["evaluation_ends_at"] is not None and row["ticket_created_at"] >= row["evaluation_ends_at"]) or
            row["production_predicted_at"] < row["evaluation_started_at"]):
        raise ShadowJobError("SHADOW_CONTEXT_CHANGED")
    if row["decided"]:
        return {"status": "SKIPPED_OPERATOR_DECIDED", "cycle_id": str(cycle_id),
                "ticket_id": str(ticket_id)}
    classifier = _candidate_model(root, candidate_version, expected_checksum, row["manifest_uri"])
    language = row["language"] if row["language"] in {"RU", "KZ"} else None
    prediction = classifier.classify(row["original_text"], language=language)
    if not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
        raise ShadowJobError("SHADOW_PREDICTION_INVALID")
    async with pool.acquire() as connection:
        saved_id = await connection.fetchval(
            "INSERT INTO classifier_shadow_predictions (cycle_id, ticket_id, production_prediction_id, candidate_model_version, candidate_artifact_checksum, topic_id, confidence) SELECT $1, $2, $3, $4, $5, $6, $7 FROM learning_cycles lc JOIN model_versions mv ON mv.model_version = lc.candidate_model_version WHERE lc.id = $1 AND lc.state = 'EVALUATE' AND lc.production_model_version = $8 AND lc.candidate_model_version = $4 AND mv.status IN ('CANDIDATE', 'SHADOW') AND mv.artifact_checksum = $5 AND NOT EXISTS (SELECT 1 FROM operator_decisions od WHERE od.ticket_id = $2) ON CONFLICT (cycle_id, ticket_id) DO NOTHING RETURNING id",
            cycle_id, ticket_id, prediction_id, candidate_version, expected_checksum,
            prediction.topic_id, prediction.confidence, production_version,
        )
    return {"status": "RECORDED" if saved_id is not None else "SKIPPED_CONTEXT_CHANGED",
            "cycle_id": str(cycle_id), "ticket_id": str(ticket_id),
            "candidate_model_version": candidate_version}
