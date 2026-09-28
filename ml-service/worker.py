"""PostgreSQL-backed ML background worker.

The worker claims queued rows with FOR UPDATE SKIP LOCKED so multiple
instances can safely process the same queue. Versioned candidate artifacts
remain in shadow until an authorized human promotes or rejects them.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import (
    CandidateEvaluationRequest,
    CandidateShadowSample,
    CandidateTrainingJob,
    EvaluationRequest,
    TrainingRequest,
)
from app.candidate_dataset import build_candidate_dataset, export_validated_feedback
from app.services import make_services, test_fake_trainer_enabled, test_fake_trainer_requested
from contracts import ContractValidationError, validate_document

_CANDIDATE_DATASET_SCHEMA_VERSION = "candidate-training-dataset.v1"
_CLASSIFIER_TRAINING_CONFIG_VERSION = "classifier-training.v1"
_LEARNING_CYCLE_LOCK = "pulse109:classifier-learning-cycle"
_LEARNING_CYCLE_SCHEDULE_VERSION = "global-classifier-cycle-v1"
_LEARNING_SCHEDULER_CHECK_SECONDS = 60.0
_DEFAULT_LEARNING_CYCLE_DURATION_HOURS = 168
_DEFAULT_LEARNING_MIN_FEEDBACK_COUNT = 1
_DEFAULT_LEARNING_PROMOTION_POLICY_VERSION = "policy-v1"
_MAX_LEARNING_CYCLE_DURATION_HOURS = 87_600
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearningCycleScheduleConfig:
    enabled: bool
    environment: str
    collect_duration_hours: int
    minimum_feedback_count: int
    promotion_policy_version: str
    evaluation_dataset_version: str | None


def learning_cycle_schedule_config_from_env() -> LearningCycleScheduleConfig:
    enabled_value = os.environ.get("PULSE_LEARNING_AUTO_CYCLES_ENABLED", "false")
    enabled_normalized = enabled_value.strip().lower()
    if enabled_normalized not in {"1", "true", "yes", "0", "false", "no"}:
        raise RuntimeError("PULSE_LEARNING_AUTO_CYCLES_ENABLED must be true or false")
    enabled = enabled_normalized in {"1", "true", "yes"}
    environment = os.environ.get("PULSE_ENV", "demo").strip().lower()

    if not enabled:
        return LearningCycleScheduleConfig(
            enabled=False,
            environment=environment,
            collect_duration_hours=_DEFAULT_LEARNING_CYCLE_DURATION_HOURS,
            minimum_feedback_count=_DEFAULT_LEARNING_MIN_FEEDBACK_COUNT,
            promotion_policy_version=_DEFAULT_LEARNING_PROMOTION_POLICY_VERSION,
            evaluation_dataset_version=None,
        )

    collect_duration_hours = int(
        os.environ.get(
            "PULSE_LEARNING_CYCLE_DURATION_HOURS",
            str(_DEFAULT_LEARNING_CYCLE_DURATION_HOURS),
        )
    )
    if not 1 <= collect_duration_hours <= _MAX_LEARNING_CYCLE_DURATION_HOURS:
        raise RuntimeError("PULSE_LEARNING_CYCLE_DURATION_HOURS is out of range")

    minimum_feedback_count = int(
        os.environ.get(
            "PULSE_LEARNING_MIN_FEEDBACK_COUNT",
            str(_DEFAULT_LEARNING_MIN_FEEDBACK_COUNT),
        )
    )
    if minimum_feedback_count < 1:
        raise RuntimeError("PULSE_LEARNING_MIN_FEEDBACK_COUNT must be positive")

    promotion_policy_version = os.environ.get(
        "PULSE_LEARNING_PROMOTION_POLICY_VERSION",
        _DEFAULT_LEARNING_PROMOTION_POLICY_VERSION,
    ).strip()
    if not promotion_policy_version:
        raise RuntimeError("PULSE_LEARNING_PROMOTION_POLICY_VERSION must not be empty")

    evaluation_dataset_version = os.environ.get(
        "PULSE_LEARNING_EVALUATION_DATASET_VERSION", ""
    ).strip() or None
    return LearningCycleScheduleConfig(
        enabled=enabled,
        environment=environment,
        collect_duration_hours=collect_duration_hours,
        minimum_feedback_count=minimum_feedback_count,
        promotion_policy_version=promotion_policy_version,
        evaluation_dataset_version=evaluation_dataset_version,
    )


async def start_next_recurring_learning_cycle(
    pool: Any,
    config: LearningCycleScheduleConfig,
) -> str:
    """Share Core's creation lock and wait for human terminal states before recurring."""

    if not config.enabled:
        return "DISABLED"
    if config.environment not in {"demo", "development", "test", "unit", "production"}:
        return "BLOCKED_UNSUPPORTED_ENVIRONMENT"
    if test_fake_trainer_requested():
        return "BLOCKED_TEST_FAKE_TRAINER_ENABLED"
    if config.evaluation_dataset_version is None:
        return "BLOCKED_EVALUATION_DATASET_NOT_CONFIGURED"

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))",
                _LEARNING_CYCLE_LOCK,
            )
            latest_cycle = await connection.fetchrow(
                "SELECT state FROM learning_cycles ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE"
            )
            if latest_cycle is not None:
                latest_state = str(latest_cycle["state"])
                if latest_state in {"COLLECT", "TRAINING", "EVALUATE", "DECISION"}:
                    return "ACTIVE_CYCLE"
                if latest_state not in {"PROMOTED", "REJECTED", "INSUFFICIENT_FEEDBACK"}:
                    return "BLOCKED_PREVIOUS_CYCLE_FAILED"

            dataset = await connection.fetchrow(
                "SELECT dv.is_synthetic, COUNT(dtl.ticket_id)::int AS linked_ticket_count "
                "FROM dataset_versions dv LEFT JOIN dataset_ticket_links dtl "
                "USING (dataset_version) WHERE dv.dataset_version = $1 "
                "GROUP BY dv.is_synthetic",
                config.evaluation_dataset_version,
            )
            if dataset is None:
                return "BLOCKED_EVALUATION_DATASET_NOT_REGISTERED"
            if int(dataset["linked_ticket_count"]) < 1:
                return "BLOCKED_EVALUATION_DATASET_EMPTY"
            if config.environment == "production" and bool(dataset["is_synthetic"]):
                return "BLOCKED_SYNTHETIC_EVALUATION_DATASET"

            production_model_version = await connection.fetchval(
                "SELECT model_version FROM model_versions WHERE status = 'PRODUCTION' "
                "ORDER BY created_at DESC LIMIT 1"
            )
            if not production_model_version:
                return "BLOCKED_PRODUCTION_MODEL_UNAVAILABLE"

            cycle_id = f"cycle-{uuid.uuid4().hex}"
            candidate_model_version = f"classifier-candidate-{uuid.uuid4().hex}"
            request_id = f"auto-cycle-{uuid.uuid4().hex}"
            cycle_database_id = await connection.fetchval(
                "INSERT INTO learning_cycles (cycle_id, state, collect_started_at, "
                "collect_ends_at, production_model_version, candidate_model_version, "
                "min_feedback_count, promotion_policy_version, manual_close_enabled, "
                "frozen_evaluation_dataset_version) "
                "VALUES ($1, 'COLLECT', now(), now() + make_interval(hours => $2), "
                "$3, $4, $5, $6, FALSE, $7) RETURNING id",
                cycle_id,
                config.collect_duration_hours,
                production_model_version,
                candidate_model_version,
                config.minimum_feedback_count,
                config.promotion_policy_version,
                config.evaluation_dataset_version,
            )
            await connection.execute(
                "INSERT INTO learning_cycle_evaluation_tickets "
                "(learning_cycle_id, dataset_version, ticket_id) "
                "SELECT $1, $2, ticket_id FROM dataset_ticket_links "
                "WHERE dataset_version = $2 ON CONFLICT DO NOTHING",
                cycle_database_id,
                config.evaluation_dataset_version,
            )
            await connection.execute(
                "INSERT INTO audit_log (actor_id, action, entity_type, entity_id, "
                "request_id, reason, metadata) VALUES ($1, 'CREATE_LEARNING_CYCLE', "
                "'learning_cycle', $2, $3, 'automatic recurring schedule', $4)",
                "learning-scheduler",
                cycle_id,
                request_id,
                json.dumps(
                    {
                        "schedule_version": _LEARNING_CYCLE_SCHEDULE_VERSION,
                        "scope": "global_classifier",
                        "candidate_model_version": candidate_model_version,
                        "evaluation_dataset_version": config.evaluation_dataset_version,
                    }
                ),
            )
            return "STARTED"


def run_stdin_job(kind: str | None, payload: dict[str, Any]) -> int:
    selected_kind = kind or payload.pop("kind", "evaluation")
    _, _, _, _, _, trainer, evaluator = make_services()
    try:
        if selected_kind == "training":
            result = trainer.train(TrainingRequest.model_validate(payload))
            if result.state == "TRAINER_NOT_CONFIGURED":
                raise RuntimeError(result.state)
        elif selected_kind == "evaluation":
            result = evaluator.evaluate(EvaluationRequest.model_validate(payload))
        else:
            raise ValueError(f"unsupported job kind: {selected_kind}")
        json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # pragma: no cover - CLI failure path
        json.dump({"error": safe_job_error(exc)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1


async def claim_job(pool: Any) -> Any | None:
    """Atomically claim the oldest queued job, without blocking peers."""

    async with pool.acquire() as connection:
        async with connection.transaction():
            return await connection.fetchrow(
                """
                WITH candidate AS (
                    SELECT id
                    FROM background_jobs
                    WHERE state = 'QUEUED'
                    ORDER BY created_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE background_jobs AS job
                SET state = 'RUNNING',
                    attempt = job.attempt + 1,
                    started_at = now(),
                    error = NULL
                FROM candidate
                WHERE job.id = candidate.id
                RETURNING job.id, job.job_type, job.payload, job.attempt
                """,
            )


async def advance_expired_learning_cycle(pool: Any) -> bool:
    """Close one expired cycle and queue its next training or evaluation job."""

    async with pool.acquire() as connection:
        async with connection.transaction():
            closed_evaluation = await connection.fetchrow(
                """
                WITH expired AS (
                    SELECT id
                    FROM learning_cycles
                    WHERE state = 'EVALUATE' AND evaluation_ends_at <= now()
                    ORDER BY evaluation_ends_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE learning_cycles AS cycle
                SET state = 'DECISION',
                    decision_note = 'EVALUATION_WINDOW_CLOSED',
                    updated_at = now()
                FROM expired
                WHERE cycle.id = expired.id
                    RETURNING cycle.id, cycle.cycle_id
                """
            )
            if closed_evaluation is not None:
                evaluation_payload = {
                    "kind": "candidate_evaluation",
                    "cycle_id": str(closed_evaluation["cycle_id"]),
                }
                await connection.execute(
                    "INSERT INTO background_jobs (job_type, payload, state) VALUES ('CANDIDATE_EVALUATION', $1::jsonb, 'QUEUED')",
                    json.dumps(evaluation_payload),
                )
                return True

            cycle = await connection.fetchrow(
                """
                SELECT lc.id, lc.cycle_id, lc.min_feedback_count,
                       lc.candidate_model_version, lc.production_model_version,
                       lc.frozen_evaluation_dataset_version,
                       (SELECT COUNT(*)::int FROM learning_feedback lf
                        WHERE lf.cycle_id = lc.id AND lf.validation_status = 'VALID') AS feedback_count,
                       (SELECT COALESCE(array_agg(lf.id::text ORDER BY lf.id), ARRAY[]::text[])
                        FROM learning_feedback lf
                        WHERE lf.cycle_id = lc.id AND lf.validation_status = 'VALID') AS feedback_ids,
                       (SELECT COALESCE(array_agg(let.ticket_id::text ORDER BY let.ticket_id), ARRAY[]::text[])
                        FROM learning_cycle_evaluation_tickets let
                        WHERE let.learning_cycle_id = lc.id) AS frozen_evaluation_ticket_ids
                FROM learning_cycles lc
                WHERE lc.state = 'COLLECT' AND lc.collect_ends_at <= now()
                ORDER BY lc.collect_ends_at, lc.id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            if cycle is None:
                return False

            cycle_id = str(cycle["cycle_id"])
            feedback_count = int(cycle["feedback_count"])
            minimum = int(cycle["min_feedback_count"])
            if feedback_count < minimum:
                await connection.execute(
                    """
                    UPDATE learning_cycles
                    SET state = 'INSUFFICIENT_FEEDBACK',
                        decision_note = 'INSUFFICIENT_FEEDBACK',
                        updated_at = now()
                    WHERE id = $1 AND state = 'COLLECT'
                    """,
                    int(cycle["id"]),
                )
                return True

            await connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'TRAINING', decision_note = 'BUILDING_CANDIDATE_DATASET', updated_at = now()
                WHERE id = $1 AND state = 'COLLECT'
                """,
                int(cycle["id"]),
            )
            payload = {
                "kind": "build_candidate_dataset",
                "cycle_id": cycle_id,
                "candidate_model_version": cycle["candidate_model_version"] or "pending",
                "production_model_version": cycle["production_model_version"],
                "feedback_ids": [str(value) for value in cycle["feedback_ids"]],
                "frozen_evaluation_dataset_version": cycle[
                    "frozen_evaluation_dataset_version"
                ],
                "frozen_evaluation_ticket_ids": [
                    str(value) for value in cycle["frozen_evaluation_ticket_ids"]
                ],
            }
            await connection.execute(
                """
                INSERT INTO background_jobs (job_type, payload, state)
                VALUES ('BUILD_CANDIDATE_DATASET', $1::jsonb, 'QUEUED')
                """,
                json.dumps(payload),
            )
            return True


async def complete_job(pool: Any, job_id: int, result: Any) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE background_jobs
            SET state = 'COMPLETED',
                finished_at = now(),
                payload = payload || jsonb_build_object('result', $2::jsonb)
            WHERE id = $1
            """,
            job_id,
            json.dumps(result, ensure_ascii=False),
        )


async def fail_job(pool: Any, job_id: int, error: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE background_jobs
            SET state = 'FAILED',
                finished_at = now(),
                error = $2
            WHERE id = $1
            """,
            job_id,
            error[:4000],
        )


def safe_job_error(error: Exception) -> str:
    safe_runtime_errors = {
        "TRAINER_NOT_CONFIGURED",
        "PRODUCTION_BASELINE_NOT_CONFIGURED",
        "FROZEN_EVALUATION_SET_NOT_CONFIGURED",
        "VALIDATED_FEEDBACK_UNAVAILABLE",
        "VALIDATED_FEEDBACK_CHANGED",
        "FEEDBACK_LABELS_UNAVAILABLE",
        "PRODUCTION_BASELINE_MISMATCH",
        "TEST_FAKE_TRAINER_NOT_ALLOWED",
        "SOURCE_TICKET_UNAVAILABLE",
        "SOURCE_TICKET_TEXT_UNAVAILABLE",
        "INSUFFICIENT_CANDIDATE_DATASET",
        "FEEDBACK_EXPORT_CHECKSUM_MISMATCH",
        "FEEDBACK_EXPORT_LINEAGE_MISMATCH",
        "CANDIDATE_DATASET_VERSION_CONFLICT",
        "CANDIDATE_MODEL_VERSION_CONFLICT",
        "CANDIDATE_TRAINING_FAILED",
        "TRAINING_ARTIFACT_URI_NOT_LOCAL",
        "CANDIDATE_ARTIFACT_URI_NOT_ALLOWED",
        "CANDIDATE_DATASET_ARTIFACT_UNAVAILABLE",
        "CANDIDATE_DATASET_MANIFEST_CHECKSUM_MISMATCH",
        "CANDIDATE_DATASET_CHECKSUM_MISMATCH",
        "CANDIDATE_DATASET_ARTIFACT_INVALID",
        "CANDIDATE_DATASET_LINEAGE_MISMATCH",
        "CANDIDATE_TRAINING_CONFIG_UNSUPPORTED",
        "CANDIDATE_DATASET_TEXT_HAS_NO_TOKENS",
        "CANDIDATE_ARTIFACT_WRITE_FAILED",
        "LEARNING_CYCLE_NOT_FOUND",
        "INVALID_CYCLE_ID",
        "INVALID_FEEDBACK_TIMESTAMP",
    }
    if isinstance(error, RuntimeError) and str(error) in safe_runtime_errors:
        return str(error)
    if isinstance(error, (ValidationError, ContractValidationError, json.JSONDecodeError)):
        return "INVALID_JOB_PAYLOAD"
    return "JOB_FAILED"


async def update_learning_cycle(pool: Any, payload: dict[str, Any], result: dict[str, Any] | None = None, error: str | None = None) -> None:
    cycle_id = payload.get("cycle_id")
    if not cycle_id:
        return
    if error:
        await pool.execute(
            "UPDATE learning_cycles SET state = 'TRAINING_FAILED', decision_note = $2, updated_at = now() WHERE cycle_id = $1 OR id::text = $1",
            str(cycle_id),
            error[:4000],
        )
        return
    result = result or {}
    candidate = result.get("candidate_model_version")
    manifest = result.get("manifest") or {}
    if candidate:
        candidate_dataset_version = (
            result.get("candidate_dataset_version")
            or result.get("dataset_version")
            or payload.get("candidate_dataset_version")
            or payload.get("dataset_version")
            or "unknown"
        )
        manifest_uri = result.get("manifest_uri")
        if manifest_uri is None:
            manifest_uri = manifest.get("artifact_uri")
        await pool.execute(
            "INSERT INTO model_versions (model_version, model_family, dataset_version, status, manifest_uri, artifact_checksum) VALUES ($1, $2, $3, 'SHADOW', $4, $5) ON CONFLICT (model_version) DO UPDATE SET status = 'SHADOW', manifest_uri = EXCLUDED.manifest_uri, artifact_checksum = EXCLUDED.artifact_checksum",
            str(candidate),
            str(manifest.get("model_family") or "classifier-baseline-candidate"),
            str(candidate_dataset_version),
            str(manifest_uri) if manifest_uri else None,
            manifest.get("artifact_checksum"),
        )
    fake_trainer_used = manifest.get("implementation") == "test-fake-trainer"
    await pool.execute(
        "UPDATE learning_cycles SET state = 'EVALUATE', evaluation_started_at = now(), evaluation_ends_at = now() + (collect_ends_at - collect_started_at), blind_ab_enabled = false, updated_at = now(), decision_note = $2 WHERE (cycle_id = $1 OR id::text = $1) AND state = 'TRAINING'",
        str(cycle_id),
        "FAKE_TRAINER_COMPLETED" if fake_trainer_used else "TRAINING_COMPLETED",
    )


def candidate_output_artifact_uri(cycle_id: str, model_version: str) -> str:
    artifact_identity = hashlib.sha256(f"{cycle_id}\0{model_version}".encode("utf-8")).hexdigest()[:24]
    artifact_path = (
        Path(os.environ.get("MODEL_DIR", "/app/trained-artifacts"))
        .resolve()
        / "candidates"
        / artifact_identity
        / "classifier.json"
    )
    return artifact_path.as_uri()


def _verified_artifact_checksum(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        return None
    checksum = value.removeprefix("sha256:")
    return value if all(character in "0123456789abcdef" for character in checksum) else None


async def evaluate_candidate_cycle(pool: Any, payload: dict[str, Any], evaluator: Any) -> dict[str, Any]:
    """Collect cycle evidence and persist the Data/ML evaluator result with its lineage."""

    cycle_key = str(payload.get("cycle_id") or "")
    if not cycle_key:
        raise RuntimeError("LEARNING_CYCLE_NOT_FOUND")

    async with pool.acquire() as connection:
        cycle = await connection.fetchrow(
            """
            SELECT lc.id, lc.cycle_id, lc.state, lc.candidate_model_version,
                   lc.production_model_version, lc.candidate_dataset_version,
                   lc.frozen_evaluation_dataset_version, lc.promotion_policy_version,
                   lc.blind_ab_enabled,
                   candidate.artifact_checksum AS candidate_artifact_checksum,
                   production.artifact_checksum AS production_artifact_checksum,
                   COALESCE(production_dataset.is_synthetic, FALSE) AS production_is_synthetic,
                   COALESCE(candidate_dataset.is_synthetic, FALSE) AS candidate_is_synthetic,
                   COALESCE(evaluation_dataset.is_synthetic, FALSE) AS evaluation_is_synthetic
            FROM learning_cycles lc
            LEFT JOIN model_versions candidate
                   ON candidate.model_version = lc.candidate_model_version
            LEFT JOIN model_versions production
                   ON production.model_version = lc.production_model_version
            LEFT JOIN dataset_versions production_dataset
                   ON production_dataset.dataset_version = production.dataset_version
            LEFT JOIN dataset_versions candidate_dataset
                   ON candidate_dataset.dataset_version = lc.candidate_dataset_version
            LEFT JOIN dataset_versions evaluation_dataset
                   ON evaluation_dataset.dataset_version = lc.frozen_evaluation_dataset_version
            WHERE lc.cycle_id = $1 OR lc.id::text = $1
            LIMIT 1
            """,
            cycle_key,
        )
        if cycle is None:
            raise RuntimeError("LEARNING_CYCLE_NOT_FOUND")
        if str(cycle["state"]) != "DECISION":
            raise RuntimeError("LEARNING_CYCLE_NOT_READY_FOR_EVALUATION")

        cycle_db_id = int(cycle["id"])
        frozen_dataset_version = cycle["frozen_evaluation_dataset_version"]
        offline_rows = []
        if frozen_dataset_version:
            offline_rows = await connection.fetch(
                """
                SELECT t.id::text AS ticket_id, t.original_text AS text,
                       t.topic_id AS label, t.language
                FROM learning_cycle_evaluation_tickets evaluation_ticket
                JOIN tickets t ON t.id = evaluation_ticket.ticket_id
                JOIN dataset_ticket_links dataset_ticket
                  ON dataset_ticket.ticket_id = t.id
                 AND dataset_ticket.dataset_version = evaluation_ticket.dataset_version
                WHERE evaluation_ticket.learning_cycle_id = $1
                  AND evaluation_ticket.dataset_version = $2
                ORDER BY t.id
                """,
                cycle_db_id,
                str(frozen_dataset_version),
            )
        shadow_rows = await connection.fetch(
            """
            SELECT sp.production_prediction->>'topic_id' AS production_topic_id,
                   sp.candidate_prediction->>'topic_id' AS candidate_topic_id,
                   od.confirmed_topic_id,
                   sp.candidate_inference_status
            FROM learning_cycle_shadow_predictions sp
            JOIN operator_decisions od ON od.id = sp.operator_decision_id
            WHERE sp.learning_cycle_id = $1
            ORDER BY sp.predicted_at, sp.ticket_id
            """,
            cycle_db_id,
        )
        shadow_failures = await connection.fetchval(
            "SELECT COUNT(*)::int FROM learning_cycle_shadow_predictions WHERE learning_cycle_id = $1 AND candidate_inference_status = 'FAILED'",
            cycle_db_id,
        )

        offline_samples = [
            {
                "ticket_id": str(row["ticket_id"]),
                "text": str(row["text"] or "").strip(),
                "label": str(row["label"] or "").strip(),
                "language": str(row["language"] or "UNKNOWN"),
            }
            for row in offline_rows
            if str(row["text"] or "").strip() and str(row["label"] or "").strip()
        ]
        shadow_samples = [
            CandidateShadowSample(
                production_topic_id=(
                    str(row["production_topic_id"])
                    if row["production_topic_id"] is not None
                    else None
                ),
                candidate_topic_id=(
                    str(row["candidate_topic_id"])
                    if row["candidate_topic_id"] is not None
                    else None
                ),
                confirmed_topic_id=str(row["confirmed_topic_id"]),
                candidate_inference_status=str(row["candidate_inference_status"]),
            )
            for row in shadow_rows
            if row["confirmed_topic_id"] is not None
        ]

        candidate_model_version = str(cycle["candidate_model_version"] or "unavailable-candidate")
        production_baseline_available = cycle["production_model_version"] is not None
        production_model_version = str(
            cycle["production_model_version"] or "unconfigured-production"
        )
        request = CandidateEvaluationRequest(
            cycle_id=str(cycle["cycle_id"]),
            candidate_model_version=candidate_model_version,
            candidate_artifact_checksum=_verified_artifact_checksum(
                cycle["candidate_artifact_checksum"]
            ),
            production_model_version=production_model_version,
            production_baseline_available=production_baseline_available,
            production_baseline_is_synthetic=bool(cycle["production_is_synthetic"]),
            production_artifact_checksum=_verified_artifact_checksum(
                cycle["production_artifact_checksum"]
            ),
            candidate_dataset_version=str(
                cycle["candidate_dataset_version"] or "unavailable-dataset"
            ),
            frozen_evaluation_dataset_version=str(
                frozen_dataset_version or "unavailable-evaluation-dataset"
            ),
            promotion_policy_version=str(
                cycle["promotion_policy_version"] or "unknown-policy"
            ),
            offline_samples=offline_samples,
            shadow_samples=shadow_samples,
            shadow_inference_failures=int(shadow_failures or 0),
            synthetic=(
                test_fake_trainer_enabled()
                or bool(cycle["production_is_synthetic"])
                or bool(cycle["candidate_is_synthetic"])
                or bool(cycle["evaluation_is_synthetic"])
            ),
            blind_ab_enabled=bool(cycle["blind_ab_enabled"]),
        )

    result = evaluator.evaluate_candidate(request)
    validate_document(result, "CandidateEvaluation")
    offline_evaluation = result["offline_evaluation"]
    offline_metrics = dict(offline_evaluation["metrics"])
    offline_metrics["status"] = offline_evaluation["status"]
    shadow_evaluation = result["shadow_evaluation"]
    regressions = sorted(
        set(offline_evaluation["critical_regressions"])
        | set(shadow_evaluation["critical_regressions"])
    )
    evaluation_digest = hashlib.sha256(request.cycle_id.encode("utf-8")).hexdigest()[:24]
    evaluation_id = f"candidate-evaluation-{evaluation_digest}"

    async with pool.acquire() as connection:
        async with connection.transaction():
            locked_cycle = await connection.fetchrow(
                "SELECT id, state, candidate_model_version FROM learning_cycles WHERE id = $1 FOR UPDATE",
                cycle_db_id,
            )
            if locked_cycle is None or str(locked_cycle["state"]) != "DECISION":
                raise RuntimeError("LEARNING_CYCLE_CHANGED_DURING_EVALUATION")
            if str(locked_cycle["candidate_model_version"]) != request.candidate_model_version:
                raise RuntimeError("CANDIDATE_MODEL_LINEAGE_MISMATCH")
            await connection.execute(
                """
                INSERT INTO model_evaluations (
                    evaluation_id, model_version, evaluation_version, split_version,
                    metrics_json, shadow_metrics_json, critical_regressions,
                    sample_size, decision, evaluator, learning_cycle_id,
                    evaluation_payload
                )
                VALUES ($1, $2, 'candidate-evaluation.v1', 'frozen-evaluation.v1',
                        $3::jsonb, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10::jsonb)
                ON CONFLICT (learning_cycle_id) WHERE learning_cycle_id IS NOT NULL
                DO NOTHING
                """,
                evaluation_id,
                request.candidate_model_version,
                json.dumps(offline_metrics),
                json.dumps(shadow_evaluation),
                json.dumps(regressions),
                int(offline_evaluation["sample_count"]),
                str(result["decision"]),
                str(offline_evaluation.get("evaluator") or "pulse109.ml.candidate-evaluator.v1"),
                cycle_db_id,
                json.dumps(result),
            )
    return result


async def persist_candidate_dataset_and_queue_training(
    pool: Any,
    build_job_id: int,
    request_payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Register dataset lineage and queue training in one transaction."""

    async with pool.acquire() as connection:
        async with connection.transaction():
            cycle = await connection.fetchrow(
                """
                SELECT id, state, production_model_version, candidate_model_version,
                       frozen_evaluation_dataset_version, candidate_dataset_version,
                       min_feedback_count
                FROM learning_cycles
                WHERE cycle_id = $1 OR id::text = $1
                LIMIT 1
                FOR UPDATE
                """,
                str(request_payload["cycle_id"]),
            )
            if cycle is None:
                raise RuntimeError("LEARNING_CYCLE_NOT_FOUND")
            if cycle["state"] != "TRAINING":
                raise RuntimeError("CANDIDATE_DATASET_VERSION_CONFLICT")
            if (
                cycle["production_model_version"] != result["production_model_version"]
                or cycle["candidate_model_version"] != result["candidate_model_version"]
                or cycle["candidate_model_version"] == cycle["production_model_version"]
                or cycle["frozen_evaluation_dataset_version"]
                != result["frozen_evaluation_dataset_version"]
                or request_payload.get("frozen_evaluation_dataset_version")
                != result["frozen_evaluation_dataset_version"]
            ):
                raise RuntimeError("CANDIDATE_DATASET_VERSION_CONFLICT")
            if int(result["record_count"]) < int(cycle["min_feedback_count"]):
                raise RuntimeError("INSUFFICIENT_CANDIDATE_DATASET")
            if cycle["candidate_dataset_version"] not in (
                None,
                result["dataset_version"],
            ):
                raise RuntimeError("CANDIDATE_DATASET_VERSION_CONFLICT")

            await connection.execute(
                """
                INSERT INTO dataset_versions (
                    dataset_version, schema_version, manifest_uri, manifest_sha256,
                    content_sha256, is_synthetic, record_count, quarantine_record_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, 0)
                ON CONFLICT (dataset_version) DO NOTHING
                """,
                result["dataset_version"],
                _CANDIDATE_DATASET_SCHEMA_VERSION,
                result["manifest_uri"],
                result["manifest_sha256"],
                result["content_sha256"],
                bool(result["synthetic"]),
                int(result["record_count"]),
            )
            registered_dataset = await connection.fetchrow(
                """
                SELECT schema_version, manifest_uri, manifest_sha256,
                       content_sha256, is_synthetic, record_count
                FROM dataset_versions
                WHERE dataset_version = $1
                FOR UPDATE
                """,
                result["dataset_version"],
            )
            if registered_dataset is None or (
                registered_dataset["schema_version"] != _CANDIDATE_DATASET_SCHEMA_VERSION
                or registered_dataset["manifest_uri"] != result["manifest_uri"]
                or registered_dataset["manifest_sha256"] != result["manifest_sha256"]
                or registered_dataset["content_sha256"] != result["content_sha256"]
                or bool(registered_dataset["is_synthetic"]) != bool(result["synthetic"])
                or int(registered_dataset["record_count"]) != int(result["record_count"])
            ):
                raise RuntimeError("CANDIDATE_DATASET_VERSION_CONFLICT")

            await connection.execute(
                """
                INSERT INTO dataset_ticket_links (dataset_version, ticket_id)
                SELECT $1, ids.ticket_id
                FROM UNNEST($2::bigint[]) AS ids(ticket_id)
                ON CONFLICT DO NOTHING
                """,
                result["dataset_version"],
                [int(value) for value in result["training_ticket_ids"]],
            )
            updated_cycle = await connection.execute(
                """
                UPDATE learning_cycles
                SET candidate_dataset_version = $2,
                    candidate_dataset_checksum = $3,
                    decision_note = 'CANDIDATE_DATASET_READY',
                    updated_at = now()
                WHERE id = $1 AND state = 'TRAINING'
                """,
                int(cycle["id"]),
                result["dataset_version"],
                result["content_sha256"],
            )
            if updated_cycle != "UPDATE 1":
                raise RuntimeError("CANDIDATE_DATASET_VERSION_CONFLICT")

            training_payload = {
                "schema_version": "candidate-training-job.v1",
                "cycle_id": str(request_payload["cycle_id"]),
                "candidate_model_version": str(cycle["candidate_model_version"] or "pending"),
                "candidate_dataset_version": str(result["dataset_version"]),
                "production_model_version": str(cycle["production_model_version"]),
                "training_config_version": _CLASSIFIER_TRAINING_CONFIG_VERSION,
                "dataset_uri": str(result["artifact_uri"]),
                "dataset_checksum": str(result["content_sha256"]),
                "dataset_manifest_uri": str(result["manifest_uri"]),
                "dataset_manifest_sha256": str(result["manifest_sha256"]),
                "output_artifact_uri": candidate_output_artifact_uri(
                    str(request_payload["cycle_id"]),
                    str(cycle["candidate_model_version"] or "pending"),
                ),
                "min_samples": int(cycle["min_feedback_count"]),
            }
            validate_document(training_payload, "CandidateTrainingJob")
            await connection.execute(
                """
                INSERT INTO background_jobs (job_type, payload, state)
                VALUES ('TRAIN_CLASSIFIER', $1::jsonb, 'QUEUED')
                """,
                json.dumps(training_payload),
            )
            await connection.execute(
                """
                UPDATE background_jobs
                SET state = 'COMPLETED',
                    finished_at = now(),
                    payload = payload || jsonb_build_object('result', $2::jsonb)
                WHERE id = $1
                """,
                build_job_id,
                json.dumps(result, ensure_ascii=False),
            )


async def fail_candidate_dataset_build(
    pool: Any,
    job_id: int,
    payload: dict[str, Any],
    error_code: str,
) -> None:
    """Expose dataset-build failures on both the job and its learning cycle."""

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                UPDATE background_jobs
                SET state = 'FAILED', finished_at = now(), error = $2
                WHERE id = $1
                """,
                job_id,
                error_code[:4000],
            )
            await connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'DATASET_BUILD_FAILED', decision_note = $2, updated_at = now()
                WHERE (cycle_id = $1 OR id::text = $1) AND state = 'TRAINING'
                """,
                str(payload.get("cycle_id") or ""),
                error_code[:4000],
            )


def default_qdrant_collection(embedder_version: str, dimension: int) -> str:
    if embedder_version == "embedder-demo-2026-09-21-001" and dimension == 32:
        return "pulse109_tickets_v1"
    suffix = "".join(character.lower() if character.isalnum() else "_" for character in embedder_version).strip("_")
    return f"pulse109_{suffix or 'embedder'}_d{dimension}"


def qdrant_distance_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "cosine":
        return "Cosine"
    if normalized == "dot":
        return "Dot"
    if normalized in {"euclid", "euclidean"}:
        return "Euclid"
    raise ValueError("unsupported vector distance metric")


def ml_distance_name(value: str) -> str:
    return {
        "Cosine": "cosine",
        "Dot": "dot",
        "Euclid": "euclidean",
    }[qdrant_distance_name(value)]


def staging_collection_name(collection_base: str, job_id: int) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in collection_base
    ).strip("_")[:32]
    prefix = "" if slug.startswith("pulse109_") else "pulse109_"
    return f"{prefix}{slug or 'embedder'}_job_{job_id}"


async def qdrant_vector_config(client: Any, qdrant_url: str, collection: str) -> tuple[int, str]:
    response = await client.get(f"{qdrant_url}/collections/{collection}")
    response.raise_for_status()
    vectors = response.json().get("result", {}).get("config", {}).get("params", {}).get("vectors")
    if not isinstance(vectors, dict) or "size" not in vectors or "distance" not in vectors:
        raise RuntimeError("Qdrant collection does not have a single unnamed vector")
    return int(vectors["size"]), qdrant_distance_name(str(vectors["distance"]))


async def create_staging_collection(
    client: Any,
    qdrant_url: str,
    collection: str,
    dimension: int,
    distance_metric: str,
) -> None:
    check = await client.get(f"{qdrant_url}/collections/{collection}")
    if check.status_code == 200:
        raise RuntimeError("Qdrant staging collection already exists")
    if check.status_code != 404:
        check.raise_for_status()
    create = await client.put(
        f"{qdrant_url}/collections/{collection}",
        json={"vectors": {"size": dimension, "distance": distance_metric}},
    )
    create.raise_for_status()
    actual_dimension, actual_distance = await qdrant_vector_config(client, qdrant_url, collection)
    if actual_dimension != dimension or actual_distance != distance_metric:
        raise RuntimeError("Qdrant staging collection config does not match embedder")


async def index_rows(
    client: Any,
    qdrant_url: str,
    ml_service_url: str,
    collection: str,
    dimension: int,
    embedder_version: str,
    rows: list[Any],
) -> int:
    if not rows:
        return 0
    embedding_response = await client.post(
        f"{ml_service_url}/internal/v1/embed",
        json={
            "texts": [str(row["original_text"]) for row in rows],
            "dimension": dimension,
            "normalize": True,
            "model_version": embedder_version,
        },
    )
    embedding_response.raise_for_status()
    result = embedding_response.json()
    if result.get("model_version") != embedder_version:
        raise RuntimeError("ML reindex returned a different model version")
    if result.get("dimension") != dimension:
        raise RuntimeError("ML reindex returned a different embedding dimension")
    embeddings = result.get("embeddings") or []
    if len(embeddings) != len(rows):
        raise RuntimeError("ML reindex returned a different embedding count")
    points = []
    for row, vector in zip(rows, embeddings):
        if len(vector) != dimension:
            raise RuntimeError("ML reindex returned an invalid embedding dimension")
        points.append(
            {
                "id": int(row["id"]),
                "vector": vector,
                "payload": {
                    "ticket_id": str(row["id"]),
                    "topic_id": str(row["topic_id"] or "unknown"),
                    "region_id": str(row["region_id"] or "unknown"),
                    "created_at": row["created_at"].isoformat(),
                },
            }
        )
    upsert = await client.put(
        f"{qdrant_url}/collections/{collection}/points?wait=true",
        json={"points": points},
    )
    upsert.raise_for_status()
    return len(points)


async def reindex_qdrant(pool: Any, payload: dict[str, Any], job_id: int) -> dict[str, Any]:
    """Build a versioned collection, then switch PostgreSQL lineage atomically."""

    import httpx

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    ml_service_url = os.environ.get("ML_SERVICE_URL", "http://ml-service:8000").rstrip("/")
    dimension = int(payload.get("embedding_dimension") or os.environ.get("EMBEDDING_DIMENSION", "32"))
    embedder_version = str(payload.get("embedder_version") or os.environ.get("EMBEDDER_VERSION", "baseline"))
    collection_base = str(
        payload.get("collection_base")
        or payload.get("collection")
        or os.environ.get("QDRANT_COLLECTION", "").strip()
        or default_qdrant_collection(embedder_version, dimension)
    )
    distance_metric = qdrant_distance_name(
        str(payload.get("distance_metric") or os.environ.get("EMBEDDING_DISTANCE_METRIC", "cosine"))
    )
    if not 8 <= dimension <= 1024 or not embedder_version.strip():
        raise ValueError("invalid target embedder configuration")
    collection = staging_collection_name(collection_base, job_id)
    max_ticket_id = int(await pool.fetchval("SELECT COALESCE(MAX(id), 0) FROM tickets"))

    async with httpx.AsyncClient(timeout=60.0) as client:
        metadata_response = await client.get(
            f"{ml_service_url}/internal/v1/models/embedder"
        )
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        if (
            metadata.get("model_version") != embedder_version
            or metadata.get("dimension") != dimension
            or metadata.get("distance_metric") != ml_distance_name(distance_metric)
        ):
            raise RuntimeError("ML embedder metadata does not match reindex target")

        indexed_rows = 0
        cursor = 0
        while True:
            batch = await pool.fetch(
                "SELECT id, original_text, topic_id, region_id, created_at FROM tickets WHERE original_text IS NOT NULL AND id > $1 AND id <= $2 ORDER BY id LIMIT 32",
                cursor,
                max_ticket_id,
            )
            if not batch:
                break
            cursor = int(batch[-1]["id"])
            if indexed_rows == 0:
                await create_staging_collection(
                    client, qdrant_url, collection, dimension, distance_metric
                )
            indexed_rows += await index_rows(
                client,
                qdrant_url,
                ml_service_url,
                collection,
                dimension,
                embedder_version,
                list(batch),
            )

        if indexed_rows == 0:
            await create_staging_collection(
                client, qdrant_url, collection, dimension, distance_metric
            )

        async with pool.acquire() as connection:
            async with connection.transaction():
                active = await connection.fetchrow(
                    "SELECT embedder_version, embedding_dimension, distance_metric, collection_name, generation FROM vector_index_state WHERE singleton_id = 1 FOR UPDATE"
                )
                if active is None:
                    raise RuntimeError("active vector index is not configured")
                last_id = max_ticket_id
                while True:
                    tail = await connection.fetch(
                        "SELECT id, original_text, topic_id, region_id, created_at FROM tickets WHERE original_text IS NOT NULL AND id > $1 ORDER BY id LIMIT 32",
                        last_id,
                    )
                    if not tail:
                        break
                    last_id = int(tail[-1]["id"])
                    indexed_rows += await index_rows(
                        client,
                        qdrant_url,
                        ml_service_url,
                        collection,
                        dimension,
                        embedder_version,
                        list(tail),
                    )
                expected_rows = int(
                    await connection.fetchval(
                        "SELECT COUNT(*) FROM tickets WHERE original_text IS NOT NULL"
                    )
                )
                if expected_rows != indexed_rows:
                    raise RuntimeError("ticket count changed during vector reindex")
                previous_collection = str(active["collection_name"])
                await connection.execute(
                    "UPDATE tickets SET embedding_ref = 'qdrant:' || $1 || ':' || id::text, model_versions = jsonb_set(model_versions, '{embedder}', to_jsonb($2::text), true), updated_in_pulse_at = now() WHERE original_text IS NOT NULL",
                    collection,
                    embedder_version,
                )
                await connection.execute(
                    "UPDATE vector_index_state SET embedder_version = $1, embedding_dimension = $2, distance_metric = $3, collection_name = $4, generation = generation + 1, updated_at = now() WHERE singleton_id = 1",
                    embedder_version,
                    dimension,
                    distance_metric,
                    collection,
                )

    return {
        "collection": collection,
        "previous_collection": previous_collection,
        "embedder_version": embedder_version,
        "embedding_dimension": dimension,
        "distance_metric": distance_metric,
        "indexed_rows": indexed_rows,
        "source": "postgres+ml+qdrant",
    }


async def process_job(pool: Any, job: Any) -> None:
    job_id = int(job["id"])
    kind = str(job["job_type"]).lower()
    payload: dict[str, Any] = {}
    try:
        raw_payload = job["payload"] or {}
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)
        payload = dict(raw_payload)
        if kind == "reindex_qdrant":
            result = await reindex_qdrant(pool, payload, job_id)
            await complete_job(pool, job_id, result)
            return
        if kind == "build_candidate_dataset":
            artifact_dir = Path(os.environ.get("MODEL_DIR", "/app/trained-artifacts"))
            builder_request = await export_validated_feedback(
                pool,
                payload,
                artifact_dir / "datasets",
            )
            result = await build_candidate_dataset(pool, builder_request, artifact_dir)
            await persist_candidate_dataset_and_queue_training(
                pool,
                job_id,
                payload,
                result,
            )
            return

        _, _, _, _, _, trainer, evaluator = make_services()
        if kind == "candidate_evaluation":
            result = await evaluate_candidate_cycle(pool, payload, evaluator)
            await complete_job(pool, job_id, result)
            return
        if kind == "train_classifier":
            if test_fake_trainer_requested():
                if not test_fake_trainer_enabled():
                    raise RuntimeError("TEST_FAKE_TRAINER_NOT_ALLOWED")
                fake_payload = {
                    "model_type": "classifier",
                    "dataset_version": payload.get("candidate_dataset_version", "test-only"),
                    "samples": [
                        {"text": "test-only synthetic sample", "label": "unknown"}
                    ],
                    "min_samples": 1,
                    "candidate_model_version": payload.get("candidate_model_version"),
                }
                result = trainer.train(
                    TrainingRequest.model_validate(fake_payload)
                ).model_dump(mode="json")
                result["candidate_model_version"] = str(
                    payload.get("candidate_model_version") or result.get("candidate_model_version")
                )
                result.setdefault("manifest", {})["model_version"] = result[
                    "candidate_model_version"
                ]
                result["manifest"]["implementation"] = "test-fake-trainer"
                result["manifest"]["synthetic"] = True
            else:
                validate_document(payload, "CandidateTrainingJob")
                training_job = CandidateTrainingJob.model_validate(payload)
                result = trainer.train_candidate(training_job).model_dump(mode="json")
                if result["status"] != "COMPLETED":
                    raise RuntimeError("CANDIDATE_TRAINING_FAILED")
            kind = "training"
        elif kind in {"training", "train"}:
            result = trainer.train(TrainingRequest.model_validate(payload)).model_dump(mode="json")
            if result["state"] == "TRAINER_NOT_CONFIGURED":
                raise RuntimeError(result["state"])
            if payload.get("candidate_model_version"):
                result["candidate_model_version"] = str(payload["candidate_model_version"])
                result.setdefault("manifest", {})["model_version"] = str(payload["candidate_model_version"])
        elif kind in {"evaluation", "evaluate"}:
            result = evaluator.evaluate(EvaluationRequest.model_validate(payload)).model_dump(mode="json")
        else:
            raise ValueError(f"unsupported job_type: {job['job_type']}")
        await complete_job(pool, job_id, result)
        if kind == "training":
            await update_learning_cycle(pool, payload, result=result)
    except Exception as exc:
        error_code = safe_job_error(exc)
        if kind == "build_candidate_dataset":
            await fail_candidate_dataset_build(pool, job_id, payload, error_code)
        elif kind == "candidate_evaluation":
            await fail_job(pool, job_id, error_code)
        else:
            await fail_job(pool, job_id, error_code)
            await update_learning_cycle(pool, payload, error=error_code)


async def run_worker() -> None:
    import asyncpg

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the PostgreSQL worker")
    poll_interval = float(os.environ.get("WORKER_POLL_INTERVAL", "1.0"))
    schedule_config = learning_cycle_schedule_config_from_env()
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=4)
    loop = asyncio.get_running_loop()
    next_schedule_check = 0.0
    try:
        while True:
            await advance_expired_learning_cycle(pool)
            current_time = loop.time()
            if schedule_config.enabled and current_time >= next_schedule_check:
                schedule_status = await start_next_recurring_learning_cycle(
                    pool,
                    schedule_config,
                )
                if schedule_status == "STARTED":
                    _LOGGER.info("automatic recurring learning cycle started")
                elif schedule_status.startswith("BLOCKED_"):
                    _LOGGER.warning(
                        "automatic recurring learning cycle not started: %s",
                        schedule_status,
                    )
                next_schedule_check = loop.time() + _LEARNING_SCHEDULER_CHECK_SECONDS
            job = await claim_job(pool)
            if job is None:
                await asyncio.sleep(poll_interval)
                continue
            await process_job(pool, job)
    finally:
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pulse 109 ML jobs")
    parser.add_argument("--kind", choices=("training", "evaluation"), help="override stdin job kind")
    parser.add_argument("--loop", action="store_true", help="poll PostgreSQL background_jobs")
    args = parser.parse_args()
    if args.loop:
        try:
            asyncio.run(run_worker())
        except KeyboardInterrupt:
            return 0
        return 0
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # pragma: no cover - CLI failure path
        json.dump({"error": safe_job_error(exc)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1
    return run_stdin_job(args.kind, payload)


if __name__ == "__main__":
    raise SystemExit(main())
