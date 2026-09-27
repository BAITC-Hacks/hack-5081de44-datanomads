"""PostgreSQL-backed ML background worker.

The worker claims queued rows with FOR UPDATE SKIP LOCKED so multiple
instances can safely process the same queue. Training/evaluation remains the
deterministic baseline for now; the durable job lifecycle is real and ready
for versioned model artifacts later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import EvaluationRequest, TrainingRequest
from app.candidate_dataset import build_candidate_dataset, export_validated_feedback
from app.services import make_services

_CANDIDATE_DATASET_SCHEMA_VERSION = "candidate-training-dataset.v1"
_CLASSIFIER_TRAINING_CONFIG_VERSION = "classifier-training.v1"


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
    """Close one expired cycle and queue dataset orchestration above its gate."""

    async with pool.acquire() as connection:
        async with connection.transaction():
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
        "SOURCE_TICKET_UNAVAILABLE",
        "SOURCE_TICKET_TEXT_UNAVAILABLE",
        "INSUFFICIENT_CANDIDATE_DATASET",
        "FEEDBACK_EXPORT_CHECKSUM_MISMATCH",
        "FEEDBACK_EXPORT_LINEAGE_MISMATCH",
        "CANDIDATE_DATASET_VERSION_CONFLICT",
        "LEARNING_CYCLE_NOT_FOUND",
        "INVALID_CYCLE_ID",
        "INVALID_FEEDBACK_TIMESTAMP",
    }
    if isinstance(error, RuntimeError) and str(error) in safe_runtime_errors:
        return str(error)
    if isinstance(error, (ValidationError, json.JSONDecodeError)):
        return "INVALID_JOB_PAYLOAD"
    return "JOB_FAILED"


async def update_learning_cycle(pool: Any, payload: dict[str, Any], result: dict[str, Any] | None = None, error: str | None = None) -> None:
    cycle_id = payload.get("cycle_id")
    if not cycle_id:
        return
    if error:
        await pool.execute(
            "UPDATE learning_cycles SET state = 'EVALUATE', decision_note = $2, updated_at = now() WHERE cycle_id = $1 OR id::text = $1",
            str(cycle_id),
            error[:4000],
        )
        return
    result = result or {}
    candidate = result.get("candidate_model_version")
    manifest = result.get("manifest") or {}
    if candidate:
        await pool.execute(
            "INSERT INTO model_versions (model_version, model_family, dataset_version, status, manifest_uri, artifact_checksum) VALUES ($1, 'classifier-baseline-candidate', $2, 'CANDIDATE', $3, $4) ON CONFLICT (model_version) DO UPDATE SET status = 'CANDIDATE', manifest_uri = EXCLUDED.manifest_uri, artifact_checksum = EXCLUDED.artifact_checksum",
            str(candidate),
            str(result.get("dataset_version") or payload.get("dataset_version") or "unknown"),
            json.dumps(manifest.get("artifact_uri")) if manifest.get("artifact_uri") else None,
            manifest.get("artifact_checksum"),
        )
    await pool.execute(
        "UPDATE learning_cycles SET state = 'EVALUATE', updated_at = now(), decision_note = $2 WHERE cycle_id = $1 OR id::text = $1",
        str(cycle_id),
        "FAKE_TRAINER_COMPLETED" if os.environ.get("PULSE_TEST_FAKE_TRAINER", "false").lower() in {"1", "true", "yes"} else "TRAINER_NOT_CONFIGURED",
    )


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
                "kind": "training",
                "cycle_id": str(request_payload["cycle_id"]),
                "candidate_model_version": str(cycle["candidate_model_version"] or "pending"),
                "model_type": "classifier",
                "dataset_version": str(result["dataset_version"]),
                "dataset_uri": str(result["artifact_uri"]),
                "dataset_checksum": str(result["content_sha256"]),
                "dataset_manifest_uri": str(result["manifest_uri"]),
                "dataset_manifest_sha256": str(result["manifest_sha256"]),
                "production_model_version": str(cycle["production_model_version"]),
                "training_config_version": _CLASSIFIER_TRAINING_CONFIG_VERSION,
                "min_samples": int(cycle["min_feedback_count"]),
            }
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
        if kind == "train_classifier":
            if os.environ.get("PULSE_TEST_FAKE_TRAINER", "false").lower() not in {"1", "true", "yes"}:
                raise RuntimeError("TRAINER_NOT_CONFIGURED")
            # The fake trainer is an integration-test adapter only.  It creates
            # a deterministic candidate artifact from one synthetic sample and
            # is never enabled by the normal Compose profile.
            if not payload.get("samples"):
                payload["samples"] = [{"text": "test-only fake sample", "label": "unknown", "topic_id": "unknown"}]
            payload["min_samples"] = 1
            kind = "training"
        if kind in {"training", "train"}:
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
        else:
            await fail_job(pool, job_id, error_code)
            await update_learning_cycle(pool, payload, error=error_code)


async def run_worker() -> None:
    import asyncpg

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the PostgreSQL worker")
    poll_interval = float(os.environ.get("WORKER_POLL_INTERVAL", "1.0"))
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=4)
    try:
        while True:
            await advance_expired_learning_cycle(pool)
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
