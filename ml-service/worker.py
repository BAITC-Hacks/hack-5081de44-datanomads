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
from typing import Any

from pydantic import ValidationError

from app.schemas import EvaluationRequest, TrainingRequest
from app.services import make_services


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
    if isinstance(error, RuntimeError) and str(error) == "TRAINER_NOT_CONFIGURED":
        return "TRAINER_NOT_CONFIGURED"
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
    try:
        raw_payload = job["payload"] or {}
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)
        payload = dict(raw_payload)
        _, _, _, _, _, trainer, evaluator = make_services()
        if kind == "reindex_qdrant":
            result = await reindex_qdrant(pool, payload, job_id)
            await complete_job(pool, job_id, result)
            return
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
