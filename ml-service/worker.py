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

from app.schemas import EvaluationRequest, TrainingRequest
from app.services import make_services


def run_stdin_job(kind: str | None, payload: dict[str, Any]) -> int:
    selected_kind = kind or payload.pop("kind", "evaluation")
    _, _, _, _, _, trainer, evaluator = make_services()
    try:
        if selected_kind == "training":
            result = trainer.train(TrainingRequest.model_validate(payload))
        elif selected_kind == "evaluation":
            result = evaluator.evaluate(EvaluationRequest.model_validate(payload))
        else:
            raise ValueError(f"unsupported job kind: {selected_kind}")
        json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # pragma: no cover - CLI failure path
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
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


async def _qdrant_existing_ids(client: Any, qdrant_url: str, collection: str) -> list[int]:
    """Read all point ids so a reindex can remove stale vectors safely."""

    point_ids: list[int] = []
    offset: Any | None = None
    while True:
        body: dict[str, Any] = {"limit": 1000, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        response = await client.post(f"{qdrant_url}/collections/{collection}/points/scroll", json=body)
        response.raise_for_status()
        result = response.json().get("result") or {}
        for point in result.get("points") or []:
            value = point.get("id")
            try:
                point_ids.append(int(value))
            except (TypeError, ValueError):
                continue
        offset = result.get("next_page_offset")
        if offset is None:
            return point_ids


def default_qdrant_collection(embedder_version: str, dimension: int) -> str:
    if embedder_version == "embedder-demo-2026-09-21-001" and dimension == 32:
        return "pulse109_tickets_v1"
    suffix = "".join(character.lower() if character.isalnum() else "_" for character in embedder_version).strip("_")
    return f"pulse109_{suffix or 'embedder'}_d{dimension}"


async def reindex_qdrant(pool: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the configured Qdrant collection from PostgreSQL source rows."""

    import httpx

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    ml_service_url = os.environ.get("ML_SERVICE_URL", "http://ml-service:8000").rstrip("/")
    dimension = int(payload.get("embedding_dimension") or os.environ.get("EMBEDDING_DIMENSION", "32"))
    embedder_version = str(payload.get("embedder_version") or os.environ.get("EMBEDDER_VERSION", "baseline"))
    collection = str(payload.get("collection") or os.environ.get("QDRANT_COLLECTION", "").strip() or default_qdrant_collection(embedder_version, dimension))
    rows = await pool.fetch(
        "SELECT id, original_text, topic_id, region_id, created_at FROM tickets WHERE original_text IS NOT NULL ORDER BY id"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        collection_response = await client.get(f"{qdrant_url}/collections/{collection}")
        if collection_response.status_code == 404:
            create_response = await client.put(
                f"{qdrant_url}/collections/{collection}",
                json={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
            create_response.raise_for_status()
        else:
            collection_response.raise_for_status()

        existing_ids = await _qdrant_existing_ids(client, qdrant_url, collection)
        deleted_points = 0
        for start in range(0, len(existing_ids), 500):
            chunk = existing_ids[start : start + 500]
            if not chunk:
                continue
            response = await client.post(
                f"{qdrant_url}/collections/{collection}/points/delete?wait=true",
                json={"points": chunk},
            )
            response.raise_for_status()
            deleted_points += len(chunk)

        indexed_rows = 0
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            embedding_response = await client.post(
                f"{ml_service_url}/internal/v1/embed",
                json={
                    "texts": [str(row["original_text"]) for row in batch],
                    "dimension": dimension,
                    "normalize": True,
                    "model_version": embedder_version,
                },
            )
            embedding_response.raise_for_status()
            embeddings = embedding_response.json().get("embeddings") or []
            if len(embeddings) != len(batch):
                raise RuntimeError("ML reindex returned a different embedding count")
            points = []
            for row, vector in zip(batch, embeddings):
                if len(vector) != dimension:
                    raise RuntimeError(f"ML reindex returned dimension {len(vector)}, expected {dimension}")
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
            upsert_response = await client.put(
                f"{qdrant_url}/collections/{collection}/points?wait=true",
                json={"points": points},
            )
            upsert_response.raise_for_status()
            await pool.executemany(
                "UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1",
                [
                    (
                        int(row["id"]),
                        f"qdrant:{collection}:{int(row['id'])}",
                        json.dumps({"embedder": embedder_version}),
                    )
                    for row in batch
                ],
            )
            indexed_rows += len(batch)

    return {
        "collection": collection,
        "embedder_version": embedder_version,
        "deleted_points": deleted_points,
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
            result = await reindex_qdrant(pool, payload)
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
        await fail_job(pool, job_id, str(exc))
        await update_learning_cycle(pool, payload, error=str(exc))


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
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1
    return run_stdin_job(args.kind, payload)


if __name__ == "__main__":
    raise SystemExit(main())
