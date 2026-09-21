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


async def process_job(pool: Any, job: Any) -> None:
    job_id = int(job["id"])
    kind = str(job["job_type"]).lower()
    try:
        raw_payload = job["payload"] or {}
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)
        payload = dict(raw_payload)
        _, _, _, _, _, trainer, evaluator = make_services()
        if kind in {"training", "train"}:
            result = trainer.train(TrainingRequest.model_validate(payload)).model_dump(mode="json")
        elif kind in {"evaluation", "evaluate"}:
            result = evaluator.evaluate(EvaluationRequest.model_validate(payload)).model_dump(mode="json")
        else:
            raise ValueError(f"unsupported job_type: {job['job_type']}")
        await complete_job(pool, job_id, result)
    except Exception as exc:
        await fail_job(pool, job_id, str(exc))


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
