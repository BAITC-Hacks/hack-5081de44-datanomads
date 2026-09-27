from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from worker import (
    advance_expired_learning_cycle,
    fail_candidate_dataset_build,
    persist_candidate_dataset_and_queue_training,
)


class _AsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> None:
        return None


class _AcquireContext:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def __aenter__(self) -> Any:
        return self.connection

    async def __aexit__(self, *_: Any) -> None:
        return None


class _Connection:
    def __init__(self, cycle: dict[str, Any] | None) -> None:
        self.cycle = cycle
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchrow(self, query: str) -> dict[str, Any] | None:
        assert "collect_ends_at <= now()" in query
        return self.cycle

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "OK"


class _Pool:
    def __init__(self, cycle: dict[str, Any] | None) -> None:
        self.connection = _Connection(cycle)

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)


class _PersistenceConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchrow(self, query: str, *_: Any) -> dict[str, Any] | None:
        if "FROM learning_cycles" in query:
            return {
                "id": 12,
                "state": "TRAINING",
                "production_model_version": "production-12",
                "candidate_model_version": "candidate-12",
                "frozen_evaluation_dataset_version": "evaluation-v1",
                "candidate_dataset_version": None,
                "min_feedback_count": 1,
            }
        if "FROM dataset_versions" in query:
            return {
                "schema_version": "candidate-training-dataset.v1",
                "manifest_uri": "file:///artifacts/dataset.manifest.json",
                "manifest_sha256": "b" * 64,
                "content_sha256": "a" * 64,
                "is_synthetic": False,
                "record_count": 1,
            }
        raise AssertionError("unexpected candidate dataset persistence query")

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        if "UPDATE learning_cycles" in query:
            return "UPDATE 1"
        return "OK"


class _PersistencePool:
    def __init__(self) -> None:
        self.connection = _PersistenceConnection()

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)


def _cycle(feedback_count: int) -> dict[str, Any]:
    return {
        "id": 12,
        "cycle_id": "cycle-12",
        "min_feedback_count": 3,
        "candidate_model_version": "candidate-12",
        "production_model_version": "production-12",
        "frozen_evaluation_dataset_version": "evaluation-v1",
        "feedback_ids": ["21", "22", "23"][:feedback_count],
        "frozen_evaluation_ticket_ids": ["900", "901"],
        "feedback_count": feedback_count,
    }


class LearningCycleWorkerTests(unittest.TestCase):
    def test_expired_cycle_below_minimum_closes_without_training_job(self) -> None:
        pool = _Pool(_cycle(feedback_count=2))

        self.assertTrue(asyncio.run(advance_expired_learning_cycle(pool)))

        self.assertEqual(len(pool.connection.executed), 1)
        self.assertIn("INSUFFICIENT_FEEDBACK", pool.connection.executed[0][0])
        self.assertFalse(
            any("INSERT INTO background_jobs" in query for query, _ in pool.connection.executed)
        )


    def test_expired_cycle_at_minimum_queues_dataset_build(self) -> None:
        pool = _Pool(_cycle(feedback_count=3))

        self.assertTrue(asyncio.run(advance_expired_learning_cycle(pool)))

        self.assertEqual(len(pool.connection.executed), 2)
        self.assertIn("state = 'TRAINING'", pool.connection.executed[0][0])
        insert_query, (payload_json,) = pool.connection.executed[1]
        self.assertIn("BUILD_CANDIDATE_DATASET", insert_query)
        payload = json.loads(payload_json)
        self.assertEqual(payload["cycle_id"], "cycle-12")
        self.assertEqual(payload["candidate_model_version"], "candidate-12")
        self.assertEqual(payload["production_model_version"], "production-12")
        self.assertEqual(payload["feedback_ids"], ["21", "22", "23"])
        self.assertEqual(payload["frozen_evaluation_dataset_version"], "evaluation-v1")
        self.assertEqual(payload["frozen_evaluation_ticket_ids"], ["900", "901"])
        self.assertNotIn("samples", payload)


    def test_worker_leaves_future_cycles_untouched(self) -> None:
        pool = _Pool(None)

        self.assertFalse(asyncio.run(advance_expired_learning_cycle(pool)))
        self.assertEqual(pool.connection.executed, [])

    def test_candidate_dataset_lineage_and_training_job_are_committed_together(self) -> None:
        pool = _PersistencePool()
        result = {
            "dataset_version": "candidate-dataset-cycle-12-aaaaaaaaaaaaaaaa",
            "artifact_uri": "file:///artifacts/dataset.jsonl",
            "manifest_uri": "file:///artifacts/dataset.manifest.json",
            "manifest_sha256": "b" * 64,
            "content_sha256": "a" * 64,
            "record_count": 1,
            "training_ticket_ids": ["101"],
            "synthetic": False,
            "production_model_version": "production-12",
            "candidate_model_version": "candidate-12",
            "frozen_evaluation_dataset_version": "evaluation-v1",
        }

        asyncio.run(
            persist_candidate_dataset_and_queue_training(
                pool,
                44,
                {
                    "cycle_id": "cycle-12",
                    "frozen_evaluation_dataset_version": "evaluation-v1",
                },
                result,
            )
        )

        queries = [query for query, _ in pool.connection.executed]
        self.assertTrue(any("INSERT INTO dataset_versions" in query for query in queries))
        self.assertTrue(any("INSERT INTO dataset_ticket_links" in query for query in queries))
        self.assertTrue(any("candidate_dataset_checksum" in query for query in queries))
        training_args = next(
            args
            for query, args in pool.connection.executed
            if "INSERT INTO background_jobs" in query
        )
        payload = json.loads(training_args[0])
        self.assertEqual(payload["dataset_version"], result["dataset_version"])
        self.assertEqual(payload["dataset_checksum"], "a" * 64)
        self.assertEqual(payload["dataset_uri"], result["artifact_uri"])
        self.assertNotIn("samples", payload)
        self.assertNotIn("Не работает насосная станция", json.dumps(payload))
        self.assertTrue(any("WHERE id = $1" in query and "result" in query for query in queries))

    def test_candidate_dataset_build_failure_is_recorded_on_job_and_cycle(self) -> None:
        pool = _PersistencePool()

        asyncio.run(
            fail_candidate_dataset_build(
                pool,
                44,
                {"cycle_id": "cycle-12"},
                "FROZEN_EVALUATION_SET_NOT_CONFIGURED",
            )
        )

        self.assertEqual(len(pool.connection.executed), 2)
        self.assertIn("state = 'FAILED'", pool.connection.executed[0][0])
        self.assertIn("FROZEN_EVALUATION_SET_NOT_CONFIGURED", pool.connection.executed[0][1])
        self.assertIn("state = 'DATASET_BUILD_FAILED'", pool.connection.executed[1][0])
        self.assertEqual(pool.connection.executed[1][1][0], "cycle-12")
