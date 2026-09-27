from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from worker import advance_expired_learning_cycle


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


def _cycle(feedback_count: int) -> dict[str, Any]:
    return {
        "id": 12,
        "cycle_id": "cycle-12",
        "min_feedback_count": 3,
        "candidate_dataset_version": None,
        "candidate_model_version": "candidate-12",
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


    def test_expired_cycle_at_minimum_queues_training(self) -> None:
        pool = _Pool(_cycle(feedback_count=3))

        self.assertTrue(asyncio.run(advance_expired_learning_cycle(pool)))

        self.assertEqual(len(pool.connection.executed), 2)
        self.assertIn("state = 'TRAINING'", pool.connection.executed[0][0])
        insert_query, (payload_json,) = pool.connection.executed[1]
        self.assertIn("INSERT INTO background_jobs", insert_query)
        payload = json.loads(payload_json)
        self.assertEqual(payload["cycle_id"], "cycle-12")
        self.assertEqual(payload["candidate_model_version"], "candidate-12")
        self.assertEqual(payload["dataset_version"], "pending")


    def test_worker_leaves_future_cycles_untouched(self) -> None:
        pool = _Pool(None)

        self.assertFalse(asyncio.run(advance_expired_learning_cycle(pool)))
        self.assertEqual(pool.connection.executed, [])
