from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json

import pytest

from worker import safe_job_error, update_learning_cycle


class TrainingConnection:
    def __init__(self, returned: list[object]) -> None:
        self.returned = iter(returned)
        self.queries: list[tuple[str, tuple]] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchval(self, query: str, *values):
        self.queries.append((query, values))
        return next(self.returned)

    async def execute(self, query: str, *values):
        self.queries.append((query, values))


class TrainingPool:
    def __init__(self, returned: list[object]) -> None:
        self.connection = TrainingConnection(returned)

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def test_success_registers_dataset_and_new_candidate_without_overwriting(monkeypatch) -> None:
    monkeypatch.delenv("PULSE_TEST_FAKE_TRAINER", raising=False)
    pool = TrainingPool(["dataset_v1", "candidate_v1", 1])
    result = {
        "state": "COMPLETED", "candidate_model_version": "candidate_v1",
        "dataset_version": "dataset_v1", "dataset_manifest_uri": "/private/dataset/manifest.json",
        "dataset_manifest_sha256": "sha256:" + "a" * 64,
        "dataset_content_sha256": "sha256:" + "b" * 64,
        "sample_count": 2,
        "manifest": {"artifact_uri": "/private/model/manifest.json", "artifact_checksum": "sha256:" + "c" * 64},
        "offline_metrics": {"report_version": "classifier-pair-evaluation.v1",
                            "frozen_evaluation_version": "eval_v1", "dataset_version": "reviewed_v1",
                            "regressed_critical_topics": [], "decision": "PENDING_HUMAN_REVIEW"},
    }
    asyncio.run(update_learning_cycle(pool, {"cycle_id": "1"}, result=result))
    queries = pool.connection.queries
    assert len(queries) == 4
    assert "manifest_sha256 = 'pending'" in queries[0][0]
    assert "ON CONFLICT (model_version) DO NOTHING" in queries[1][0]
    assert queries[1][1][0] == "candidate_v1"
    assert queries[1][1][3] == "/private/model/manifest.json"
    assert "INSERT INTO model_evaluations" in queries[2][0]
    assert "'{}'::jsonb" in queries[2][0]
    assert json.loads(queries[2][1][4])["decision"] == "PENDING_HUMAN_REVIEW"
    assert "state = 'EVALUATE'" in queries[3][0]
    assert queries[3][1][1] == "CANDIDATE_TRAINED"


def test_version_collision_fails_without_changing_cycle() -> None:
    pool = TrainingPool(["dataset_v1", None])
    result = {
        "state": "COMPLETED", "candidate_model_version": "candidate_v1",
        "dataset_version": "dataset_v1", "dataset_manifest_uri": "/private/dataset/manifest.json",
        "dataset_manifest_sha256": "sha256:" + "a" * 64,
        "dataset_content_sha256": "sha256:" + "b" * 64,
        "sample_count": 2, "manifest": {"artifact_uri": "/private/model/manifest.json"},
    }
    with pytest.raises(RuntimeError, match="CANDIDATE_VERSION_EXISTS") as caught:
        asyncio.run(update_learning_cycle(pool, {"cycle_id": "1"}, result=result))
    assert safe_job_error(caught.value) == "CANDIDATE_VERSION_EXISTS"
    assert len(pool.connection.queries) == 2


def test_insufficient_feedback_closes_cycle_without_candidate() -> None:
    pool = TrainingPool([1])
    asyncio.run(update_learning_cycle(pool, {"cycle_id": "1"},
                                      result={"state": "INSUFFICIENT_FEEDBACK", "sample_count": 0}))
    assert len(pool.connection.queries) == 1
    assert "state = 'INSUFFICIENT_FEEDBACK'" in pool.connection.queries[0][0]
