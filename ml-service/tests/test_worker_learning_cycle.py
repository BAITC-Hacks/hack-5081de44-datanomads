from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from app.services import make_services
from worker import (
    LearningCycleScheduleConfig,
    advance_expired_learning_cycle,
    evaluate_candidate_cycle,
    fail_candidate_dataset_build,
    learning_cycle_schedule_config_from_env,
    persist_candidate_dataset_and_queue_training,
    start_next_recurring_learning_cycle,
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
    def __init__(
        self,
        cycle: dict[str, Any] | None,
        evaluation_cycle: dict[str, Any] | None = None,
    ) -> None:
        self.cycle = cycle
        self.evaluation_cycle = evaluation_cycle
        self.fetchrow_calls: list[str] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchrow(self, query: str) -> dict[str, Any] | None:
        self.fetchrow_calls.append(query)
        if "WITH expired AS" in query:
            assert "state = 'EVALUATE' AND evaluation_ends_at <= now()" in query
            return self.evaluation_cycle
        assert "collect_ends_at <= now()" in query
        return self.cycle

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "OK"


class _Pool:
    def __init__(
        self,
        cycle: dict[str, Any] | None,
        evaluation_cycle: dict[str, Any] | None = None,
    ) -> None:
        self.connection = _Connection(cycle, evaluation_cycle)

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)


class _SchedulerConnection:
    def __init__(
        self,
        latest_state: str | None = None,
        dataset: dict[str, Any] | None = None,
        dataset_lookup_missing: bool = False,
        production_model_version: str | None = "production-v1",
    ) -> None:
        self.latest_state = latest_state
        self.dataset = dataset or {"is_synthetic": False, "linked_ticket_count": 4}
        self.dataset_lookup_missing = dataset_lookup_missing
        self.production_model_version = production_model_version
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        if "ORDER BY created_at DESC, id DESC" in query:
            return {"state": self.latest_state} if self.latest_state else None
        if "FROM dataset_versions" in query:
            return None if self.dataset_lookup_missing else self.dataset
        raise AssertionError("unexpected scheduler fetchrow query")

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        if "SELECT model_version FROM model_versions" in query:
            return self.production_model_version
        if "INSERT INTO learning_cycles" in query:
            return 99
        raise AssertionError("unexpected scheduler fetchval query")

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "OK"


class _SchedulerPool:
    def __init__(self, connection: _SchedulerConnection) -> None:
        self.connection = connection

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


class _CandidateEvaluationConnection:
    def __init__(self, candidate_model_version: str = "candidate-12") -> None:
        self.candidate_model_version = candidate_model_version
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "LEFT JOIN model_versions" in query:
            candidate_model_version = args[1] or self.candidate_model_version
            return {
                "id": 12,
                "cycle_id": "cycle-12",
                "state": "DECISION",
                "candidate_model_version": candidate_model_version,
                "production_model_version": "production-12",
                "candidate_dataset_version": "candidate-dataset-12",
                "model_dataset_version": "candidate-dataset-12",
                "frozen_evaluation_dataset_version": "evaluation-12",
                "promotion_policy_version": "policy-v1",
                "blind_ab_enabled": False,
                "candidate_artifact_checksum": f"sha256:{'a' * 64}",
                "production_artifact_checksum": f"sha256:{'b' * 64}",
                "production_is_synthetic": False,
                "candidate_is_synthetic": False,
                "evaluation_is_synthetic": False,
            }
        if "FROM learning_cycle_candidates" in query:
            return {"status": "REGISTERED"}
        if "FOR UPDATE" in query:
            return {"id": 12, "state": "DECISION", "candidate_model_version": "candidate-12"}
        raise AssertionError("unexpected candidate evaluation fetchrow")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        if "FROM learning_cycle_evaluation_tickets" in query:
            return [
                {
                    "ticket_id": str(index),
                    "text": f"label:{'water' if index % 2 == 0 else 'lighting'}",
                    "label": "water" if index % 2 == 0 else "lighting",
                    "language": "RU",
                }
                for index in range(30)
            ]
        if "FROM learning_cycle_shadow_predictions" in query:
            return [
                {
                    "production_topic_id": "water" if index % 2 == 0 else "lighting",
                    "candidate_topic_id": "water" if index % 2 == 0 else "lighting",
                    "confirmed_topic_id": "water" if index % 2 == 0 else "lighting",
                    "candidate_inference_status": "COMPLETED",
                }
                for index in range(20)
            ]
        raise AssertionError("unexpected candidate evaluation fetch")

    async def fetchval(self, query: str, *_: Any) -> int:
        assert "candidate_inference_status = 'FAILED'" in query
        return 0

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "INSERT 0 1"


class _CandidateEvaluationPool:
    def __init__(self, candidate_model_version: str = "candidate-12") -> None:
        self.connection = _CandidateEvaluationConnection(candidate_model_version)

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
    @staticmethod
    def _schedule_config(**overrides: Any) -> LearningCycleScheduleConfig:
        settings: dict[str, Any] = {
            "enabled": True,
            "environment": "production",
            "collect_duration_hours": 24,
            "minimum_feedback_count": 3,
            "promotion_policy_version": "policy-v1",
            "evaluation_dataset_version": "evaluation-v3",
        }
        settings.update(overrides)
        return LearningCycleScheduleConfig(**settings)

    def test_schedule_config_is_opt_in_and_rejects_invalid_values(self) -> None:
        with patch.dict(
            "os.environ",
            {"PULSE_LEARNING_CYCLE_DURATION_HOURS": "invalid-while-disabled"},
            clear=True,
        ):
            config = learning_cycle_schedule_config_from_env()
        self.assertFalse(config.enabled)
        self.assertEqual(config.collect_duration_hours, 168)
        self.assertIsNone(config.evaluation_dataset_version)

        with patch.dict(
            "os.environ",
            {"PULSE_LEARNING_AUTO_CYCLES_ENABLED": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "must be true or false"):
                learning_cycle_schedule_config_from_env()

    def test_schedule_starts_one_audited_global_cycle_with_frozen_evaluation_set(self) -> None:
        connection = _SchedulerConnection()
        pool = _SchedulerPool(connection)

        with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
            status = asyncio.run(
                start_next_recurring_learning_cycle(pool, self._schedule_config())
            )

        self.assertEqual(status, "STARTED")
        self.assertIn("pg_advisory_xact_lock", connection.executed[0][0])
        self.assertEqual(connection.executed[0][1], ("pulse109:classifier-learning-cycle",))
        cycle_query, cycle_args = next(
            call for call in connection.fetchval_calls if "INSERT INTO learning_cycles" in call[0]
        )
        self.assertIn("manual_close_enabled", cycle_query)
        self.assertTrue(cycle_args[0].startswith("cycle-"))
        self.assertEqual(cycle_args[1:3], (24, "production-v1"))
        self.assertTrue(cycle_args[3].startswith("classifier-candidate-"))
        self.assertEqual(cycle_args[4:], (3, "policy-v1", "evaluation-v3"))

        candidate_query, candidate_args = connection.executed[1]
        self.assertIn("INSERT INTO learning_cycle_candidates", candidate_query)
        self.assertEqual(candidate_args, (99, cycle_args[3]))
        freeze_query, freeze_args = connection.executed[2]
        self.assertIn("INSERT INTO learning_cycle_evaluation_tickets", freeze_query)
        self.assertEqual(freeze_args, (99, "evaluation-v3"))
        audit_query, audit_args = connection.executed[3]
        self.assertIn("CREATE_LEARNING_CYCLE", audit_query)
        self.assertEqual(audit_args[0], "learning-scheduler")
        audit_metadata = json.loads(audit_args[3])
        self.assertEqual(audit_metadata["scope"], "global_classifier")
        self.assertEqual(audit_metadata["schedule_version"], "global-classifier-cycle-v1")
        self.assertEqual(audit_metadata["evaluation_dataset_version"], "evaluation-v3")
        self.assertFalse(any("UPDATE model_versions" in query for query, _ in connection.executed))

    def test_schedule_waits_for_human_decision_and_stops_after_cycle_failures(self) -> None:
        for state, expected in (
            ("DECISION", "ACTIVE_CYCLE"),
            ("TRAINING_FAILED", "BLOCKED_PREVIOUS_CYCLE_FAILED"),
            ("DATASET_BUILD_FAILED", "BLOCKED_PREVIOUS_CYCLE_FAILED"),
        ):
            with self.subTest(state=state):
                connection = _SchedulerConnection(latest_state=state)
                with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
                    status = asyncio.run(
                        start_next_recurring_learning_cycle(
                            _SchedulerPool(connection),
                            self._schedule_config(),
                        )
                    )
                self.assertEqual(status, expected)
                self.assertEqual(len(connection.executed), 1)
                self.assertIn("pg_advisory_xact_lock", connection.executed[0][0])
                self.assertFalse(
                    any("INSERT INTO learning_cycles" in query for query, _ in connection.fetchval_calls)
                )

    def test_schedule_only_repeats_from_approved_terminal_cycle_states(self) -> None:
        for state in ("PROMOTED", "REJECTED", "INSUFFICIENT_FEEDBACK"):
            with self.subTest(state=state):
                connection = _SchedulerConnection(latest_state=state)
                with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
                    status = asyncio.run(
                        start_next_recurring_learning_cycle(
                            _SchedulerPool(connection),
                            self._schedule_config(),
                        )
                    )
                self.assertEqual(status, "STARTED")

    def test_schedule_blocks_synthetic_or_unregistered_production_evaluation_data(self) -> None:
        synthetic_connection = _SchedulerConnection(
            dataset={"is_synthetic": True, "linked_ticket_count": 4}
        )
        empty_connection = _SchedulerConnection(
            dataset={"is_synthetic": False, "linked_ticket_count": 0}
        )
        unregistered_connection = _SchedulerConnection(dataset_lookup_missing=True)

        with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
            synthetic_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(synthetic_connection),
                    self._schedule_config(),
                )
            )
            empty_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(empty_connection),
                    self._schedule_config(),
                )
            )
            unregistered_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(unregistered_connection),
                    self._schedule_config(),
                )
            )
        self.assertEqual(synthetic_status, "BLOCKED_SYNTHETIC_EVALUATION_DATASET")
        self.assertEqual(empty_status, "BLOCKED_EVALUATION_DATASET_EMPTY")
        self.assertEqual(unregistered_status, "BLOCKED_EVALUATION_DATASET_NOT_REGISTERED")
        for connection in (synthetic_connection, empty_connection, unregistered_connection):
            self.assertFalse(
                any("INSERT INTO learning_cycles" in query for query, _ in connection.fetchval_calls)
            )

        no_production_model = _SchedulerConnection(production_model_version=None)
        with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
            no_production_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(no_production_model),
                    self._schedule_config(),
                )
            )
        self.assertEqual(no_production_status, "BLOCKED_PRODUCTION_MODEL_UNAVAILABLE")
        self.assertFalse(
            any("INSERT INTO learning_cycles" in query for query, _ in no_production_model.fetchval_calls)
        )

        no_dataset = _SchedulerConnection()
        with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "false"}):
            no_dataset_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(no_dataset),
                    self._schedule_config(evaluation_dataset_version=None),
                )
            )
        self.assertEqual(no_dataset_status, "BLOCKED_EVALUATION_DATASET_NOT_CONFIGURED")
        self.assertEqual(no_dataset.fetchrow_calls, [])

    def test_disabled_or_fake_trainer_schedule_does_not_create_a_cycle(self) -> None:
        connection = _SchedulerConnection()
        disabled_status = asyncio.run(
            start_next_recurring_learning_cycle(
                _SchedulerPool(connection),
                self._schedule_config(enabled=False),
            )
        )
        self.assertEqual(disabled_status, "DISABLED")
        self.assertEqual(connection.fetchrow_calls, [])

        with patch.dict("os.environ", {"PULSE_TEST_FAKE_TRAINER": "true"}):
            fake_status = asyncio.run(
                start_next_recurring_learning_cycle(
                    _SchedulerPool(connection),
                    self._schedule_config(),
                )
            )
        self.assertEqual(fake_status, "BLOCKED_TEST_FAKE_TRAINER_ENABLED")
        self.assertEqual(connection.fetchrow_calls, [])

    def test_expired_evaluation_window_advances_to_decision_before_collect(self) -> None:
        pool = _Pool(
            _cycle(feedback_count=3),
            evaluation_cycle={"id": 12, "cycle_id": "cycle-12"},
        )

        self.assertTrue(asyncio.run(advance_expired_learning_cycle(pool)))

        self.assertEqual(len(pool.connection.fetchrow_calls), 1)
        self.assertIn("SET state = 'DECISION'", pool.connection.fetchrow_calls[0])
        self.assertIn("EVALUATION_WINDOW_CLOSED", pool.connection.fetchrow_calls[0])
        self.assertEqual(len(pool.connection.executed), 1)
        insert_query, (payload_json,) = pool.connection.executed[0]
        self.assertIn("CANDIDATE_EVALUATION", insert_query)
        self.assertEqual(
            json.loads(payload_json),
            {"kind": "candidate_evaluation", "cycle_id": "cycle-12"},
        )

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
        self.assertEqual(payload["schema_version"], "candidate-training-job.v1")
        self.assertEqual(payload["candidate_dataset_version"], result["dataset_version"])
        self.assertEqual(payload["dataset_checksum"], "a" * 64)
        self.assertEqual(payload["dataset_uri"], result["artifact_uri"])
        self.assertEqual(payload["production_model_version"], "production-12")
        self.assertEqual(payload["training_config_version"], "classifier-training.v1")
        self.assertTrue(payload["output_artifact_uri"].startswith("file://"))
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

    def test_candidate_evaluation_uses_frozen_evidence_and_persists_cycle_lineage(self) -> None:
        pool = _CandidateEvaluationPool()
        *_, evaluator = make_services()

        def classify_version(text: str, model_version: str, artifact_checksum: str | None) -> str:
            return text.removeprefix("label:")

        evaluator._classify_version = classify_version

        result = asyncio.run(
            evaluate_candidate_cycle(
                pool,
                {"cycle_id": "cycle-12"},
                evaluator,
            )
        )

        self.assertEqual(result["decision"], "PENDING_HUMAN_DECISION")
        self.assertEqual(result["offline_evaluation"]["dataset_version"], "evaluation-12")
        self.assertEqual(result["offline_evaluation"]["sample_count"], 30)
        self.assertEqual(result["shadow_evaluation"]["sample_count"], 20)
        insert_query, args = next(
            call for call in pool.connection.executed if "INSERT INTO model_evaluations" in call[0]
        )
        self.assertIn("learning_cycle_id", insert_query)
        self.assertIn("evaluation_payload", insert_query)
        self.assertEqual(args[8], 12)
        persisted = json.loads(args[9])
        self.assertEqual(persisted["cycle_id"], "cycle-12")
        self.assertEqual(persisted["policy_version"], "policy-v1")
        self.assertNotIn("label:water", args[9])
        self.assertTrue(
            any("UPDATE learning_cycle_candidates SET status = 'EVALUATED'" in query for query, _ in pool.connection.executed)
        )

    def test_each_registered_candidate_uses_the_same_frozen_offline_and_shadow_sets(self) -> None:
        *_, evaluator = make_services()

        def classify_version(text: str, model_version: str, artifact_checksum: str | None) -> str:
            return text.removeprefix("label:")

        evaluator._classify_version = classify_version
        results = []
        for model_version in ("candidate-12", "candidate-13"):
            pool = _CandidateEvaluationPool(model_version)
            result = asyncio.run(
                evaluate_candidate_cycle(
                    pool,
                    {
                        "cycle_id": "cycle-12",
                        "candidate_model_version": model_version,
                    },
                    evaluator,
                )
            )
            self.assertEqual(result["candidate_model_version"], model_version)
            self.assertEqual(result["offline_evaluation"]["dataset_version"], "evaluation-12")
            self.assertEqual(result["offline_evaluation"]["sample_count"], 30)
            self.assertEqual(result["shadow_evaluation"]["sample_count"], 20)
            shadow_query, shadow_args = next(
                call for call in pool.connection.fetch_calls if "FROM learning_cycle_shadow_predictions" in call[0]
            )
            self.assertIn("sp.candidate_model_version = $2", shadow_query)
            self.assertEqual(shadow_args[1], model_version)
            results.append(result)

        self.assertEqual(
            [result["offline_evaluation"]["sample_count"] for result in results],
            [30, 30],
        )
        self.assertEqual(
            [result["shadow_evaluation"]["sample_count"] for result in results],
            [20, 20],
        )
