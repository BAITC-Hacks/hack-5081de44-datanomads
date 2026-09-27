from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from typing import Any

from app.candidate_dataset import build_candidate_dataset, export_validated_feedback


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
    def __init__(self) -> None:
        self.feedback = [
            {
                "feedback_id": "21",
                "ticket_id": "101",
                "production_model_version": "production-12",
                "production_prediction": {
                    "topic_id": "water_supply",
                    "confidence": 0.72,
                    "model_version": "production-12",
                },
                "operator_confirmed_decision": {"topic_id": "water_supply"},
                "accepted_or_corrected": "ACCEPTED",
                "validation_status": "VALID",
                "feedback_created_at": datetime(2026, 9, 27, tzinfo=timezone.utc),
                "source_is_synthetic": False,
                "source_version_count": 1,
            },
            {
                "feedback_id": "22",
                "ticket_id": "102",
                "production_model_version": "production-12",
                "production_prediction": {
                    "topic_id": "other",
                    "confidence": 0.4,
                    "model_version": "production-12",
                },
                "operator_confirmed_decision": {"topic_id": "street_lighting"},
                "accepted_or_corrected": "CORRECTED",
                "validation_status": "VALID",
                "feedback_created_at": datetime(2026, 9, 27, 1, tzinfo=timezone.utc),
                "source_is_synthetic": False,
                "source_version_count": 1,
            },
        ]
        self.tickets = [
            {
                "ticket_id": "101",
                "original_text": "Не работает насосная станция",
                "language": "RU",
                "source_dataset_versions": ["source-v1"],
                "source_is_synthetic": False,
            },
            {
                "ticket_id": "102",
                "original_text": "Не горит фонарь во дворе",
                "language": "RU",
                "source_dataset_versions": ["source-v1"],
                "source_is_synthetic": False,
            },
        ]

    def transaction(self) -> _AsyncContext:
        return _AsyncContext()

    async def fetchval(self, query: str, *_: Any) -> int | None:
        assert "FROM learning_cycles" in query
        return 12

    async def fetch(self, query: str, *_: Any) -> list[dict[str, Any]]:
        if "FROM learning_feedback lf" in query:
            return self.feedback
        if "FROM tickets t" in query:
            return self.tickets
        raise AssertionError("unexpected candidate dataset query")


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)


class CandidateDatasetTests(unittest.TestCase):
    def test_builder_excludes_frozen_evaluation_ids_and_returns_lineage(self) -> None:
        pool = _Pool()
        request_payload = {
            "cycle_id": "cycle-12",
            "candidate_model_version": "candidate-12",
            "production_model_version": "production-12",
            "feedback_ids": ["21", "22"],
            "frozen_evaluation_dataset_version": "evaluation-v1",
            "frozen_evaluation_ticket_ids": ["102", "999"],
        }

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            request = asyncio.run(
                export_validated_feedback(pool, request_payload, artifact_dir / "datasets")
            )
            result = asyncio.run(build_candidate_dataset(pool, request, artifact_dir))

            self.assertEqual(result["record_count"], 1)
            self.assertEqual(result["training_ticket_ids"], ["101"])
            self.assertEqual(result["frozen_evaluation_dataset_version"], "evaluation-v1")
            self.assertEqual(result["frozen_evaluation_ticket_ids"], ["102", "999"])
            self.assertEqual(result["source_dataset_versions"], ["source-v1"])
            self.assertFalse(result["synthetic"])

            export_path = Path(request["feedback_export_uri"].removeprefix("file://"))
            feedback_export = json.loads(export_path.read_text(encoding="utf-8"))
            self.assertNotIn("original_text", feedback_export["records"][0])
            self.assertEqual(request["feedback_export_sha256"], hashlib.sha256(export_path.read_bytes()).hexdigest())

            dataset_path = Path(result["artifact_uri"].removeprefix("file://"))
            samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(samples, [{
                "language": "RU",
                "label": "water_supply",
                "text": "Не работает насосная станция",
                "topic_id": "water_supply",
            }])
            self.assertNotIn("samples", request)

            manifest_path = Path(result["manifest_uri"].removeprefix("file://"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frozen_evaluation_dataset_version"], "evaluation-v1")
            self.assertEqual(manifest["training_ticket_ids"], ["101"])
            self.assertEqual(manifest["content_sha256"], result["content_sha256"])
            self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    def test_feedback_export_requires_a_frozen_evaluation_set(self) -> None:
        payload = {
            "cycle_id": "cycle-12",
            "candidate_model_version": "candidate-12",
            "production_model_version": "production-12",
            "feedback_ids": ["21"],
            "frozen_evaluation_dataset_version": None,
            "frozen_evaluation_ticket_ids": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "FROZEN_EVALUATION_SET_NOT_CONFIGURED"):
                asyncio.run(export_validated_feedback(_Pool(), payload, Path(directory)))

    def test_builder_rejects_a_training_set_fully_overlapped_by_evaluation(self) -> None:
        pool = _Pool()
        payload = {
            "cycle_id": "cycle-12",
            "candidate_model_version": "candidate-12",
            "production_model_version": "production-12",
            "feedback_ids": ["21", "22"],
            "frozen_evaluation_dataset_version": "evaluation-v1",
            "frozen_evaluation_ticket_ids": ["101", "102"],
        }

        with tempfile.TemporaryDirectory() as directory:
            request = asyncio.run(
                export_validated_feedback(pool, payload, Path(directory) / "datasets")
            )
            with self.assertRaisesRegex(RuntimeError, "INSUFFICIENT_CANDIDATE_DATASET"):
                asyncio.run(build_candidate_dataset(pool, request, Path(directory)))


if __name__ == "__main__":
    unittest.main()
