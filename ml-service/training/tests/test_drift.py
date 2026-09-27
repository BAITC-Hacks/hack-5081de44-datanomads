from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from scripts.export_drift_snapshot import build_snapshot
from training.drift import DriftPolicy, DriftSnapshot, compare_snapshots, evaluate_drift


START = datetime(2026, 9, 20, tzinfo=timezone.utc)


def snapshot(start: datetime) -> DriftSnapshot:
    return DriftSnapshot.model_validate({
        "snapshot_version": "pulse-drift-snapshot.v1", "source": "postgres",
        "model_version": "classifier_v1", "window_start": start,
        "window_end": start + timedelta(days=1), "ticket_count": 4,
        "character_lengths": [2, 2, 0, 0, 0, 0],
        "token_lengths": None, "tokenizer_file_checksums": None,
        "language_counts": {"RU": 2, "KZ": 2},
        "topic_counts": {"roads": 2, "water_supply": 2}, "topic_prediction_count": 4,
        "confidence_counts": [0, 0, 2, 0, 2], "prediction_count": 4,
        "decision_count": 3, "corrected_count": 1,
        "similarity_feedback_count": 2,
    })


def policy() -> DriftPolicy:
    return DriftPolicy.model_validate({
        "policy_version": "pulse-drift-policy.v1", "min_tickets": 4,
        "min_predictions": 4, "min_decisions": 3,
        "max_distribution_tv": 0.25,
        "max_correction_rate_increase": 0.2,
    })


class DriftTests(unittest.TestCase):
    def test_detects_distribution_and_correction_drift_without_retraining(self) -> None:
        earlier = snapshot(START)
        later = snapshot(START + timedelta(days=1)).model_copy(update={
            "character_lengths": [0, 4, 0, 0, 0, 0],
            "language_counts": {"RU": 4},
            "corrected_count": 2,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("baseline.json", "recent.json", "policy.json")]
            for path, value in zip(paths, (earlier, later, policy())):
                path.write_text(value.model_dump_json(), encoding="utf-8")
            report = evaluate_drift(*paths)
        self.assertEqual(report["status"], "REVIEW_TRIGGER")
        self.assertEqual(report["drifted_signals"], ["character_length", "language", "correction_rate"])
        self.assertFalse(report["automatic_retraining"])
        self.assertEqual(report["signals"]["retrieval_quality"]["status"],
                         "UNAVAILABLE_NO_RANKED_RELEVANCE")
        self.assertEqual(report["signals"]["model_token_length"]["status"],
                         "UNAVAILABLE_TOKENIZER_MISMATCH_OR_MISSING")
        self.assertNotIn("original_text", json.dumps(report))

    def test_low_support_and_mismatched_model_are_not_drift_claims(self) -> None:
        earlier = snapshot(START)
        later = snapshot(START + timedelta(days=1)).model_copy(update={
            "decision_count": 0, "corrected_count": 0,
        })
        report = compare_snapshots(earlier, later, policy())
        self.assertEqual(report["status"], "PARTIAL_NO_DRIFT")
        self.assertEqual(report["signals"]["correction_rate"]["status"],
                         "INSUFFICIENT_GROUND_TRUTH")
        with self.assertRaisesRegex(ValueError, "overlapping windows"):
            compare_snapshots(earlier, earlier, policy())
        with self.assertRaisesRegex(ValueError, "different models"):
            compare_snapshots(earlier, later.model_copy(update={"model_version": "v2"}), policy())

    def test_rejects_inconsistent_counts_and_compares_matching_tokenizer(self) -> None:
        with self.assertRaises(ValidationError):
            DriftSnapshot.model_validate({**snapshot(START).model_dump(), "ticket_count": 5})
        checksums = {name: "sha256:" + "a" * 64 for name in
                     ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")}
        earlier = snapshot(START).model_copy(update={
            "token_lengths": [4, 0, 0, 0, 0, 0], "tokenizer_file_checksums": checksums,
        })
        later = snapshot(START + timedelta(days=1)).model_copy(update={
            "token_lengths": [0, 4, 0, 0, 0, 0], "tokenizer_file_checksums": checksums,
        })
        report = compare_snapshots(earlier, later, policy())
        self.assertEqual(report["signals"]["model_token_length"]["status"], "DRIFT")

    def test_thresholds_use_unrounded_scores(self) -> None:
        earlier = DriftSnapshot.model_validate({**snapshot(START).model_dump(),
                                                "ticket_count": 6,
                                                "character_lengths": [2, 4, 0, 0, 0, 0],
                                                "language_counts": {"RU": 6}})
        later = DriftSnapshot.model_validate({**snapshot(START + timedelta(days=1)).model_dump(),
                                              "ticket_count": 6,
                                              "character_lengths": [4, 2, 0, 0, 0, 0],
                                              "language_counts": {"RU": 6},
                                              "corrected_count": 2})
        threshold_policy = policy().model_copy(update={
            "max_distribution_tv": 0.3333331,
            "max_correction_rate_increase": 0.3333331,
        })
        report = compare_snapshots(earlier, later, threshold_policy)
        self.assertEqual(report["signals"]["character_length"]["status"], "DRIFT")
        self.assertEqual(report["signals"]["character_length"]["total_variation"], 0.333333)
        self.assertEqual(report["signals"]["correction_rate"]["status"], "DRIFT")
        self.assertEqual(report["signals"]["correction_rate"]["rate_increase"], 0.333333)

    def test_empty_windows_are_insufficient_instead_of_stable(self) -> None:
        empty = snapshot(START).model_copy(update={
            "ticket_count": 0, "character_lengths": [0] * 6,
            "language_counts": {}, "topic_counts": {}, "topic_prediction_count": 0,
            "confidence_counts": [0] * 5, "prediction_count": 0,
            "decision_count": 0, "corrected_count": 0,
        })
        later = empty.model_copy(update={"window_start": START + timedelta(days=1),
                                         "window_end": START + timedelta(days=2)})
        report = compare_snapshots(empty, later, policy())
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["signals"]["topic"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_snapshot_builder_outputs_counts_without_text(self) -> None:
        class Transaction:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class Connection:
            def transaction(self, **_kwargs):
                return Transaction()

            async def cursor(self, *_args, **_kwargs):
                yield {"original_text": "Перекрыта дорога", "language": "RU",
                       "topic_id": "roads", "confidence": 0.9}

            async def fetchrow(self, *_args):
                return {"decision_count": 1, "corrected_count": 0}

            async def fetchval(self, *_args):
                return 1

            async def close(self):
                return None

        async def connect(_url):
            return Connection()

        with patch.dict(sys.modules, {"asyncpg": types.SimpleNamespace(connect=connect)}):
            result = asyncio.run(build_snapshot("postgres://local", START, START + timedelta(days=1),
                                                "classifier_v1"))
        self.assertEqual(result["ticket_count"], 1)
        self.assertEqual(result["topic_counts"], {"roads": 1})
        self.assertEqual(result["topic_prediction_count"], 1)
        self.assertNotIn("Перекрыта", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
