from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from training.shadow_job import ShadowJobError, score_shadow_ticket


CHECKSUM = "sha256:" + "a" * 64


class ShadowPool:
    def __init__(self, row: dict, saved_id: int | None = 9) -> None:
        self.row = row
        self.saved_id = saved_id
        self.saved_query = None
        self.saved_values = None

    @asynccontextmanager
    async def acquire(self):
        yield self

    async def fetchrow(self, query: str, *values):
        return self.row

    async def fetchval(self, query: str, *values):
        self.saved_query = query
        self.saved_values = values
        return self.saved_id


class ShadowJobTests(unittest.TestCase):
    def test_scores_only_fresh_undecided_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/candidate_v1"
            model.mkdir(parents=True)
            (model / "manifest.json").write_text("{}", encoding="utf-8")
            started = datetime(2026, 9, 27, tzinfo=timezone.utc)
            row = {
                "original_text": "На дороге появилась яма.", "language": "RU",
                "ticket_created_at": started + timedelta(seconds=1),
                "state": "EVALUATE", "production_model_version": "production_v1",
                "candidate_model_version": "candidate_v1", "evaluation_started_at": started,
                "evaluation_ends_at": None,
                "candidate_status": "CANDIDATE", "artifact_checksum": CHECKSUM,
                "manifest_uri": str(model / "manifest.json"),
                "ticket_production_model_version": "production_v1",
                "production_predicted_at": started + timedelta(seconds=1), "decided": False,
            }
            payload = {"cycle_id": "1", "ticket_id": "7", "production_prediction_id": "8",
                       "production_model_version": "production_v1",
                       "candidate_model_version": "candidate_v1",
                       "candidate_artifact_checksum": CHECKSUM}
            pool = ShadowPool(row)

            class StubClassifier:
                model_version = "candidate_v1"
                metadata = SimpleNamespace(artifact_checksum=CHECKSUM, model_extra={"status": "CANDIDATE"})

                def classify(self, text: str, language: str | None = None):
                    self_seen.append((text, language))
                    return SimpleNamespace(topic_id="roads", confidence=0.7)

            self_seen = []
            with (patch.dict(os.environ, {"PULSE_TRAINING_ROOT": str(root)}),
                  patch("training.shadow_job._cached_model", None),
                  patch("training.shadow_job.TrainedClassifierService", return_value=StubClassifier()) as loader):
                result = asyncio.run(score_shadow_ticket(pool, payload))
                self.assertEqual(result["status"], "RECORDED")
                self.assertEqual(self_seen, [(row["original_text"], "RU")])
                self.assertIn("NOT EXISTS (SELECT 1 FROM operator_decisions", pool.saved_query)
                self.assertNotIn(row["original_text"], str(pool.saved_values))
                self.assertNotIn(row["original_text"], str(result))
                self.assertEqual(pool.saved_values[:5], (1, 7, 8, "candidate_v1", CHECKSUM))
                loader.assert_called_once()

            pool = ShadowPool(row, saved_id=None)
            with (patch.dict(os.environ, {"PULSE_TRAINING_ROOT": str(root)}),
                  patch("training.shadow_job._cached_model", None),
                  patch("training.shadow_job.TrainedClassifierService", return_value=StubClassifier())):
                result = asyncio.run(score_shadow_ticket(pool, payload))
                self.assertEqual(result["status"], "SKIPPED_CONTEXT_CHANGED")

            pool = ShadowPool({**row, "decided": True})
            with (patch.dict(os.environ, {"PULSE_TRAINING_ROOT": str(root)}),
                  patch("training.shadow_job._cached_model", None),
                  patch("training.shadow_job.TrainedClassifierService") as loader):
                result = asyncio.run(score_shadow_ticket(pool, payload))
                self.assertEqual(result["status"], "SKIPPED_OPERATOR_DECIDED")
                self.assertIsNone(pool.saved_query)
                loader.assert_not_called()

            with patch.dict(os.environ, {"PULSE_TRAINING_ROOT": str(root)}):
                with self.assertRaisesRegex(ShadowJobError, "INVALID_SHADOW_JOB"):
                    asyncio.run(score_shadow_ticket(pool, {**payload, "ticket_id": "../7"}))
                with self.assertRaisesRegex(ShadowJobError, "SHADOW_CONTEXT_CHANGED"):
                    asyncio.run(score_shadow_ticket(ShadowPool({**row, "artifact_checksum": "sha256:" + "0" * 64}),
                                                    payload))
                with self.assertRaisesRegex(ShadowJobError, "SHADOW_CONTEXT_CHANGED"):
                    asyncio.run(score_shadow_ticket(ShadowPool({**row, "ticket_created_at": started - timedelta(days=1)}),
                                                    payload))


if __name__ == "__main__":
    unittest.main()
