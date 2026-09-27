from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.dataset_builder import checksum
from training.feedback_dataset import load_verified_candidate
from training.feedback_job import FeedbackJobError, train_classifier_job
from test_dataset_builder import build_fixture_package, fixture_inputs
from test_feedback_export import feedback_row, link


class FeedbackPool:
    def __init__(self, rows: list[dict], *, state: str = "TRAINING") -> None:
        self.rows = rows
        self.state = state
        self.cycle_id = None

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self, **options):
        self.options = options
        yield self

    async def fetch(self, query: str, cycle_id: str) -> list[dict]:
        self.cycle_id = cycle_id
        self.query = query
        return self.rows

    async def fetchrow(self, query: str, cycle_id: str) -> dict:
        return {"state": self.state, "production_model_version": "production_v1",
                "candidate_dataset_version": "feedback_candidate_v1" if self.rows else "candidate_v1",
                "candidate_model_version": "candidate_v1" if self.rows else "model_v1",
                "min_feedback_count": 1}


class FeedbackJobTests(unittest.TestCase):
    def test_builds_verified_dataset_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs = fixture_inputs(base, groups_per_topic=3, retrieval_groups=3,
                                    prefix="feedbackjob", topic_count=16)
            dataset = build_fixture_package(inputs, base / "frozen-root", "reviewed_v1", "eval_v1", 109)
            frozen = base / "frozen-root/reviewed_v1"
            production = base / "production"
            production.mkdir()
            (production / "model.safetensors").write_bytes(b"tiny-test-model")
            (production / "manifest.json").write_text(json.dumps({
                "model_version": "production_v1", "model_family": "test-classifier",
                "base_model": "local-test", "dataset_version": "reviewed_v1",
                "frozen_evaluation_version": "eval_v1", "created_at": "2026-09-27T00:00:00Z",
                "labels": list(dataset.topics),
                "artifact_checksum": checksum(production / "model.safetensors"),
            }), encoding="utf-8")
            review_links = base / "review-links.jsonl"
            review_links.write_text(json.dumps(link().model_dump(mode="json")) + "\n", encoding="utf-8")
            policy_path = base / "critical-policy.json"
            policy_path.write_text(json.dumps({
                "policy_version": "classifier-critical-regression.v1",
                "critical_topics": ["electricity"], "max_f1_drop": 0.1,
                "min_topic_support": 1, "min_total_samples": 1,
            }), encoding="utf-8")
            root = base / "private"
            environment = {
                "PULSE_TRAINING_ROOT": str(root),
                "PULSE_TRAINING_REVIEW_LINKS": str(review_links),
                "PULSE_TRAINING_FROZEN_DATASET": str(frozen),
                "PULSE_TRAINING_PRODUCTION_MODEL": str(production),
                "PULSE_TRAINING_CRITICAL_POLICY": str(policy_path),
            }
            payload = {
                "cycle_id": "1", "dataset_version": "feedback_candidate_v1",
                "production_model_version": "production_v1",
                "candidate_model_version": "candidate_v1", "min_samples": 1,
            }
            pool = FeedbackPool([feedback_row()])

            def fake_trainer(package: Path, frozen_package: Path, model: Path, output: Path,
                             *, candidate_model_version: str) -> dict:
                manifest, samples = load_verified_candidate(package, frozen_package)
                self.assertEqual(model, production)
                self.assertEqual(candidate_model_version, "candidate_v1")
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].topic_id, "electricity")
                self.assertTrue((root / "policies/1.json").is_file())
                return {"candidate_model_version": candidate_model_version,
                        "dataset_version": manifest.candidate_dataset_version,
                        "dataset_content_sha256": manifest.content_sha256,
                        "sample_count": len(samples), "artifact_uri": str(output),
                        "artifact_checksum": "sha256:" + "a" * 64}

            def fake_evaluator(package: Path, base_model: Path, candidate: Path, policy: Path) -> dict:
                self.assertEqual(package, frozen)
                self.assertEqual(base_model, production)
                self.assertEqual(candidate, root / "models/candidate_v1")
                self.assertEqual(policy.read_text(encoding="utf-8"), policy_path.read_text(encoding="utf-8"))
                return {"report_version": "classifier-pair-evaluation.v1",
                        "dataset_version": "reviewed_v1", "frozen_evaluation_version": "eval_v1",
                        "regressed_critical_topics": [], "decision": "PENDING_HUMAN_REVIEW"}

            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate", side_effect=fake_trainer),
                  patch("training.feedback_job.compare_classifiers", side_effect=fake_evaluator)):
                result = asyncio.run(train_classifier_job(pool, payload))
            self.assertEqual(result["state"], "COMPLETED")
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(pool.cycle_id, "1")
            self.assertIn("lc.id::text = $1", pool.query)
            self.assertEqual(pool.options, {"isolation": "repeatable_read", "readonly": True})
            self.assertFalse(root.stat().st_mode & 0o077)
            self.assertEqual((root / "exports/1.jsonl").stat().st_mode & 0o777, 0o600)
            self.assertEqual(result["offline_metrics"]["decision"], "PENDING_HUMAN_REVIEW")
            self.assertTrue((root / "reports/1-offline.json").is_file())
            self.assertNotIn("На дороге", json.dumps(result, ensure_ascii=False))

            with patch.dict(os.environ, environment):
                with self.assertRaises(FileExistsError):
                    asyncio.run(train_classifier_job(pool, payload))

            production_manifest_path = production / "manifest.json"
            wrong_production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
            wrong_production["frozen_evaluation_version"] = "wrong_eval"
            production_manifest_path.write_text(json.dumps(wrong_production), encoding="utf-8")
            with (patch.dict(os.environ, {**environment, "PULSE_TRAINING_ROOT": str(base / "invalid-private")}),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "INVALID_PRODUCTION_ARTIFACT"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertFalse((base / "invalid-private/datasets").exists())

    def test_insufficient_feedback_and_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen = base / "frozen"
            production = base / "production"
            frozen.mkdir()
            production.mkdir()
            review_links = base / "review-links.jsonl"
            review_links.write_text(json.dumps(link().model_dump(mode="json")) + "\n", encoding="utf-8")
            policy_path = base / "critical-policy.json"
            policy_path.write_text("{}", encoding="utf-8")
            environment = {
                "PULSE_TRAINING_ROOT": str(base / "private"),
                "PULSE_TRAINING_REVIEW_LINKS": str(review_links),
                "PULSE_TRAINING_FROZEN_DATASET": str(frozen),
                "PULSE_TRAINING_PRODUCTION_MODEL": str(production),
                "PULSE_TRAINING_CRITICAL_POLICY": str(policy_path),
            }
            payload = {"cycle_id": "1", "dataset_version": "candidate_v1",
                       "production_model_version": "production_v1",
                       "candidate_model_version": "model_v1", "min_samples": 1}
            with patch.dict(os.environ, environment):
                result = asyncio.run(train_classifier_job(FeedbackPool([]), payload))
                self.assertEqual(result["state"], "INSUFFICIENT_FEEDBACK")
                self.assertFalse((base / "private/datasets").exists())
                with self.assertRaisesRegex(FeedbackJobError, "INVALID_JOB_PAYLOAD"):
                    asyncio.run(train_classifier_job(FeedbackPool([]), {**payload, "dataset_version": "../bad"}))
                with self.assertRaisesRegex(FeedbackJobError, "TRAINING_CYCLE_CHANGED"):
                    asyncio.run(train_classifier_job(FeedbackPool([], state="EVALUATE"), payload))


if __name__ == "__main__":
    unittest.main()
