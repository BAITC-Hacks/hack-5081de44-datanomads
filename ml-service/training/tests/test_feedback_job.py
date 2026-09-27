from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.confidence import POLICY_VERSION
from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum
from training.feedback_dataset import load_verified_candidate
from training.feedback_job import FeedbackJobError, _publish_directory, train_classifier_job
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

            policy_text = policy_path.read_text(encoding="utf-8")
            policy_path.write_text(json.dumps({**json.loads(policy_text),
                                               "critical_topics": ["unknown_topic"]}), encoding="utf-8")
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "INVALID_CRITICAL_POLICY"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertFalse((root / "exports").exists())
            self.assertFalse((root / "datasets").exists())
            self.assertFalse((root / "models").exists())
            policy_path.write_text(policy_text, encoding="utf-8")

            frozen_manifest, frozen_splits = load_verified_classifier_package(frozen)
            missing_topic = next(topic for topic in dataset.topics if topic != "electricity")
            incomplete_splits = {**frozen_splits,
                                 "test": [row for row in frozen_splits["test"]
                                          if row["topic_id"] != missing_topic]}
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.load_verified_classifier_package",
                        return_value=(frozen_manifest, incomplete_splits)),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "INVALID_PRODUCTION_ARTIFACT"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertFalse((root / "exports").exists())
            self.assertFalse((root / "datasets").exists())
            self.assertFalse((root / "models").exists())

            def fake_trainer(package: Path, frozen_package: Path, model: Path, output: Path,
                             *, candidate_model_version: str) -> dict:
                manifest, samples = load_verified_candidate(package, frozen_package)
                self.assertEqual(model, production)
                self.assertEqual(candidate_model_version, "candidate_v1")
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].topic_id, "electricity")
                self.assertEqual(output.parent.name, "models")
                self.assertEqual((output.parent.parent / "policies/critical.json").read_text(encoding="utf-8"),
                                 policy_path.read_text(encoding="utf-8"))
                output.mkdir(parents=True)
                (output / "model.safetensors").write_bytes(b"tiny-candidate-model")
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    (output / name).write_text("{}", encoding="utf-8")
                candidate_checksum = checksum(output / "model.safetensors")
                candidate_manifest = json.loads((production / "manifest.json").read_text(encoding="utf-8"))
                candidate_manifest.update({"model_version": candidate_model_version,
                                           "base_model": "production_v1", "status": "CANDIDATE",
                                           "base_model_artifact_checksum": checksum(production / "model.safetensors"),
                                           "dataset_version": manifest.candidate_dataset_version,
                                           "dataset_content_sha256": manifest.content_sha256,
                                           "frozen_evaluation_sha256": dataset.frozen_evaluation_sha256,
                                           "metrics": {"status": "feedback_candidate_unverified",
                                                       "frozen_test_evaluated": False},
                                           "confidence_policy_version": POLICY_VERSION,
                                           "confidence_thresholds": {"low_confidence_below": 0.55,
                                                                     "confident_at_or_above": 1.0},
                                           "confident_enabled": False,
                                           "training_config": {"max_length": 32,
                                                               "input_length_strategy": "head-32",
                                                               "temperature": 1.0,
                                                               "train_samples": len(samples)},
                                           "languages": ["RU", "KZ", "MIXED"],
                                           "artifact_checksum": candidate_checksum})
                (output / "manifest.json").write_text(json.dumps(candidate_manifest), encoding="utf-8")
                return {"candidate_model_version": candidate_model_version,
                        "dataset_version": manifest.candidate_dataset_version,
                        "dataset_content_sha256": manifest.content_sha256,
                        "sample_count": len(samples), "artifact_uri": str(output),
                        "artifact_checksum": candidate_checksum}

            def fake_evaluator(package: Path, base_model: Path, candidate: Path, policy: Path) -> dict:
                self.assertEqual(package, frozen)
                self.assertEqual(base_model, production)
                self.assertEqual(candidate.name, "candidate_v1")
                self.assertEqual(candidate.parent.name, "models")
                self.assertEqual(policy.read_text(encoding="utf-8"), policy_path.read_text(encoding="utf-8"))
                sample_ids = sorted(row["variant_id"] for row in frozen_splits["test"])
                placeholder_metrics = {"macro_f1": 0.5, "weighted_f1": 0.5,
                                       "per_class": {}, "confusion_matrix": [], "accuracy": 0.5}
                return {"report_version": "classifier-pair-evaluation.v1",
                        "dataset_version": "reviewed_v1", "dataset_content_sha256": dataset.content_sha256,
                        "frozen_evaluation_version": "eval_v1",
                        "frozen_evaluation_sha256": dataset.frozen_evaluation_sha256,
                        "synthetic": True, "sample_count": len(sample_ids),
                        "sample_ids_sha256": "sha256:" + hashlib.sha256(
                            json.dumps(sample_ids, ensure_ascii=False).encode("utf-8")
                        ).hexdigest(),
                        "labels": sorted(dataset.topics), "policy_sha256": checksum(policy),
                        "policy_version": "classifier-critical-regression.v1",
                        "policy": json.loads(policy.read_text(encoding="utf-8")),
                        "candidate": {"model_version": "candidate_v1",
                                      "dataset_version": "feedback_candidate_v1",
                                      "artifact_checksum": checksum(candidate / "model.safetensors"),
                                      "metrics": placeholder_metrics},
                        "production": {"model_version": "production_v1",
                                       "artifact_checksum": checksum(production / "model.safetensors"),
                                       "metrics": placeholder_metrics},
                        "critical_topics": {}, "insufficient_critical_topics": [],
                        "regressed_critical_topics": [], "decision": "PENDING_HUMAN_REVIEW"}

            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate", side_effect=RuntimeError("interrupted"))):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    asyncio.run(train_classifier_job(pool, payload))
            self.assertFalse((root / "cycles/1").exists())

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
            self.assertEqual((root / "cycles/1/exports/feedback.jsonl").stat().st_mode & 0o777, 0o600)
            self.assertEqual(result["offline_metrics"]["decision"], "PENDING_HUMAN_REVIEW")
            self.assertTrue((root / "cycles/1/reports/offline.json").is_file())
            self.assertEqual(result["dataset_manifest_uri"],
                             str(root / "cycles/1/datasets/feedback_candidate_v1/manifest.json"))
            self.assertEqual(result["manifest"]["artifact_uri"],
                             str(root / "cycles/1/models/candidate_v1/manifest.json"))
            self.assertTrue((root / "cycles/1/models/candidate_v1/MODEL_CARD.md").is_file())
            self.assertEqual((root / "cycles/1/models/candidate_v1/metrics.json").read_bytes(),
                             (root / "cycles/1/reports/offline.json").read_bytes())
            self.assertNotIn("На дороге", json.dumps(result, ensure_ascii=False))

            stale = root / ".staging/1-interrupted"
            stale.mkdir(mode=0o700)
            (stale / "stage_marker.json").write_text(json.dumps({
                "kind": "pulse-feedback-stage.v1", "cycle_id": "1", "directory": stale.name,
            }), encoding="utf-8")
            (stale / "exports").mkdir()
            (stale / "exports/feedback.jsonl").write_text("unfinished", encoding="utf-8")
            empty_unmarked = root / ".staging/1-before-marker"
            empty_unmarked.mkdir(mode=0o700)
            partial_marker = root / ".staging/1-partial-marker"
            partial_marker.mkdir(mode=0o700)
            (partial_marker / "stage_marker.json").write_text("{", encoding="utf-8")

            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                self.assertEqual(asyncio.run(train_classifier_job(pool, payload)), result)
                trainer.assert_not_called()
            self.assertFalse(stale.exists())
            self.assertTrue(empty_unmarked.is_dir())
            self.assertEqual((partial_marker / "stage_marker.json").read_text(encoding="utf-8"), "{")

            unsafe = root / ".staging/1-unsafe"
            unsafe.mkdir(mode=0o700)
            (unsafe / "stage_marker.json").write_text(json.dumps({
                "kind": "pulse-feedback-stage.v1", "cycle_id": "1", "directory": unsafe.name,
            }), encoding="utf-8")
            (unsafe / "exports").symlink_to(base, target_is_directory=True)
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "STALE_STAGE_UNSAFE"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertTrue(unsafe.is_dir())
            (unsafe / "exports").unlink()
            (unsafe / "stage_marker.json").unlink()
            unsafe.rmdir()

            original_rows = pool.rows
            pool.rows = [{**original_rows[0], "prediction_confidence": original_rows[0]["prediction_confidence"] + 1}]
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "TRAINING_CYCLE_ARTIFACT_CHANGED"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            pool.rows = original_rows

            policy_path.write_text(policy_text.replace("0.1", "0.2"), encoding="utf-8")
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "TRAINING_CYCLE_ARTIFACT_CHANGED"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            policy_path.write_text(policy_text, encoding="utf-8")

            report = root / "cycles/1/reports/offline.json"
            report.write_text("{}", encoding="utf-8")
            with (patch.dict(os.environ, environment),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "TRAINING_CYCLE_ARTIFACT_CHANGED"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertEqual(report.read_text(encoding="utf-8"), "{}")

            production_manifest_path = production / "manifest.json"
            wrong_production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
            wrong_production["frozen_evaluation_version"] = "wrong_eval"
            production_manifest_path.write_text(json.dumps(wrong_production), encoding="utf-8")
            with (patch.dict(os.environ, {**environment, "PULSE_TRAINING_ROOT": str(base / "invalid-private")}),
                  patch("training.feedback_job.train_feedback_candidate") as trainer):
                with self.assertRaisesRegex(FeedbackJobError, "INVALID_PRODUCTION_ARTIFACT"):
                    asyncio.run(train_classifier_job(pool, payload))
                trainer.assert_not_called()
            self.assertFalse((base / "invalid-private/exports").exists())
            self.assertFalse((base / "invalid-private/datasets").exists())
            self.assertFalse((base / "invalid-private/models").exists())

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

    def test_atomic_publish_does_not_replace_existing_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            destination = root / "cycle"
            stage.mkdir()
            destination.mkdir()
            with self.assertRaisesRegex(FeedbackJobError, "TRAINING_CYCLE_EXISTS"):
                _publish_directory(stage, destination)
            self.assertEqual(list(destination.iterdir()), [])
            self.assertTrue(stage.is_dir())


if __name__ == "__main__":
    unittest.main()
