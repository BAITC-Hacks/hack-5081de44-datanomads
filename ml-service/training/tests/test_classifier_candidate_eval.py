from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.classifier_baselines import evaluate_baselines, load_verified_classifier_package
from training.classifier_candidate_eval import evaluate_candidate
from training.dataset_builder import build_package
from test_dataset_builder import fixture_inputs


class StubClassifier:
    device = torch.device("cpu")

    def __init__(self, labels_by_text: dict[str, str]) -> None:
        self.labels_by_text = labels_by_text

    def classify(self, text: str, language: str | None = None):
        return SimpleNamespace(topic_id=self.labels_by_text[text], confidence_state="UNCERTAIN", needs_review=True)


class ClassifierCandidateEvaluationTests(unittest.TestCase):
    def test_compares_same_frozen_test_without_exporting_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3,
                                    prefix="evaluation", topic_count=16)
            dataset = build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(evaluate_baselines(package)), encoding="utf-8")
            model_dir = root / "model"
            model_dir.mkdir()
            labels = sorted(dataset.topics)
            (model_dir / "manifest.json").write_text(json.dumps({
                "model_version": "classifier-test",
                "model_family": "xlm-roberta-sequence-classification",
                "base_model": "local-test",
                "dataset_version": dataset.dataset_version,
                "dataset_content_sha256": dataset.content_sha256,
                "frozen_evaluation_version": dataset.frozen_evaluation_version,
                "created_at": "2026-09-27T00:00:00Z",
                "metrics": {"status": "reviewed_synthetic_holdout_only"},
                "labels": labels,
                "training_config": {"reviewed_dataset": True, "max_length": 384, "temperature": 1.0},
                "artifact_checksum": "sha256:" + "0" * 64,
            }), encoding="utf-8")
            _, splits = load_verified_classifier_package(package)
            stub = StubClassifier({row["text"]: row["topic_id"] for row in splits["test"]})
            with patch("training.classifier_candidate_eval.TrainedClassifierService", return_value=stub):
                report = evaluate_candidate(package, model_dir, baseline_path)

            self.assertEqual(report["sample_count"], 48)
            self.assertEqual(report["candidate_metrics"]["accuracy"], 1.0)
            self.assertEqual(report["decision"], "PENDING_HUMAN_REVIEW")
            self.assertEqual(report["latency"]["timed_samples"], 48)
            self.assertLessEqual(report["latency"]["p50_ms"], report["latency"]["p95_ms"])
            self.assertEqual(report["needs_review_share"], 1.0)
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertTrue(all(row["text"] not in serialized for row in splits["test"]))

            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline["dataset_content_sha256"] = "sha256:" + "0" * 64
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline does not match"):
                evaluate_candidate(package, model_dir, baseline_path)

            baseline = evaluate_baselines(package)
            baseline["models"]["tfidf_linear_svc"]["test"]["macro_f1"] = 0.0
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline metrics do not match"):
                evaluate_candidate(package, model_dir, baseline_path)


if __name__ == "__main__":
    unittest.main()
