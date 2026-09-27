from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from training.classifier_baselines import load_verified_classifier_package
from training.classifier_pair_eval import compare_classifiers
from training.dataset_builder import checksum
from test_classifier_candidate_eval import StubClassifier
from test_dataset_builder import build_fixture_package, fixture_inputs


class ClassifierPairEvaluationTests(unittest.TestCase):
    def test_pair_uses_same_frozen_ids_and_explicit_critical_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3,
                                    prefix="pair", topic_count=16)
            dataset = build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            _, splits = load_verified_classifier_package(package)
            labels = sorted(dataset.topics)
            critical = labels[0]
            alternative = labels[1]
            for name in ("production", "candidate"):
                model = root / name
                model.mkdir()
                (model / "manifest.json").write_text(json.dumps({
                    "model_version": name + "-v1",
                    "model_family": "xlm-roberta-sequence-classification",
                    "base_model": "production-v1" if name == "candidate" else "local-test",
                    "dataset_version": "feedback_v1" if name == "candidate" else dataset.dataset_version,
                    "frozen_evaluation_version": dataset.frozen_evaluation_version,
                    "frozen_evaluation_sha256": dataset.frozen_evaluation_sha256 if name == "candidate" else None,
                    "created_at": "2026-09-27T00:00:00Z",
                    "labels": labels,
                    "training_config": {"max_length": 32, "input_length_strategy": "head-32"},
                    "artifact_checksum": "sha256:" + "0" * 64,
                }), encoding="utf-8")
            candidate_path = root / "candidate/manifest.json"
            candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate_data["base_model_artifact_checksum"] = "sha256:" + "0" * 64
            candidate_data["base_model_manifest_sha256"] = checksum(root / "production/manifest.json")
            candidate_path.write_text(json.dumps(candidate_data), encoding="utf-8")
            policy_path = root / "policy.json"
            policy = {"policy_version": "classifier-critical-regression.v1",
                      "critical_topics": [critical], "max_f1_drop": 0.1,
                      "min_topic_support": 3, "min_total_samples": 48}
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            production_labels = {row["text"]: row["topic_id"] for row in splits["test"]}
            candidate_labels = {text: alternative if topic == critical else topic
                                for text, topic in production_labels.items()}

            def evaluate(*, token_offset: int = 0) -> dict:
                active = [0]

                class StubTokenizer:
                    def __init__(self, offset: int):
                        self.offset = offset

                    def __call__(self, texts: list[str], **kwargs):
                        return {"input_ids": torch.tensor([[len(texts[0]) + self.offset]]),
                                "attention_mask": torch.tensor([[1]])}

                def load_classifier(path: Path):
                    self.assertEqual(active[0], 0, "both classifiers were held in memory")
                    classifier = StubClassifier(production_labels if path.name == "production" else candidate_labels)
                    active[0] += 1
                    weakref.finalize(classifier, lambda: active.__setitem__(0, active[0] - 1))
                    return classifier

                with (patch("training.classifier_pair_eval.TrainedClassifierService", side_effect=load_classifier),
                      patch("training.classifier_pair_eval.AutoTokenizer.from_pretrained",
                            side_effect=[StubTokenizer(0), StubTokenizer(token_offset)])):
                    report = compare_classifiers(package, root / "production", root / "candidate", policy_path)
                self.assertEqual(active[0], 0)
                return report

            report = evaluate()
            self.assertEqual(report["decision"], "CRITICAL_REGRESSION")
            self.assertEqual(report["sample_count"], 48)
            self.assertEqual(report["regressed_critical_topics"], [critical])
            self.assertEqual(report["candidate"]["dataset_version"], "feedback_v1")
            self.assertEqual(report["critical_topics"][critical]["support"], 3)
            self.assertTrue(all(row["text"] not in json.dumps(report, ensure_ascii=False)
                                for row in splits["test"]))

            policy["max_f1_drop"] = 1.0
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            self.assertEqual(evaluate()["decision"], "PENDING_HUMAN_REVIEW")
            policy["min_topic_support"] = 4
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            self.assertEqual(evaluate()["decision"], "INSUFFICIENT_EVIDENCE")
            policy["critical_topics"] = ["not_a_topic"]
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown topics"):
                compare_classifiers(package, root / "production", root / "candidate", policy_path)

            policy["critical_topics"] = [critical]
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenization differs"):
                evaluate(token_offset=1)
            candidate_data["training_config"]["max_length"] = 64
            candidate_path.write_text(json.dumps(candidate_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                evaluate()


if __name__ == "__main__":
    unittest.main()
