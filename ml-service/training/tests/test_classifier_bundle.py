from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.confidence import POLICY_VERSION
from train_classifier import LABELS
from training.classifier_baselines import evaluate_baselines, load_verified_classifier_package
from training.classifier_bundle import MODEL_FILES, build_classifier_bundle, verify_classifier_bundle
from training.dataset_builder import checksum
from test_classifier_candidate_eval import StubClassifier
from test_dataset_builder import build_fixture_package, fixture_inputs


class ClassifierBundleTests(unittest.TestCase):
    def test_handoff_checks_lineage_files_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3,
                                    prefix="bundle", topic_count=16)
            dataset = build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(evaluate_baselines(package)), encoding="utf-8")
            model = root / "model"
            model.mkdir()
            for name in MODEL_FILES:
                if name != "manifest.json":
                    (model / name).write_bytes(b"fixture artifact")
            labels = list(LABELS)
            (model / "manifest.json").write_text(json.dumps({
                "model_version": "classifier-test",
                "model_family": "xlm-roberta-sequence-classification",
                "base_model": "local-test",
                "dataset_version": dataset.dataset_version,
                "dataset_content_sha256": dataset.content_sha256,
                "frozen_evaluation_version": dataset.frozen_evaluation_version,
                "created_at": "2026-09-27T00:00:00Z",
                "metrics": {"status": "reviewed_synthetic_holdout_only",
                            "validation": {"calibration": {"method": "temperature"}}},
                "labels": labels,
                "languages": ["RU", "KZ", "MIXED"],
                "training_config": {"reviewed_dataset": True, "seed": 109,
                                    "input_length_strategy": "head-384", "temperature": 1.0},
                "confidence_policy_version": POLICY_VERSION,
                "confidence_thresholds": {"low_confidence_below": 0.55,
                                          "confident_at_or_above": 1.0},
                "confident_enabled": False,
                "artifact_checksum": checksum(model / "model.safetensors"),
            }), encoding="utf-8")
            _, splits = load_verified_classifier_package(package)
            stub = StubClassifier({row["text"]: row["topic_id"] for row in splits["test"]})
            bundle = root / "bundle"
            with patch("training.classifier_candidate_eval.TrainedClassifierService", return_value=stub):
                manifest = build_classifier_bundle(package, model, baseline, bundle)

            self.assertEqual(manifest, verify_classifier_bundle(bundle))
            self.assertEqual(manifest.artifact_uri, "model")
            self.assertEqual(manifest.status, "CANDIDATE")
            self.assertEqual(manifest.evaluation_version, "eval_v1")
            self.assertEqual(manifest.held_out_metrics["accuracy"], 1.0)
            self.assertIn("0% real citizen texts", (bundle / "MODEL_CARD.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                build_classifier_bundle(package, model, baseline, bundle)

            metrics_path = bundle / "metrics.json"
            original_metrics = metrics_path.read_bytes()
            metrics_path.write_bytes(original_metrics + b" ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_classifier_bundle(bundle)
            metrics_path.write_bytes(original_metrics)

            thresholds_path = bundle / "thresholds.json"
            original_thresholds = thresholds_path.read_bytes()
            manifest_path = bundle / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            thresholds = json.loads(original_thresholds)
            thresholds["confidence_thresholds"]["low_confidence_below"] = 0.6
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
            changed_manifest = json.loads(original_manifest)
            changed_manifest["confidence_thresholds"] = thresholds["confidence_thresholds"]
            changed_manifest["bundle_files"]["thresholds.json"] = checksum(thresholds_path)
            manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match its evidence"):
                verify_classifier_bundle(bundle)
            thresholds_path.write_bytes(original_thresholds)
            manifest_path.write_bytes(original_manifest)

            (bundle / "model" / "extra.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected or missing"):
                verify_classifier_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
