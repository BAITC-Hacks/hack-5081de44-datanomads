from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from training.classifier_baselines import evaluate_baselines
from test_dataset_builder import build_fixture_package, fixture_inputs


class ClassifierBaselineTests(unittest.TestCase):
    def test_baselines_share_frozen_package_and_report_all_slices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="baseline")
            manifest = build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            first = evaluate_baselines(package)
            self.assertEqual(first, evaluate_baselines(package))
            self.assertEqual(first["frozen_evaluation_sha256"], manifest.frozen_evaluation_sha256)
            self.assertEqual(first["train_class_counts"], {topic: 3 for topic in first["labels"]})
            self.assertEqual(first["class_imbalance_ratio"], 1)
            self.assertEqual(set(first["models"]), {"majority", "tfidf_linear_svc"})
            for model in first["models"].values():
                for split in ("validation", "test"):
                    metrics = model[split]
                    self.assertEqual(metrics["sample_count"], 30)
                    self.assertEqual(sum(item["support"] for item in metrics["per_class"].values()), 30)
                    self.assertEqual(set(metrics["by_language"]), {"RU", "KZ", "MIXED"})
                    self.assertEqual(len(metrics["confusion_matrix"]), 10)
            self.assertEqual(first["models"]["majority"]["test"]["accuracy"], 0.1)

            command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/evaluate_classifier_baselines.py"),
                       "--dataset", str(package), "--output", str(root / "report.json")]
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads((root / "report.json").read_text(encoding="utf-8")), first)
            command[-1] = str(package / "report.json")
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 2)
            self.assertFalse((package / "report.json").exists())

            test_file = package / "classifier/test.jsonl"
            test_file.write_text(test_file.read_text(encoding="utf-8") + json.dumps({"text": "tampered"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split checksum mismatch"):
                evaluate_baselines(package)


if __name__ == "__main__":
    unittest.main()
