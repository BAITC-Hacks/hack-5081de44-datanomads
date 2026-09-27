from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from training.classifier_input_strategy_comparison import STRATEGIES, compare_input_strategies


CHECKSUM = "sha256:" + "a" * 64


def artifact(root: Path, strategy: str, score: float) -> Path:
    directory = root / strategy
    directory.mkdir()
    weights = f"dummy safetensors for {strategy}".encode()
    (directory / "model.safetensors").write_bytes(weights)
    manifest = {
        "model_version": "private name and phone must not appear in report",
        "model_family": "xlm-roberta-sequence-classification",
        "base_model": "local-base-model",
        "dataset_version": "dataset-v1",
        "dataset_content_sha256": CHECKSUM,
        "frozen_evaluation_version": "frozen-v1",
        "base_model_checksum": CHECKSUM,
        "token_audit_sha256": CHECKSUM,
        "labels": ["roads", "water_supply"],
        "languages": ["RU", "KZ", "MIXED"],
        "metrics": {
            "status": "validation_only",
            "validation": {"macro_f1": score, "sample_count": 20, "scenario_count": 10},
            "test": None,
        },
        "training_config": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "seed": 109,
            "max_length": int(strategy.rsplit("-", 1)[-1]),
            "input_length_strategy": strategy,
            "temperature": 1.0 if "tail" not in strategy else 1.2,
            "reviewed_dataset": True,
            "validation_only": True,
            "train_samples": 80,
        },
        "artifact_checksum": "sha256:" + hashlib.sha256(weights).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def change_manifest(directory: Path, change) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


class ClassifierInputStrategyComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = [artifact(self.root, strategy, score)
                          for strategy, score in zip(STRATEGIES, (0.70, 0.72, 0.74, 0.73))]

    def test_selects_highest_validation_score_without_exposing_manifest_fields(self) -> None:
        report = compare_input_strategies(list(reversed(self.artifacts)))
        self.assertEqual(report["status"], "VALIDATION_ONLY")
        self.assertEqual(report["recommendation"]["input_length_strategy"], "head-512")
        self.assertEqual(report["dataset_content_sha256"], CHECKSUM)
        self.assertEqual(report["validation_sample_count"], 20)
        self.assertEqual([item["input_length_strategy"] for item in report["candidates"]], list(STRATEGIES))
        serialized = json.dumps(report)
        self.assertNotIn("private name", serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("test", serialized)

    def test_ties_choose_shorter_length_then_head(self) -> None:
        change_manifest(self.artifacts[0], lambda data: data["metrics"]["validation"].update(macro_f1=0.74))
        change_manifest(self.artifacts[1], lambda data: data["metrics"]["validation"].update(macro_f1=0.74))
        self.assertEqual(compare_input_strategies(self.artifacts)["recommendation"]["input_length_strategy"],
                         "head-384")

    def test_rejects_test_metrics_and_unverified_weights(self) -> None:
        change_manifest(self.artifacts[0], lambda data: data["metrics"].update(test={"macro_f1": 0.99}))
        with self.assertRaisesRegex(ValueError, "null test"):
            compare_input_strategies(self.artifacts)
        change_manifest(self.artifacts[0], lambda data: data["metrics"].update(test=None))
        (self.artifacts[0] / "model.safetensors").write_bytes(b"tampered weights")
        with self.assertRaisesRegex(ValueError, "checksum does not match"):
            compare_input_strategies(self.artifacts)

    def test_rejects_different_lineage_config_and_validation_population(self) -> None:
        changes = (
            lambda data: data.update(dataset_content_sha256="sha256:" + "b" * 64),
            lambda data: data["training_config"].update(learning_rate=1e-5),
            lambda data: data["metrics"]["validation"].update(sample_count=19),
        )
        for change in changes:
            with self.subTest(change=change):
                original = (self.artifacts[1] / "manifest.json").read_text(encoding="utf-8")
                change_manifest(self.artifacts[1], change)
                with self.assertRaisesRegex(ValueError, "do not share lineage"):
                    compare_input_strategies(self.artifacts)
                (self.artifacts[1] / "manifest.json").write_text(original, encoding="utf-8")

    def test_requires_exactly_four_unique_valid_strategies(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            compare_input_strategies(self.artifacts[:3])
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_input_strategies(self.artifacts[:3] + [self.artifacts[0]])
        change_manifest(self.artifacts[0], lambda data: data["training_config"].update(
            input_length_strategy="head-512"))
        with self.assertRaisesRegex(ValueError, "invalid input-length strategy"):
            compare_input_strategies(self.artifacts)

    def test_rejects_duplicate_manifest_keys(self) -> None:
        path = self.artifacts[0] / "manifest.json"
        value = path.read_text(encoding="utf-8")
        path.write_text(value[:-1] + ', "metrics": {"test": null}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON keys"):
            compare_input_strategies(self.artifacts)

    def test_cli_writes_exclusive_pii_free_report(self) -> None:
        script = Path(__file__).resolve().parents[3] / "scripts" / "compare_classifier_input_strategies.py"
        output = self.root / "comparison.json"
        command = [sys.executable, str(script)]
        for directory in self.artifacts:
            command.extend(("--artifact", str(directory)))
        command.extend(("--output", str(output)))
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout), json.loads(output.read_text(encoding="utf-8")))
        self.assertNotIn("private name", first.stdout)
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), first.stdout)


if __name__ == "__main__":
    unittest.main()
