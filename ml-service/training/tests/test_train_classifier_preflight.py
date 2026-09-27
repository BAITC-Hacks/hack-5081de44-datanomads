from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train_classifier import LABELS, calibration_report, load_reviewed_splits, score, validate_splits
from training.classifier_token_audit import audit_token_lengths
from test_dataset_builder import build_fixture_package, fixture_inputs


class StubTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]:
        return [1] * (len(text.split()) + 2)


class TrainClassifierPreflightTests(unittest.TestCase):
    def test_demo_split_rejects_canonically_equivalent_text(self) -> None:
        splits = {
            name: [
                {"scenario_id": f"{name}_{label}_{language}", "topic_id": label,
                 "language": language, "text": f"Текст {name} {label} {language}"}
                for label in LABELS for language in ("RU", "KZ")
            ]
            for name in ("train", "validation", "test")
        }
        splits["train"][0]["text"] = "Обращение: ё"
        splits["validation"][0]["text"] = "Обращение: е\u0308"
        with self.assertRaisesRegex(ValueError, "text leakage"):
            validate_splits(splits)

    def test_score_includes_class_balance_and_error_matrix(self) -> None:
        metrics = score(["roads", "roads", "water_supply"], ["roads", "water_supply", "water_supply"])
        self.assertEqual(metrics["accuracy"], 0.666667)
        self.assertEqual(metrics["weighted_f1"], 0.666667)
        self.assertEqual(metrics["per_class"]["roads"]["support"], 2)
        self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), 3)

    def test_calibration_report_keeps_language_slices(self) -> None:
        logits = torch.tensor([[3.0, 1.0], [0.5, 2.0], [2.0, 0.5]])
        labels = torch.tensor([0, 1, 0])
        rows = [{"language": language} for language in ("RU", "KZ", "MIXED")]
        report = calibration_report(logits, labels, rows, 1.0)
        self.assertEqual(set(report["by_language"]), {"RU", "KZ", "MIXED"})
        self.assertEqual(report["accuracy"], 1.0)
        self.assertGreaterEqual(report["ece_10_bins"], 0.0)

    def test_requires_matching_reviewed_package_and_token_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3,
                                    prefix="trainer", topic_count=len(LABELS))
            manifest = build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            base_model = root / "base-model"
            base_model.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                (base_model / name).write_text("{}", encoding="utf-8")
            (base_model / "model.safetensors").write_bytes(b"test base weights")
            with patch("training.classifier_token_audit.AutoTokenizer.from_pretrained", return_value=StubTokenizer()):
                audit = audit_token_lengths(package, base_model)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            loaded_manifest, splits, base_checksum = load_reviewed_splits(package, audit_path, base_model, 384)
            self.assertEqual(loaded_manifest.content_sha256, manifest.content_sha256)
            self.assertTrue(base_checksum.startswith("sha256:"))
            self.assertEqual({row["topic_id"] for row in splits["train"]}, set(LABELS))
            load_reviewed_splits(package, audit_path, base_model, 384, "head-tail")
            with self.assertRaisesRegex(ValueError, "max-length"):
                load_reviewed_splits(package, audit_path, base_model, 96)
            with self.assertRaisesRegex(ValueError, "input strategy"):
                load_reviewed_splits(package, audit_path, base_model, 384, "unsupported")

            audit["dataset_content_sha256"] = "sha256:" + "0" * 64
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "token length audit"):
                load_reviewed_splits(package, audit_path, base_model, 384)

    def test_rejects_subset_of_runtime_topics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="subset")
            build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            (root / "model.safetensors").write_bytes(b"test base weights")
            with self.assertRaisesRegex(ValueError, "all canonical runtime labels"):
                load_reviewed_splits(root / "dataset_v1", root / "missing-audit.json", root, 384)


if __name__ == "__main__":
    unittest.main()
