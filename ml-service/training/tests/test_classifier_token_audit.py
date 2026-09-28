from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.classifier_token_audit import audit_token_lengths
from test_dataset_builder import build_fixture_package, fixture_inputs


class StubTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]:
        assert add_special_tokens and not truncation
        self.calls += 1
        return [1] * (len(text.split()) + 2)


class ClassifierTokenAuditTests(unittest.TestCase):
    def test_uses_verified_train_validation_only_and_reports_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="tokens")
            build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                (tokenizer_dir / name).write_text("{}", encoding="utf-8")
            tokenizer = StubTokenizer()
            with patch("training.classifier_token_audit.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
                report = audit_token_lengths(package, tokenizer_dir)

            load.assert_called_once_with(tokenizer_dir, local_files_only=True, use_fast=True)
            self.assertEqual(tokenizer.calls, 60)
            self.assertEqual(set(report["splits"]), {"train", "validation"})
            self.assertEqual(report["splits"]["train"]["all"]["sample_count"], 30)
            self.assertEqual(report["splits"]["validation"]["all"]["sample_count"], 30)
            self.assertEqual(set(report["splits"]["train"]), {"RU", "KZ", "MIXED", "all"})
            self.assertFalse(report["test_token_lengths_computed"])
            self.assertNotIn("text", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
