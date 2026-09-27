from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.generate_synthetic_classifier import SCENARIO_BANK, generate, read_scenarios, sha256


class SyntheticClassifierTests(unittest.TestCase):
    def test_scenarios_cover_every_topic(self) -> None:
        scenarios = read_scenarios(SCENARIO_BANK)
        self.assertEqual(len(scenarios), 160)
        self.assertEqual(len({scenario["topic_id"] for scenario in scenarios}), 16)

    def test_generation_has_balanced_training_and_disjoint_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "classifier"
            manifest = generate(output)
            self.assertEqual(manifest["split_counts"], {"train": 20000, "validation": 2000, "test": 4000})
            self.assertEqual(manifest["language_counts"], {"RU": 13000, "KZ": 13000})
            self.assertEqual(manifest["scenario_counts_by_split"], {"train": 112, "validation": 16, "test": 32})

            groups = {}
            texts = set()
            for split in ("train", "validation", "test"):
                with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle]
                groups[split] = {row["scenario_id"] for row in rows}
                by_topic_language = Counter((row["topic_id"], row["language"]) for row in rows)
                expected_per_pair = {"train": {625}, "validation": {62, 63}, "test": {125}}
                self.assertEqual(set(by_topic_language.values()), expected_per_pair[split])
                self.assertTrue(all(row["synthetic"] and row["split"] == split for row in rows))
                self.assertTrue(all(row["review_status"] == "PENDING" for row in rows))
                self.assertEqual(len({row["text"] for row in rows}), len(rows))
                texts.update(row["text"] for row in rows)
                self.assertEqual(manifest["files"][split]["sha256"], sha256(output / f"{split}.jsonl"))
            self.assertEqual(len(texts), 26000)
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])

            repeated = generate(Path(directory) / "repeated")
            self.assertEqual(repeated["files"], manifest["files"])
            self.assertEqual(repeated["seed"], manifest["seed"])


if __name__ == "__main__":
    unittest.main()
