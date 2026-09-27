from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.generate_synthetic_classifier import (
    CHALLENGE_BANK, MIXED_BANK, REQUIRED_BOUNDARIES, SCENARIO_BANK, generate,
    read_challenges, read_mixed_scenarios, read_scenarios, render_variants, sha256,
)


class SyntheticClassifierTests(unittest.TestCase):
    def test_variants_do_not_invent_when_problem_started(self) -> None:
        for language, fact, unsupported in (
            ("RU", "В доме нет холодной воды.", "Проблема сохраняется второй день."),
            ("KZ", "Үйде суық су жоқ.", "Мәселе екінші күн сақталып тұр."),
        ):
            with self.subTest(language=language):
                variants = render_variants(fact, language, "water_supply-01")
                self.assertEqual(len(variants), 20)
                self.assertTrue(all(fact in text and unsupported not in text for text in variants))

    def test_scenarios_cover_every_topic(self) -> None:
        scenarios = read_scenarios(SCENARIO_BANK)
        self.assertEqual(len(scenarios), 160)
        self.assertEqual(len({scenario["topic_id"] for scenario in scenarios}), 16)

    def test_challenges_cover_review_decisions_and_hard_boundaries(self) -> None:
        challenges = read_challenges(CHALLENGE_BANK)
        self.assertEqual(len(challenges), 21)
        self.assertEqual({row["proposed_decision"] for row in challenges},
                         {"UNKNOWN", "OTHER", "NEEDS_REVIEW"})
        self.assertEqual({frozenset(row["candidate_topics"]) for row in challenges
                          if row["proposed_decision"] == "NEEDS_REVIEW"}, REQUIRED_BOUNDARIES)
        self.assertTrue(all(row["review_status"] == "PENDING" and
                            row["approved_for_training"] is False and
                            "topic_id" not in row for row in challenges))
        for scenario_id in {row["scenario_id"] for row in challenges}:
            self.assertEqual({row["language"] for row in challenges
                              if row["scenario_id"] == scenario_id}, {"RU", "KZ", "MIXED"})

    def test_incomplete_challenge_language_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "challenges.tsv"
            lines = CHALLENGE_BANK.read_text(encoding="utf-8").splitlines()
            truncated = [line for line in lines
                         if not (line.startswith("light_power_01\t") and "\tMIXED\t" in line)]
            bank.write_text("\n".join(truncated) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks RU/KZ/MIXED"):
                read_challenges(bank)

    def test_mixed_test_bank_covers_each_topic_and_split(self) -> None:
        mixed = read_mixed_scenarios(MIXED_BANK, read_scenarios(SCENARIO_BANK))
        self.assertEqual(len(mixed), 48)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "classifier-v3"
            manifest = generate(output, mixed_bank_path=MIXED_BANK)
            snapshot = json.loads((MIXED_BANK.parent.parent / "manifests/synthetic-classifier-test-v3.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(manifest, snapshot)
            self.assertEqual(manifest["dataset_version"], "synthetic-classifier-v3")
            self.assertEqual(manifest["record_count"], 6448)
            self.assertEqual(manifest["language_counts"], {"RU": 3200, "KZ": 3200, "MIXED": 48})
            self.assertEqual(manifest["split_counts"], {"train": 4496, "validation": 656, "test": 1296})
            self.assertEqual(manifest["mixed_bank_sha256"], sha256(MIXED_BANK))
            self.assertFalse(manifest["approved_for_training"])
            self.assertEqual(manifest["profile_status"], "SYNTHETIC_TEST_ONLY")
            groups = {}
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                mixed_rows = [row for row in rows if row["language"] == "MIXED"]
                self.assertEqual(len(mixed_rows), 16)
                self.assertEqual(len({row["topic_id"] for row in mixed_rows}), 16)
                self.assertTrue(all(row["review_status"] == "PENDING" and
                                    row["approved_for_training"] is False for row in rows))
                groups[split] = {row["scenario_id"] for row in rows}
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            repeated = generate(Path(directory) / "repeated", mixed_bank_path=MIXED_BANK)
            self.assertEqual(repeated["files"], manifest["files"])

    def test_incomplete_mixed_bank_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / "mixed.tsv"
            lines = MIXED_BANK.read_text(encoding="utf-8").splitlines()
            bank.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            output = root / "blocked"
            with self.assertRaisesRegex(ValueError, "one scenario per topic and split"):
                generate(output, mixed_bank_path=bank)
            self.assertFalse(output.exists())

    def test_generation_has_balanced_training_and_disjoint_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "classifier"
            manifest = generate(output)
            self.assertEqual(manifest["dataset_version"], "synthetic-classifier-v2")
            self.assertEqual(manifest["split_counts"], {"train": 4480, "validation": 640, "test": 1280})
            self.assertEqual(manifest["language_counts"], {"RU": 3200, "KZ": 3200})
            self.assertEqual(manifest["scenario_counts_by_split"], {"train": 112, "validation": 16, "test": 32})
            self.assertEqual(manifest["challenge_count"], 21)
            self.assertEqual(manifest["challenge_scenario_count"], 7)
            self.assertEqual(manifest["challenge_proposed_decision_counts"],
                             {"NEEDS_REVIEW": 15, "OTHER": 3, "UNKNOWN": 3})
            self.assertEqual(manifest["challenge_bank_sha256"], sha256(CHALLENGE_BANK))
            self.assertEqual(manifest["challenge_file"]["sha256"], sha256(output / "challenges.jsonl"))

            groups = {}
            texts = set()
            for split in ("train", "validation", "test"):
                with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle]
                groups[split] = {row["scenario_id"] for row in rows}
                by_topic_language = Counter((row["topic_id"], row["language"]) for row in rows)
                expected_per_pair = {"train": {140}, "validation": {20}, "test": {40}}
                self.assertEqual(set(by_topic_language.values()), expected_per_pair[split])
                self.assertTrue(all(row["synthetic"] and row["split"] == split for row in rows))
                self.assertTrue(all(row["review_status"] == "PENDING" for row in rows))
                self.assertEqual(len({row["text"] for row in rows}), len(rows))
                texts.update(row["text"] for row in rows)
                self.assertEqual(manifest["files"][split]["sha256"], sha256(output / f"{split}.jsonl"))
            self.assertEqual(len(texts), 6400)
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            with (output / "challenges.jsonl").open(encoding="utf-8") as handle:
                challenge_rows = [json.loads(line) for line in handle]
            self.assertFalse({row["scenario_id"] for row in challenge_rows} & set().union(*groups.values()))
            self.assertFalse({row["text"] for row in challenge_rows} & texts)

            repeated = generate(Path(directory) / "repeated")
            self.assertEqual(repeated["files"], manifest["files"])
            self.assertEqual(repeated["seed"], manifest["seed"])


if __name__ == "__main__":
    unittest.main()
