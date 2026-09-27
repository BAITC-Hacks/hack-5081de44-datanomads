from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.generate_synthetic_classifier import (
    CHALLENGE_BANK, REQUIRED_BOUNDARIES, SCENARIO_BANK, generate, read_challenges,
    read_scenarios, render_variants, sha256,
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
