from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import unicodedata

from scripts.pulse_sdg import (
    DEFAULT_SEEDS, GENERATION_SEED_FIELDS, PROMPT_VERSION, export_candidates,
    read_seeds, source_checksum, write_generation_seeds,
)


class PilotSdgTests(unittest.TestCase):
    def test_export_deduplicates_canonically_equivalent_text(self) -> None:
        seeds = read_seeds(DEFAULT_SEEDS)
        seed = seeds["light_001"]
        text = "Возле остановки ёлка, фонарь не горит вечером."
        rows = [
            {**seed, "language": "RU", "style": "short", "appeal_text": text},
            {**seed, "language": "RU", "style": "neutral",
             "appeal_text": unicodedata.normalize("NFD", text)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.jsonl"
            counts = export_candidates(rows, seeds, output, "local-test-model", source_checksum(DEFAULT_SEEDS), 109)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(counts["exact_duplicate"], 1)

    def test_generation_projection_contains_only_scalar_seed_columns(self) -> None:
        seeds = read_seeds(DEFAULT_SEEDS)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation.jsonl"
            write_generation_seeds(seeds, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(seeds))
            self.assertTrue(all(set(row) == set(GENERATION_SEED_FIELDS) for row in rows))
            self.assertTrue(all(all(isinstance(value, str) for value in row.values()) for row in rows))

    def test_seed_metadata_is_grounded_and_pending(self) -> None:
        seed = next(iter(read_seeds(DEFAULT_SEEDS).values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.jsonl"
            for change in (
                {"critical_facts": []},
                {"critical_facts": ["Новая неподтверждённая деталь"]},
                {"forbidden_invented_facts": []},
                {"source_provenance": "REAL_CUSTOMER_APPEAL"},
                {"review_status": "APPROVED"},
                {"object_type": "неизвестный объект"},
            ):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**seed, **change}, ensure_ascii=False) + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_seeds(path)

    def test_export_keeps_review_pending_and_stable_provenance(self) -> None:
        seeds = read_seeds(DEFAULT_SEEDS)
        seed = next(iter(seeds.values()))
        rows = [
            {**seed, "language": "RU", "style": "short", "appeal_text": "Во дворе сегодня не работает уличный фонарь."},
            {**seed, "language": "KZ", "style": "neutral", "appeal_text": "Аулада бүгін шам жанбай тұр, тексеріңізші."},
            {**seed, "language": "RU", "style": "short", "appeal_text": "ИИН 000000000000"},
        ]
        digest = source_checksum(DEFAULT_SEEDS)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            counts = export_candidates(rows, seeds, first, "local-test-model", digest, 109)
            export_candidates(list(reversed(rows)), seeds, second, "local-test-model", digest, 109)
            candidates = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            reversed_candidates = [json.loads(line) for line in second.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(counts["pending_review"], 2)
        self.assertEqual(counts["invalid_text"], 1)
        self.assertEqual({row["text"]: row["variant_id"] for row in candidates},
                         {row["text"]: row["variant_id"] for row in reversed_candidates})
        self.assertTrue(all(row["review_status"] == "PENDING" for row in candidates))
        self.assertTrue(all(row["source_scenario_sha256"] == digest for row in candidates))
        self.assertTrue(all(row["prompt_version"] == PROMPT_VERSION and row["generator_seed"] == 109 for row in candidates))
        self.assertNotIn("000000000000", json.dumps(candidates, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
