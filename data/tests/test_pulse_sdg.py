from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.pulse_sdg import DEFAULT_SEEDS, PROMPT_VERSION, export_candidates, read_seeds, source_checksum


class PilotSdgTests(unittest.TestCase):
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
