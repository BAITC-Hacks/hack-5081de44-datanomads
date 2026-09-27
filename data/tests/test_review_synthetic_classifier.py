from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.pulse_sdg import DEFAULT_SEEDS, export_candidates, read_seeds, source_checksum
from scripts.review_synthetic_classifier import CHECKS, export_approved, prepare, read_reviews


class SyntheticClassifierReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.candidates = root / "candidates.jsonl"
        self.review = root / "review.jsonl"
        self.approved = root / "approved.jsonl"
        seeds = read_seeds(DEFAULT_SEEDS)
        seed = seeds["light_001"]
        rows = [
            {**seed, "language": "RU", "style": "short", "appeal_text": "Возле остановки ночью не работает уличный фонарь."},
            {**seed, "language": "KZ", "style": "neutral", "appeal_text": "Аялдама жанындағы көше шамы түнде жанбайды."},
        ]
        export_candidates(rows, seeds, self.candidates, "local-model", source_checksum(DEFAULT_SEEDS), 109)
        self.assertEqual(prepare(self.candidates, DEFAULT_SEEDS, self.review), 2)

    def _edit(self, rows: list[dict]) -> None:
        self.review.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def _rows(self) -> list[dict]:
        return [json.loads(line) for line in self.review.read_text(encoding="utf-8").splitlines()]

    def _decide(self, row: dict, decision: str, reason: str) -> None:
        row["decision"] = decision
        row["reviewer_id"] = "reviewer_1"
        row["reviewed_at"] = "2026-09-27T12:00:00+05:00"
        row["review_reason"] = reason
        if decision == "APPROVED":
            row["checks"] = {check: True for check in CHECKS}

    def test_queue_stays_pending_and_cannot_export(self) -> None:
        seed = read_seeds(DEFAULT_SEEDS)["light_001"]
        row = self._rows()[0]
        self.assertEqual(row["scenario_critical_facts"], seed["critical_facts"])
        self.assertEqual(row["scenario_forbidden_invented_facts"], seed["forbidden_invented_facts"])
        self.assertEqual(row["scenario_source_provenance"], seed["source_provenance"])
        self.assertEqual(row["scenario_review_status"], "PENDING")
        self.assertEqual(row["scenario_context"]["object_type"], seed["object_type"])
        self.assertEqual(len(read_reviews(self.review, self.candidates, DEFAULT_SEEDS)), 2)
        with self.assertRaisesRegex(ValueError, "no human-approved"):
            export_approved(self.review, self.candidates, DEFAULT_SEEDS, self.approved)
        self.assertFalse(self.approved.exists())

    def test_only_checked_approved_record_exports_with_review_checksum(self) -> None:
        rows = self._rows()
        self._decide(rows[0], "APPROVED", "VERIFIED")
        self._decide(rows[1], "REJECTED", "WRONG_LANGUAGE")
        rows[1]["checks"]["language_correct"] = False
        self._edit(rows)
        self.assertEqual(export_approved(self.review, self.candidates, DEFAULT_SEEDS, self.approved), 1)
        approved = json.loads(self.approved.read_text(encoding="utf-8"))
        self.assertEqual(approved["review_status"], "APPROVED")
        self.assertEqual(approved["review_evidence_sha256"], source_checksum(self.review))
        self.assertEqual(approved["source_scenario_sha256"], source_checksum(DEFAULT_SEEDS))
        self.assertEqual(approved["variant_id"], rows[0]["candidate"]["variant_id"])
        self.assertNotIn("scenario_facts_ru", approved)
        self.assertNotIn("checks", approved)

    def test_approval_requires_all_checks_and_timezone(self) -> None:
        rows = self._rows()
        self._decide(rows[0], "APPROVED", "VERIFIED")
        rows[0]["checks"]["facts_preserved"] = None
        self._edit(rows)
        with self.assertRaisesRegex(ValueError, "every human check"):
            read_reviews(self.review, self.candidates, DEFAULT_SEEDS)
        rows[0]["checks"]["facts_preserved"] = True
        rows[0]["reviewed_at"] = "2026-09-27T12:00:00"
        self._edit(rows)
        with self.assertRaisesRegex(ValueError, "timezone"):
            read_reviews(self.review, self.candidates, DEFAULT_SEEDS)

    def test_source_binding_and_complete_queue(self) -> None:
        rows = self._rows()
        rows[0]["scenario_critical_facts"] = ["Подменённый факт"]
        self._edit(rows)
        with self.assertRaisesRegex(ValueError, "candidate or scenario changed"):
            read_reviews(self.review, self.candidates, DEFAULT_SEEDS)
        rows = self._rows()
        rows[0]["scenario_critical_facts"] = read_seeds(DEFAULT_SEEDS)["light_001"]["critical_facts"]
        rows[0]["candidate"]["text"] = "Подменённый текст"
        self._edit(rows)
        with self.assertRaisesRegex(ValueError, "candidate or scenario changed"):
            read_reviews(self.review, self.candidates, DEFAULT_SEEDS)
        self._edit(rows[1:])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_reviews(self.review, self.candidates, DEFAULT_SEEDS)

    def test_rejects_sensitive_candidate_before_queue(self) -> None:
        rows = [json.loads(line) for line in self.candidates.read_text(encoding="utf-8").splitlines()]
        rows[0]["text"] = "ИИН 000000000000"
        self.candidates.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "invalid origin"):
            prepare(self.candidates, DEFAULT_SEEDS, Path(self.temporary.name) / "other.jsonl")


if __name__ == "__main__":
    unittest.main()
