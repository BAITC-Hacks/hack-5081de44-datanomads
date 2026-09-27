from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_retrieval_relations import (
    CHECKS, checksum, export_approved, prepare, read_reviews, read_source,
)


PILOT = Path(__file__).resolve().parents[1] / "sdg/pilot_relations.jsonl"


class RetrievalRelationReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.review = root / "review.jsonl"
        self.export = root / "approved.jsonl"
        self.assertEqual(prepare(PILOT, self.review), 15)

    def _rows(self) -> list[dict]:
        return [json.loads(line) for line in self.review.read_text(encoding="utf-8").splitlines()]

    def _write(self, rows: list[dict]) -> None:
        self.review.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    @staticmethod
    def _approve(row: dict, label: str) -> None:
        row.update({
            "decision": "APPROVED",
            "relation_label": label,
            "reviewer_id": "reviewer_1",
            "reviewed_at": "2026-09-27T12:00:00+05:00",
            "review_reason": "VERIFIED",
            "checks": {check: True for check in CHECKS},
            "same_region": True,
            "same_object": True,
            "same_issue": True,
            "same_episode": True,
            "prior_episode_resolved": False,
        })
        if label in {"SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"}:
            row["same_object"] = False
            row["same_issue"] = False
            row["same_episode"] = False

    @staticmethod
    def _reject(row: dict) -> None:
        row.update({
            "decision": "REJECTED",
            "reviewer_id": "reviewer_1",
            "reviewed_at": "2026-09-27T12:00:00+05:00",
            "review_reason": "OTHER_QUALITY",
        })

    def test_pending_pilot_cannot_export(self) -> None:
        self.assertEqual(len(read_reviews(self.review, PILOT)), 15)
        with self.assertRaisesRegex(ValueError, "no human-approved"):
            export_approved(self.review, PILOT, self.export)
        self.assertFalse(self.export.exists())

    def test_approved_pair_has_source_and_review_checksums(self) -> None:
        rows = self._rows()
        self._approve(rows[0], "DUPLICATE")
        self._approve(rows[1], "SIMILAR_BUT_NOT_DUPLICATE")
        self._approve(rows[2], "UNRELATED")
        for row in rows[3:]:
            self._reject(row)
        self._write(rows)
        self.assertEqual(export_approved(self.review, PILOT, self.export), 3)
        pair = json.loads(self.export.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(pair["relation_label"], "DUPLICATE")
        self.assertEqual(pair["review_status"], "APPROVED")
        self.assertEqual(pair["source_relation_sha256"], checksum(PILOT))
        self.assertEqual(pair["review_evidence_sha256"], checksum(self.review))
        self.assertNotIn("query_context", pair)
        self.assertNotIn("same_episode", pair)

    def test_export_requires_adjudicated_candidate_set(self) -> None:
        rows = self._rows()
        self._approve(rows[0], "DUPLICATE")
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            export_approved(self.review, PILOT, self.export)
        for row in rows[1:]:
            self._reject(row)
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "at least two"):
            export_approved(self.review, PILOT, self.export)
        self._approve(rows[1], "SIMILAR_BUT_NOT_DUPLICATE")
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "required relation labels"):
            export_approved(self.review, PILOT, self.export)
        self.assertFalse(self.export.exists())

    def test_semantic_gate_rejects_wrong_duplicate_or_unproven_repeat(self) -> None:
        rows = self._rows()
        self._approve(rows[0], "DUPLICATE")
        rows[0]["same_episode"] = False
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "duplicate needs same"):
            read_reviews(self.review, PILOT)
        self._approve(rows[0], "REPEAT")
        rows[0]["same_episode"] = False
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "resolved prior episode"):
            read_reviews(self.review, PILOT)
        rows[0]["prior_episode_resolved"] = True
        self._write(rows)
        self.assertEqual(len(read_reviews(self.review, PILOT)), 15)

        self._approve(rows[0], "SIMILAR_BUT_NOT_DUPLICATE")
        rows[0]["same_object"] = True
        rows[0]["same_issue"] = True
        rows[0]["same_episode"] = False
        rows[0]["prior_episode_resolved"] = True
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "resolved recurrence"):
            read_reviews(self.review, PILOT)

    def test_cannot_change_source_or_skip_review_rows(self) -> None:
        rows = self._rows()
        rows[0]["source"]["candidate_text"] = "Подменённый текст"
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "source pair changed"):
            read_reviews(self.review, PILOT)
        self._write(rows[1:])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_reviews(self.review, PILOT)

    def test_source_rejects_cross_group_entity_and_sensitive_context(self) -> None:
        rows = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines()]
        modified = Path(self.temporary.name) / "modified.jsonl"
        road = next(index for index, row in enumerate(rows) if row["relation_group"] == "rel_road")
        rows[road]["query_id"] = rows[0]["query_id"]
        modified.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "entity crosses groups"):
            read_source(modified)
        rows[road]["query_id"] = "rel_road_q"
        rows[road]["query_context"] = "ИИН 000000000000"
        modified.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sensitive"):
            read_source(modified)


if __name__ == "__main__":
    unittest.main()
