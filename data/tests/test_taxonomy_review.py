from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_taxonomy import CATALOG, export_approved, prepare_almaty, read_reviews


class TaxonomyReviewTests(unittest.TestCase):
    def test_catalog_only_creates_pending_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "review.jsonl"
            self.assertEqual(prepare_almaty(CATALOG, queue), 242)
            rows = read_reviews(queue)
            self.assertEqual(len(rows), 242)
            self.assertTrue(all(row["decision"] == "PENDING" for row in rows))
            self.assertTrue(all(row["source_system"] is None for row in rows))
            self.assertTrue(all(row["approved_for_training"] is False and row["approved_for_routing"] is False for row in rows))
            with self.assertRaisesRegex(ValueError, "no human-approved"):
                export_approved(queue, Path(directory) / "approved.json")
            with self.assertRaises(FileExistsError):
                prepare_almaty(CATALOG, queue)

    def test_review_cannot_change_catalog_evidence_or_skip_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "review.jsonl"
            prepare_almaty(CATALOG, queue)
            rows = read_reviews(queue)
            rows[0]["observed_count"] += 1
            queue.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source proposal changed"):
                read_reviews(queue)
            rows[0]["observed_count"] -= 1
            queue.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_reviews(queue)

    def test_export_requires_explicit_human_provenance_and_keeps_composite_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "review.jsonl"
            approved_path = Path(directory) / "approved.json"
            prepare_almaty(CATALOG, queue)
            rows = read_reviews(queue)
            row = rows[0]
            row.update({
                "decision": "APPROVED",
                "source_system": "ikomek109",
                "source_profile_verified": True,
                "canonical_topic_id": "street_lighting",
                "canonical_subtopic_id": "fixture_subtopic",
                "reviewer": "reviewer_test",
                "reviewed_at": "2026-09-27T12:00:00Z",
                "evidence_ref": "sha256:" + "b" * 64,
                "review_reason": "SOURCE_CONTEXT_VERIFIED",
            })
            row["source_profile_verified"] = False
            queue.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified source"):
                export_approved(queue, approved_path)
            row["source_profile_verified"] = True
            row["approved_for_training"] = True
            queue.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot approve training"):
                export_approved(queue, approved_path)
            row["approved_for_training"] = False
            queue.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n", encoding="utf-8")
            self.assertEqual(export_approved(queue, approved_path), 1)
            exported = json.loads(approved_path.read_text(encoding="utf-8"))
            self.assertEqual(exported["mapping_key_fields"], ["source_system", "raw_direction", "raw_subdirection"])
            self.assertEqual(exported["mappings"][0]["raw_subdirection"], row["raw_subdirection"])
            self.assertFalse(exported["mappings"][0]["approved_for_training"])
            self.assertFalse(exported["mappings"][0]["approved_for_routing"])


if __name__ == "__main__":
    unittest.main()
