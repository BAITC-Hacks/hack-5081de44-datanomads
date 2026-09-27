from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.dataset_builder import build_package, checksum
from training.repeat_windows import evaluate_repeat_windows
from test_dataset_builder import fixture_inputs, write_jsonl


class RepeatWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        inputs = fixture_inputs(self.root, groups_per_topic=3, retrieval_groups=3, prefix="repeat")
        retrieval = inputs[5]
        for index in range(3):
            group = f"repeat_relation_{index}"
            retrieval.append({
                **next(row for row in retrieval if row["relation_group"] == group),
                "pair_id": f"{group}_repeat",
                "candidate_id": f"{group}_repeat_candidate",
                "candidate_text": f"Повторное обращение {group}",
                "relation_label": "REPEAT",
            })
        write_jsonl(inputs[1], retrieval)
        build_package(*inputs[:4], self.root, "dataset_v1", "eval_v1", 109)
        self.package = self.root / "dataset_v1"
        self.source_path = self.root / "temporal_source.jsonl"
        self.evidence_path = self.root / "temporal.jsonl"
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_text(json.dumps({
            "policy_version": "repeat-window.v1", "window_days": [7, 14, 30],
            "min_repeat_count": 1, "min_nonrepeat_count": 1,
            "min_predictions": 1, "min_precision": 0.9,
        }), encoding="utf-8")
        self.rows = []
        for split in ("validation", "test"):
            pairs = [json.loads(line) for line in
                     (self.package / f"retrieval/{split}_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
            for pair in pairs:
                label = pair["relation_label"]
                repeat = label == "REPEAT"
                hard_negative = label == "SIMILAR_BUT_NOT_DUPLICATE"
                self.rows.append({
                    "evidence_version": "repeat-temporal-evidence.v1",
                    "pair_id": pair["pair_id"],
                    "source_relation_sha256": pair["source_relation_sha256"],
                    "review_evidence_sha256": pair["review_evidence_sha256"],
                    "temporal_source_sha256": "",
                    "review_status": "APPROVED",
                    "reviewer_id": "temporal_reviewer",
                    "reviewed_at": "2026-09-27T12:00:00Z",
                    "query_created_at": "2026-09-20T12:00:00Z",
                    "candidate_created_at": "2026-08-01T12:00:00Z",
                    "candidate_resolved_at": ("2026-09-10T12:00:00Z" if repeat else
                                              "2026-08-06T12:00:00Z" if hard_negative else None),
                    "same_region": label in {"DUPLICATE", "REPEAT", "SIMILAR_BUT_NOT_DUPLICATE"},
                    "same_object": label in {"DUPLICATE", "REPEAT"},
                    "same_issue": label in {"DUPLICATE", "REPEAT"},
                    "same_episode": label == "DUPLICATE",
                    "prior_episode_resolved": repeat or hard_negative,
                })
        self._write_source()
        self._write()

    def _write_source(self) -> None:
        write_jsonl(self.source_path, [{key: row[key] for key in
                                      ("pair_id", "query_created_at", "candidate_created_at", "candidate_resolved_at")}
                                      for row in self.rows])
        for row in self.rows:
            row["temporal_source_sha256"] = checksum(self.source_path)

    def _write(self) -> None:
        write_jsonl(self.evidence_path, self.rows)

    def test_selects_window_on_validation_then_checks_frozen_test(self) -> None:
        report = evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(report["validation"]["selected_window_days"], 14)
        self.assertEqual(report["test"]["fn"], 0)
        self.assertEqual(report["runtime_window_status"], "NOT_APPROVED")
        self.assertNotIn("pair_id", json.dumps(report))
        self.assertNotIn("query_text", json.dumps(report))

        test_pair_ids = {json.loads(line)["pair_id"] for line in
                         (self.package / "retrieval/test_pairs.jsonl").read_text(encoding="utf-8").splitlines()}
        for row in self.rows:
            if row["pair_id"] in test_pair_ids and row["prior_episode_resolved"]:
                row["candidate_resolved_at"] = "2026-09-01T12:00:00Z"
        self._write_source()
        self._write()
        changed = evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self.assertEqual(changed["validation"]["selected_window_days"], 14)
        self.assertEqual(changed["status"], "INSUFFICIENT_TEST_EVIDENCE")
        self.assertEqual(changed["test"]["fn"], 1)

    def test_time_signal_exposes_different_object_false_positive(self) -> None:
        similar = next(row for row in self.rows if row["pair_id"].endswith("similar_but_not_duplicate"))
        similar["candidate_resolved_at"] = "2026-09-15T12:00:00Z"
        self._write_source()
        self._write()
        report = evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self.assertEqual(report["validation"]["selected_window_days"], None)
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["validation"]["curve"][1]["fp"], 1)
        self.assertEqual(len(report["validation"]["curve"][1]["false_positive_pair_hashes"]), 1)

    def test_requires_hard_negative_in_each_evaluated_split(self) -> None:
        validation_ids = {json.loads(line)["pair_id"] for line in
                          (self.package / "retrieval/validation_pairs.jsonl").read_text(encoding="utf-8").splitlines()}
        similar = next(row for row in self.rows if row["pair_id"] in validation_ids and
                       row["pair_id"].endswith("similar_but_not_duplicate"))
        similar["same_region"] = False
        self._write()
        report = evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["validation"]["hard_negative_count"], 0)
        self.assertIsNone(report["validation"]["selected_window_days"])

    def test_rejects_candidate_created_after_query(self) -> None:
        negative = next(row for row in self.rows if row["candidate_resolved_at"] is None)
        negative["candidate_created_at"] = "2026-09-21T12:00:00Z"
        self._write_source()
        self._write()
        with self.assertRaisesRegex(ValueError, "chronology"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)

    def test_rejects_missing_or_unapproved_temporal_evidence(self) -> None:
        self.rows.pop()
        self._write()
        with self.assertRaisesRegex(ValueError, "cover every"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self.rows[0]["review_status"] = "PENDING"
        self._write()
        with self.assertRaisesRegex(ValueError, "unapproved"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)

    def test_rejects_future_resolution_and_changed_pair_provenance(self) -> None:
        repeat = next(row for row in self.rows if row["prior_episode_resolved"])
        repeat["candidate_resolved_at"] = "2026-09-21T12:00:00Z"
        self._write()
        with self.assertRaisesRegex(ValueError, "mismatched"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        self._write_source()
        self._write()
        with self.assertRaisesRegex(ValueError, "chronology"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)
        repeat["candidate_resolved_at"] = "2026-09-10T12:00:00Z"
        self._write_source()
        repeat["review_evidence_sha256"] = "sha256:" + "f" * 64
        self._write()
        with self.assertRaisesRegex(ValueError, "mismatched"):
            evaluate_repeat_windows(self.package, self.source_path, self.evidence_path, self.policy_path)


if __name__ == "__main__":
    unittest.main()
