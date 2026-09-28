from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_spike_review import build_report, review_candidates
from scripts.evaluate_spike_incidents import evaluate_incident_registry
from scripts.prepare_spike_incident_review import prepare_template


class SpikeReviewExportTests(unittest.TestCase):
    def test_registry_template_is_complete_blind_and_unapproved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["creation_date", "unused_text"])
                for offset in range(65):
                    day = date(2025, 1, 1) + timedelta(days=offset)
                    writer.writerow([day.strftime("%d.%m.%Y 10:00:00"), "PRIVATE_SENTINEL"])
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps(build_report(source)), encoding="utf-8")
            output = root / "registry.json"
            template = prepare_template(source, candidate, output)

            self.assertEqual(len(template["daily_labels"]), len(build_report(source)["evaluated_dates"]))
            self.assertTrue(all(row["review_status"] == "PENDING" and row["incident_ids"] is None
                                for row in template["daily_labels"]))
            self.assertFalse(template["coverage_assertion"]["all_evaluated_dates_reviewed"])
            self.assertFalse(template["coverage_assertion"]["independent_of_detector"])
            self.assertEqual(template["candidate_report_sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())
            self.assertNotIn("PRIVATE_SENTINEL", output.read_text(encoding="utf-8"))
            self.assertNotIn("triggered_rules", output.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "complete independent coverage"):
                evaluate_incident_registry(build_report(source), template, template["candidate_report_sha256"])
            with self.assertRaises(FileExistsError):
                prepare_template(source, candidate, output)
            stale = json.loads(candidate.read_text(encoding="utf-8"))
            stale["source_sha256"] = "0" * 64
            candidate.write_text(json.dumps(stale), encoding="utf-8")
            rejected_output = root / "stale-registry.json"
            with self.assertRaisesRegex(ValueError, "differs from regenerated source"):
                prepare_template(source, candidate, rejected_output)
            self.assertFalse(rejected_output.exists())

    def test_exports_every_candidate_and_matches_exploratory_rule_counts(self) -> None:
        series = [40] * 300
        for index in range(64, 273, 8):
            series[index] = 200

        result = review_candidates(series, date(2025, 1, 1))
        self.assertGreater(result["review_item_count"], 20)
        self.assertEqual(result["review_item_count"], len(result["review_items"]))
        self.assertTrue(all(item["review_status"] == "PENDING" for item in result["review_items"]))
        for rule in result["rules"]:
            matching = [
                trigger
                for item in result["review_items"]
                for trigger in item["triggered_rules"]
                if trigger["method"] == rule["method"] and trigger["threshold"] == rule["threshold"]
            ]
            self.assertEqual(len(matching), rule["raw_alert_count"])
            self.assertEqual(
                sum(trigger["emitted_after_cooldown"] for trigger in matching),
                rule["cooldown_alert_count"],
            )

    def test_missing_history_has_no_candidates(self) -> None:
        result = review_candidates([40] * 56, date(2025, 1, 1))
        self.assertEqual(result["status"], "INSUFFICIENT_HISTORY")
        self.assertEqual(result["review_items"], [])

    def test_report_has_source_provenance_without_raw_text_or_quality_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["creation_date", "unused_text"])
                for offset in range(65):
                    day = date(2025, 1, 1) + timedelta(days=offset)
                    for _ in range(120 if offset == 63 else 40):
                        writer.writerow([day.strftime("%d.%m.%Y 10:00:00"), "PRIVATE_SENTINEL"])

            report = build_report(path)
            self.assertEqual(report["report_version"], "spike-review-candidates.v2")
            self.assertEqual(report["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report["evaluated_days"], len(report["evaluated_dates"]))
            self.assertEqual(
                report["evaluated_date_sha256"],
                hashlib.sha256("\n".join(report["evaluated_dates"]).encode("ascii")).hexdigest(),
            )
            self.assertEqual(report["record_count"], 65 * 40 + 80)
            self.assertEqual(report["review_item_count"], 1)
            self.assertEqual(report["review_items"][0]["date"], "2025-03-05")
            self.assertEqual(report["review_items"][0]["daily_count"], 120)
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(report))
            self.assertIsNone(report["precision"])
            self.assertIsNone(report["recall"])
            self.assertFalse(report["runtime_eligible"])
            self.assertEqual(report["incident_ground_truth_requirements"]["status"], "MISSING")


if __name__ == "__main__":
    unittest.main()
