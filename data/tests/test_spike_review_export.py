from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_spike_review import build_report, review_candidates


class SpikeReviewExportTests(unittest.TestCase):
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
            self.assertEqual(report["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
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
