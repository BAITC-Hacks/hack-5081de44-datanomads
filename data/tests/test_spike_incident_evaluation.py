from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluate_spike_incidents import evaluate_incident_registry
from scripts.export_spike_review import build_report, review_candidates


class SpikeIncidentEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        series = [40] * 65
        series[63] = 120
        first_day = date(2025, 1, 1)
        candidate = review_candidates(series, first_day)
        self.alert_date = (first_day + timedelta(days=63)).isoformat()
        self.quiet_date = (first_day + timedelta(days=64)).isoformat()
        self.candidate = {
            "report_version": "spike-review-candidates.v2",
            "source_sha256": "a" * 64,
            "scope": "total_daily_appeals_in_one_export",
            "unit": "one_region_total_per_day",
            **candidate,
        }
        self.registry = {
            "schema_version": "spike-incident-registry.v1",
            "source_sha256": self.candidate["source_sha256"],
            "scope": self.candidate["scope"],
            "unit": self.candidate["unit"],
            "evaluated_date_sha256": self.candidate["evaluated_date_sha256"],
            "candidate_report_sha256": self._candidate_digest(),
            "coverage_assertion": {
                "all_evaluated_dates_reviewed": True,
                "independent_of_detector": True,
                "reviewer_id": "reviewer-7",
                "reviewed_at": "2025-04-01T12:00:00+00:00",
            },
            "daily_labels": [
                {"date": day, "incident_ids": [], "review_status": "CONFIRMED"}
                for day in self.candidate["evaluated_dates"]
            ],
            "incidents": [],
        }

    def _candidate_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.candidate).encode("utf-8")).hexdigest()

    def _evaluate(self) -> dict:
        return evaluate_incident_registry(self.candidate, self.registry, self._candidate_digest())

    def _add_incident(self, incident_id: str, day: str) -> None:
        self.registry["incidents"].append({
            "incident_id": incident_id,
            "onset_date": day,
            "end_date": day,
            "review_status": "CONFIRMED",
        })
        next(label for label in self.registry["daily_labels"] if label["date"] == day)["incident_ids"].append(incident_id)

    def test_confirmed_alert_has_precision_recall_and_detection_delay(self) -> None:
        self._add_incident("I-1", self.alert_date)
        report = self._evaluate()
        self.assertFalse(report["runtime_eligible"])
        self.assertEqual(report["selection_status"], "NO_RUNTIME_THRESHOLD_SELECTED")
        self.assertEqual(len(report["rules"]), 6)
        for rule in report["rules"]:
            self.assertEqual(rule["emitted_alert_count"], 1)
            self.assertEqual(rule["precision"], 1.0)
            self.assertEqual(rule["recall"], 1.0)
            self.assertEqual(rule["false_alert_count"], 0)
            self.assertEqual(rule["time_to_detect_days"], [0])

    def test_false_alert_and_missed_incident_are_counted(self) -> None:
        self._add_incident("I-quiet", self.quiet_date)
        report = self._evaluate()
        for rule in report["rules"]:
            self.assertEqual(rule["precision"], 0.0)
            self.assertEqual(rule["recall"], 0.0)
            self.assertEqual(rule["false_alert_dates_for_review"], [self.alert_date])
            self.assertEqual(rule["false_alerts_per_30_evaluated_days"], round(30 / 9, 4))
            self.assertIsNone(rule["mean_time_to_detect_days_for_detected_incidents"])

    def test_no_incidents_does_not_claim_recall(self) -> None:
        report = self._evaluate()
        self.assertTrue(all(rule["recall"] is None for rule in report["rules"]))
        self.assertTrue(all(rule["precision"] == 0.0 for rule in report["rules"]))

    def test_rejects_missing_and_unconfirmed_daily_coverage(self) -> None:
        self.registry["daily_labels"].pop()
        with self.assertRaisesRegex(ValueError, "cover every evaluated date"):
            self._evaluate()
        self.registry["daily_labels"].append({
            "date": self.quiet_date, "incident_ids": [], "review_status": "PENDING",
        })
        with self.assertRaisesRegex(ValueError, "unconfirmed"):
            self._evaluate()

    def test_rejects_unasserted_or_mismatched_provenance(self) -> None:
        self.registry["coverage_assertion"]["independent_of_detector"] = False
        with self.assertRaisesRegex(ValueError, "independent coverage"):
            self._evaluate()
        self.registry["coverage_assertion"]["independent_of_detector"] = True
        self.registry["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            self._evaluate()

    def test_rejects_contradictory_incident_interval(self) -> None:
        self._add_incident("I-1", self.alert_date)
        self.registry["incidents"][0]["end_date"] = self.quiet_date
        with self.assertRaisesRegex(ValueError, "contradict"):
            self._evaluate()

    def test_rejects_overlapping_incidents(self) -> None:
        self._add_incident("I-1", self.alert_date)
        self._add_incident("I-2", self.alert_date)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self._evaluate()

    def test_rejects_candidate_file_hash_change(self) -> None:
        self.registry["candidate_report_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "candidate_report_sha256"):
            self._evaluate()

    def test_rejects_candidate_date_digest_or_alert_count_changes(self) -> None:
        self.candidate["evaluated_date_sha256"] = hashlib.sha256(b"altered").hexdigest()
        with self.assertRaisesRegex(ValueError, "digest differs"):
            self._evaluate()
        self.candidate["evaluated_date_sha256"] = self.registry["evaluated_date_sha256"]
        self.candidate["rules"][0]["cooldown_alert_count"] += 1
        self.registry["candidate_report_sha256"] = self._candidate_digest()
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            self._evaluate()

    def test_cli_evaluates_exact_source_and_rejects_modified_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["creation_date"])
                for offset in range(65):
                    day = date(2025, 1, 1) + timedelta(days=offset)
                    for _ in range(120 if offset == 63 else 40):
                        writer.writerow([day.strftime("%d.%m.%Y 10:00:00")])
            candidate = build_report(source)
            self.registry["source_sha256"] = candidate["source_sha256"]
            candidate_file = folder / "candidate.json"
            candidate_file.write_text(json.dumps(candidate), encoding="utf-8")
            self.registry["candidate_report_sha256"] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
            registry_file = folder / "registry.json"
            registry_file.write_text(json.dumps(self.registry), encoding="utf-8")
            output = folder / "evaluation.json"
            command = [
                sys.executable, "scripts/evaluate_spike_incidents.py", str(source),
                str(candidate_file), str(registry_file), "--output", str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["coverage_status"], "COMPLETE_ASSERTED_AND_DATE_VERIFIED")

            candidate["review_items"][0]["daily_count"] += 1
            candidate_file.write_text(json.dumps(candidate), encoding="utf-8")
            self.registry["candidate_report_sha256"] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
            registry_file.write_text(json.dumps(self.registry), encoding="utf-8")
            output.unlink()
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertNotIn("source.csv", result.stderr)

            candidate["review_items"][0]["daily_count"] -= 1
            candidate_file.write_text(json.dumps(candidate), encoding="utf-8")
            self.registry["candidate_report_sha256"] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
            registry_file.write_text(
                json.dumps(self.registry).replace('"schema_version":',
                                                  '"schema_version": "spike-incident-registry.v1", "schema_version":', 1),
                encoding="utf-8",
            )
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
