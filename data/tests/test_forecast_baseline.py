from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluate_forecast_csv import build_report, daily_counts, evaluate_horizon


class ForecastBaselineTests(unittest.TestCase):
    def test_weekly_pattern_is_evaluated_without_future_leakage(self) -> None:
        series = [10 + (day % 7) for day in range(500)]
        baseline = evaluate_horizon(series, 90)
        self.assertEqual(baseline["window_count"], 2)
        self.assertEqual(baseline["mae"], 0)

        series[370] += 100
        self.assertGreater(evaluate_horizon(series, 90)["mae"], 0)

    def test_csv_report_contains_only_aggregate_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["creation_date", "full_name"])
                for offset in range(500):
                    day = date(2024, 1, 1) + timedelta(days=offset)
                    writer.writerow([f"{day:%d.%m.%Y} 12:00:00", "Иван Иванов"])
            report = build_report(path)

        self.assertEqual(report["record_count"], 500)
        self.assertEqual(report["unobserved_calendar_days"], 0)
        self.assertNotIn("Иван Иванов", json.dumps(report, ensure_ascii=False))
        self.assertNotIn(str(path), json.dumps(report, ensure_ascii=False))

    def test_short_history_reports_insufficient_without_fake_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["creation_date"])
                for offset in range(300):
                    day = date(2025, 1, 1) + timedelta(days=offset)
                    writer.writerow([f"{day:%d.%m.%Y} 12:00:00"])
            report = build_report(path)
        self.assertEqual(report["status"], "INSUFFICIENT_HISTORY")
        self.assertTrue(all(result["status"] == "INSUFFICIENT_HISTORY" for result in report["results"]))
        self.assertTrue(all("mae" not in result for result in report["results"]))

    def test_missing_calendar_day_is_unknown_and_does_not_enter_backtest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "with_gap.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["creation_date"])
                for offset in range(512):
                    if offset == 10:
                        continue
                    day = date(2024, 1, 1) + timedelta(days=offset)
                    writer.writerow([f"{day:%d.%m.%Y} 12:00:00"])
            series, *_ = daily_counts(path)
            report = build_report(path)
        self.assertIsNone(series[10])
        self.assertEqual(report["unobserved_calendar_days"], 1)
        self.assertEqual(report["longest_observed_run_days"], 501)
        self.assertEqual(report["results"][0]["window_count"], 3)
        self.assertEqual(report["results"][0]["mae"], 0)

        target_gap = [10] * 500
        target_gap[400] = None
        self.assertEqual(evaluate_horizon(target_gap, 30)["window_count"], 1)

    def test_malformed_quoted_csv_fails_all_forecast_and_spike_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "malformed.csv"
            source.write_text(
                'creation_date,note\n01.01.2025 12:00:00,ok\n02.01.2025 12:00:00,"PRIVATE_SENTINEL\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CSV is malformed"):
                daily_counts(source)

            scripts = Path(__file__).resolve().parents[2] / "scripts"
            for name in (
                "evaluate_forecast_csv.py",
                "evaluate_forecast_candidates.py",
                "evaluate_spike_csv.py",
                "export_spike_review.py",
            ):
                with self.subTest(script=name):
                    output = Path(directory) / f"{name}.json"
                    result = subprocess.run(
                        [sys.executable, str(scripts / name), str(source), "--output", str(output)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(output.exists())
                    self.assertEqual(result.stdout, "")
                    self.assertNotIn("PRIVATE_SENTINEL", result.stderr)
                    self.assertEqual(json.loads(result.stderr)["error"], "ValueError")


if __name__ == "__main__":
    unittest.main()
