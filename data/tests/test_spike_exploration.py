from __future__ import annotations

from datetime import date
import unittest

from scripts.evaluate_spike_csv import evaluate_series, score_series


class SpikeExplorationTests(unittest.TestCase):
    def test_injected_spike_is_flagged_without_future_leakage(self) -> None:
        first_day = date(2025, 1, 1)
        series = [40 + index % 7 for index in range(90)]
        series[63] = 150
        before = score_series(series, first_day)
        changed = series.copy()
        changed[89] = 999
        after = score_series(changed, first_day)
        self.assertEqual(before[:20], after[:20])
        report = evaluate_series(series, first_day)
        self.assertEqual(report["status"], "EXPLORATORY_NO_GROUND_TRUTH")
        self.assertIn("2025-03-05", {row["date"] for row in report["review_queue"]})
        self.assertTrue(all(row["review_status"] == "PENDING" for row in report["review_queue"]))
        self.assertGreater(report["rules"][0]["raw_alert_count"], 0)

    def test_short_history_does_not_invent_alert_metrics(self) -> None:
        report = evaluate_series([10] * 56, date(2025, 1, 1))
        self.assertEqual(report["status"], "INSUFFICIENT_HISTORY")
        self.assertEqual(report["rules"], [])
        self.assertEqual(report["review_queue"], [])

    def test_cooldown_suppresses_repeated_alerts(self) -> None:
        series = [40] * 90
        series[63:66] = [120, 120, 120]
        report = evaluate_series(series, date(2025, 1, 1))
        ratio = next(rule for rule in report["rules"] if rule["method"] == "ratio" and rule["threshold"] == 2.0)
        self.assertEqual(ratio["raw_alert_count"], 3)
        self.assertEqual(ratio["cooldown_alert_count"], 1)


if __name__ == "__main__":
    unittest.main()
