from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from scripts.evaluate_forecast_candidates import evaluate_series
from scripts.evaluate_forecast_csv import HORIZONS, evaluate_horizon


class ForecastCandidateTests(unittest.TestCase):
    def test_models_share_windows_and_baseline_metrics(self) -> None:
        series = [10 + day % 7 for day in range(500)]
        reports = evaluate_series(series, date(2024, 1, 1))
        self.assertEqual([report["horizon_days"] for report in reports], list(HORIZONS))
        for report in reports:
            horizon = report["horizon_days"]
            self.assertEqual(report["status"], "EVALUATED")
            expected = evaluate_horizon(series, horizon)
            self.assertEqual(report["models"]["seasonal_naive"],
                             {key: value for key, value in expected.items() if key not in {"horizon_days", "status"}})
            self.assertEqual(report["models"]["prophet"]["window_count"], report["models"]["seasonal_naive"]["window_count"])
            self.assertEqual(report["models"]["prophet"]["sample_count"], report["models"]["seasonal_naive"]["sample_count"])
            self.assertEqual(report["best_on_backtest"], "seasonal_naive")

    def test_short_series_has_no_candidate_metrics(self) -> None:
        reports = evaluate_series([1] * 300, date(2025, 1, 1))
        self.assertTrue(all(report["status"] == "INSUFFICIENT_HISTORY" for report in reports))
        self.assertTrue(all("models" not in report for report in reports))

        gapped = evaluate_series([None if day % 100 == 0 else 10 for day in range(500)], date(2025, 1, 1))
        self.assertTrue(all(report["reason"] == "NO_FULLY_OBSERVED_WINDOW" for report in gapped))
        self.assertTrue(all("models" not in report for report in gapped))

    def test_gap_excludes_windows_and_models_share_remaining_origins(self) -> None:
        first_day = date(2024, 1, 1)
        series = [1] * 10 + [None] + [10 + day % 7 for day in range(500)]
        with patch("scripts.evaluate_forecast_candidates._prophet_predict",
                   side_effect=lambda history, _first_day, horizon: [float(history[-7 + offset % 7])
                                                               for offset in range(horizon)]) as predict:
            reports = evaluate_series(series, first_day)
        self.assertEqual(predict.call_args_list[0].args[1], date(2024, 1, 12))
        for report in reports:
            expected = evaluate_horizon(series, report["horizon_days"])
            self.assertEqual(report["models"]["seasonal_naive"]["window_count"], expected["window_count"])
            self.assertEqual(report["models"]["seasonal_naive"]["mae"], expected["mae"])


if __name__ == "__main__":
    unittest.main()
