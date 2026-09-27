from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from scripts.evaluate_forecast_candidates import evaluate_series
from scripts.evaluate_forecast_csv import HORIZONS, evaluate_horizon


class ForecastCandidateTests(unittest.TestCase):
    def test_models_share_windows_and_baseline_metrics(self) -> None:
        series = [10 + day % 7 for day in range(500)]
        with patch("scripts.evaluate_forecast_candidates._prophet_predict",
                   side_effect=lambda history, _first_day, horizon:
                   [float(history[-7 + offset % 7]) for offset in range(horizon)]):
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

    def test_high_load_proxy_uses_only_history_and_shared_windows(self) -> None:
        series = [10] * 365 + [20, 10, 20] + [10] * 27
        with patch("scripts.evaluate_forecast_candidates._prophet_predict",
                   side_effect=lambda _history, _first_day, horizon:
                   [20.0 if offset in (0, 1) else 10.0 for offset in range(horizon)]):
            reports = evaluate_series(series, date(2024, 1, 1))

        proxy = reports[0]["high_load_day_proxy"]
        self.assertEqual(proxy["status"], "PROXY_ONLY_NO_REVIEWED_PEAK_LABELS")
        self.assertEqual(proxy["models"]["seasonal_naive"], {
            "tp": 0, "fp": 0, "fn": 2, "tn": 28,
            "precision": None, "recall": 0.0, "f1": 0.0,
        })
        self.assertEqual(proxy["models"]["prophet"], {
            "tp": 1, "fp": 1, "fn": 1, "tn": 27,
            "precision": 0.5, "recall": 0.5, "f1": 0.5,
        })
        self.assertEqual(reports[0]["best_on_backtest"], "seasonal_naive")
        self.assertTrue(all("high_load_day_proxy" not in report for report in reports[1:]))

    def test_high_load_proxy_has_undefined_metrics_without_positive_days(self) -> None:
        with patch("scripts.evaluate_forecast_candidates._prophet_predict",
                   side_effect=lambda _history, _first_day, horizon: [0.0] * horizon):
            report = evaluate_series([0] * 395, date(2024, 1, 1))[0]

        for model in ("seasonal_naive", "prophet"):
            self.assertEqual(report["high_load_day_proxy"]["models"][model], {
                "tp": 0, "fp": 0, "fn": 0, "tn": 30,
                "precision": None, "recall": None, "f1": None,
            })


if __name__ == "__main__":
    unittest.main()
