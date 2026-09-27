#!/usr/bin/env python3
"""Compare weekly seasonal naive and Prophet on identical rolling windows."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_forecast_csv import (
    HORIZONS, MIN_TRAIN_DAYS, MISSING_DAY_POLICY, ORIGIN_STEP_DAYS, SEASON_LENGTH,
    daily_counts, eligible_origins, observed_segments,
)


PROPHET_CONFIG = {
    "weekly_seasonality": True,
    "yearly_seasonality": False,
    "daily_seasonality": False,
    "uncertainty_samples": 0,
    "changepoint_prior_scale": 0.05,
}


def _prophet_predict(history: list[int], first_day: date, horizon: int) -> list[float]:
    import pandas as pd
    from prophet import Prophet

    training = pd.DataFrame({
        "ds": pd.date_range(first_day, periods=len(history), freq="D"),
        "y": history,
    })
    model = Prophet(**PROPHET_CONFIG)
    model.fit(training)
    future = pd.DataFrame({"ds": pd.date_range(first_day + timedelta(days=len(history)), periods=horizon, freq="D")})
    predictions = [float(value) for value in model.predict(future)["yhat"]]
    if any(not math.isfinite(value) for value in predictions):
        raise ValueError("Prophet produced a non-finite forecast")
    return [max(0.0, value) for value in predictions]


def _metrics(actual: list[int], predicted: list[float], window_count: int) -> dict:
    errors = [abs(value - estimate) for value, estimate in zip(actual, predicted)]
    actual_total = sum(actual)
    smape_terms = [2 * error / (value + estimate) for value, estimate, error in zip(actual, predicted, errors)
                   if value + estimate]
    return {
        "window_count": window_count,
        "sample_count": len(actual),
        "mae": round(sum(errors) / len(errors), 4),
        "rmse": round(math.sqrt(sum(error * error for error in errors) / len(errors)), 4),
        "wape": round(sum(errors) / actual_total, 4) if actual_total else None,
        "smape": round(sum(smape_terms) / len(smape_terms), 4) if smape_terms else 0.0,
    }


def evaluate_series(series: list[int | None], first_day: date) -> list[dict]:
    results = {horizon: {"actual": [], "seasonal_naive": [], "prophet": [], "window_count": 0}
               for horizon in HORIZONS}
    segments = observed_segments(series)
    longest_run = max((len(observed) for _, observed in segments), default=0)
    valid_origins = {horizon: dict(eligible_origins(series, horizon)) for horizon in HORIZONS}
    all_origins = sorted({origin for origins in valid_origins.values() for origin in origins})
    for origin in all_origins:
        available_horizons = [horizon for horizon in HORIZONS if origin in valid_origins[horizon]]
        history_start = valid_origins[available_horizons[0]][origin]
        max_horizon = max(available_horizons)
        history = series[history_start:origin]
        prophet = _prophet_predict(history, first_day + timedelta(days=history_start), max_horizon)
        weekly_pattern = history[-SEASON_LENGTH:]
        for horizon in available_horizons:
            result = results[horizon]
            result["actual"].extend(series[origin:origin + horizon])
            result["seasonal_naive"].extend(weekly_pattern[offset % SEASON_LENGTH] for offset in range(horizon))
            result["prophet"].extend(prophet[:horizon])
            result["window_count"] += 1
    reports = []
    for horizon in HORIZONS:
        result = results[horizon]
        if not result["window_count"]:
            reports.append({
                "horizon_days": horizon,
                "status": "INSUFFICIENT_HISTORY",
                "reason": "CALENDAR_TOO_SHORT" if len(series) < MIN_TRAIN_DAYS + horizon else "NO_FULLY_OBSERVED_WINDOW",
                "required_calendar_days": MIN_TRAIN_DAYS + horizon,
                "available_calendar_days": len(series),
                "longest_observed_run_days": longest_run,
                "window_count": 0,
            })
            continue
        baseline = _metrics(result["actual"], result["seasonal_naive"], result["window_count"])
        candidate = _metrics(result["actual"], result["prophet"], result["window_count"])
        if baseline["wape"] is None or candidate["wape"] is None:
            best = None
            selection_status = "NO_COMPARABLE_WAPE"
        else:
            best = "prophet" if candidate["wape"] < baseline["wape"] else "seasonal_naive"
            selection_status = "PROVISIONAL_BACKTEST_ONLY"
        reports.append({
            "horizon_days": horizon,
            "status": "EVALUATED",
            "models": {"seasonal_naive": baseline, "prophet": candidate},
            "selection_status": selection_status,
            "best_on_backtest": best,
        })
    return reports


def build_report(path: Path) -> dict:
    series, first_day, last_day, record_count, invalid_rows, digest = daily_counts(path)
    results = evaluate_series(series, first_day)
    evaluated_count = sum(result["status"] == "EVALUATED" for result in results)
    if evaluated_count == len(HORIZONS):
        status = "EVALUATED"
    elif evaluated_count:
        status = "PARTIAL"
    else:
        status = results[0]["status"]
    return {
        "report_version": "forecast-candidates.v2",
        "source_sha256": digest,
        "status": status,
        "scope": "total_daily_appeals_in_one_export",
        "first_date": first_day.isoformat(),
        "last_date": last_day.isoformat(),
        "record_count": record_count,
        "invalid_csv_or_date_row_count": invalid_rows,
        "calendar_days": len(series),
        "unobserved_calendar_days": series.count(None),
        "missing_day_policy": MISSING_DAY_POLICY,
        "longest_observed_run_days": max(len(observed) for _, observed in observed_segments(series)),
        "minimum_train_days": MIN_TRAIN_DAYS,
        "origin_step_days": ORIGIN_STEP_DAYS,
        "selection_policy": "lowest_wape_on_same_windows_tie_seasonal_naive.v1",
        "prophet_config": PROPHET_CONFIG,
        "prophet_version": version("prophet") if evaluated_count else None,
        "count_prediction_policy": "clip_negative_to_zero",
        "interval_status": "NOT_EVALUATED",
        "peak_detection_status": "NO_REVIEWED_PEAK_LABELS",
        "runtime_eligible": False,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "forecast evaluation failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "source_sha256": report["source_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
