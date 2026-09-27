#!/usr/bin/env python3
"""Compare exploratory daily spike rules on one regional 109 CSV export."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_forecast_csv import daily_counts


HISTORY_WEEKS = 8
MIN_COUNT = 20
RATIO_THRESHOLDS = (1.5, 2.0, 3.0)
ROBUST_Z_THRESHOLDS = (3.0, 4.0, 5.0)
COOLDOWN_DAYS = 7


def score_series(series: list[int], first_day: date) -> list[dict]:
    if any(type(value) is not int or value < 0 for value in series):
        raise ValueError("daily series must contain nonnegative integer counts")
    scored = []
    for index in range(HISTORY_WEEKS * 7, len(series)):
        history = [series[index - 7 * week] for week in range(1, HISTORY_WEEKS + 1)]
        expected = statistics.median(history)
        mad = statistics.median(abs(value - expected) for value in history)
        scale = max(1.0, math.sqrt(expected), 1.4826 * mad)
        observed = series[index]
        scored.append({
            "index": index,
            "date": (first_day + timedelta(days=index)).isoformat(),
            "count": observed,
            "expected": expected,
            "ratio": observed / expected if expected else None,
            "robust_z": max(0.0, (observed - expected) / scale),
        })
    return scored


def _cooldown_count(indices: list[int]) -> int:
    selected = 0
    previous = -COOLDOWN_DAYS
    for index in indices:
        if index - previous >= COOLDOWN_DAYS:
            selected += 1
            previous = index
    return selected


def evaluate_series(series: list[int], first_day: date) -> dict:
    scored = score_series(series, first_day)
    if not scored:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "required_calendar_days": HISTORY_WEEKS * 7 + 1,
            "available_calendar_days": len(series),
            "evaluated_days": 0,
            "rules": [],
            "review_queue": [],
        }
    rules = []
    for method, thresholds in (("ratio", RATIO_THRESHOLDS), ("weekday_robust_z", ROBUST_Z_THRESHOLDS)):
        for threshold in thresholds:
            alerts = [
                row["index"] for row in scored
                if row["count"] >= MIN_COUNT and (
                    row["count"] >= row["expected"] * threshold if method == "ratio"
                    else row["robust_z"] >= threshold
                )
            ]
            rules.append({
                "method": method,
                "threshold": threshold,
                "minimum_count": MIN_COUNT,
                "raw_alert_count": len(alerts),
                "cooldown_alert_count": _cooldown_count(alerts),
                "cooldown_days": COOLDOWN_DAYS,
                "raw_alerts_per_30_days": round(len(alerts) * 30 / len(scored), 4),
            })
    review = sorted(
        (row for row in scored if row["count"] >= MIN_COUNT and
         (row["count"] >= row["expected"] * RATIO_THRESHOLDS[0] or
          row["robust_z"] >= ROBUST_Z_THRESHOLDS[0])),
        key=lambda row: (-row["robust_z"], row["date"]),
    )[:20]
    return {
        "status": "EXPLORATORY_NO_GROUND_TRUTH",
        "evaluated_days": len(scored),
        "rules": rules,
        "review_queue": [{
            "date": row["date"],
            "daily_count": row["count"],
            "weekday_median": row["expected"],
            "ratio": round(row["ratio"], 4) if row["ratio"] is not None else None,
            "robust_z": round(row["robust_z"], 4),
            "review_status": "PENDING",
        } for row in review],
    }


def build_report(path: Path) -> dict:
    series, first_day, last_day, record_count, invalid_rows, digest = daily_counts(path)
    evaluation = evaluate_series(series, first_day)
    return {
        "report_version": "spike-exploration.v1",
        "source_sha256": digest,
        "scope": "total_daily_appeals_in_one_export",
        "unit": "one_region_total_per_day",
        "first_date": first_day.isoformat(),
        "last_date": last_day.isoformat(),
        "record_count": record_count,
        "invalid_csv_or_date_row_count": invalid_rows,
        "calendar_days": len(series),
        "days_without_records_treated_as_zero": series.count(0),
        "history_weeks_same_weekday": HISTORY_WEEKS,
        "robust_scale": "max(1,sqrt(weekday_median),1.4826*MAD)",
        "selection_status": "NO_REVIEWED_INCIDENT_LABELS",
        "precision": None,
        "recall": None,
        "time_to_detect": None,
        "runtime_eligible": False,
        **evaluation,
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
    except (OSError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "spike evaluation failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "source_sha256": report["source_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
