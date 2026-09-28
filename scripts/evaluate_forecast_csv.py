#!/usr/bin/env python3
"""Evaluate a weekly seasonal baseline on daily counts from a 109 CSV export."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys


HORIZONS = (30, 60, 90)
SEASON_LENGTH = 7
MIN_TRAIN_DAYS = 365
ORIGIN_STEP_DAYS = 30
MISSING_DAY_POLICY = "fixed_calendar_origins_require_365_consecutive_observed_train_days_and_observed_targets"


def daily_counts(path: Path) -> tuple[list[int | None], date, date, int, int, str]:
    counts: Counter[date] = Counter()
    invalid_rows = 0
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("CSV is empty") from error
            if "creation_date" not in header:
                raise ValueError("CSV is missing creation_date")
            date_index = header.index("creation_date")
            for row in reader:
                if len(row) != len(header):
                    invalid_rows += 1
                    continue
                try:
                    day = datetime.strptime(row[date_index].strip(), "%d.%m.%Y %H:%M:%S").date()
                except ValueError:
                    invalid_rows += 1
                    continue
                counts[day] += 1
    except csv.Error as error:
        raise ValueError("CSV is malformed") from error

    if not counts:
        raise ValueError("CSV has no valid creation_date values")
    first_day, last_day = min(counts), max(counts)
    series = [counts.get(first_day + timedelta(days=offset)) for offset in range((last_day - first_day).days + 1)]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return series, first_day, last_day, sum(counts.values()), invalid_rows, digest.hexdigest()


def observed_segments(series: list[int | None]) -> list[tuple[int, list[int]]]:
    segments = []
    observed = []
    start = 0
    for index, value in enumerate(series):
        if value is None:
            if observed:
                segments.append((start, observed))
                observed = []
            start = index + 1
        else:
            if not observed:
                start = index
            observed.append(value)
    if observed:
        segments.append((start, observed))
    return segments


def eligible_origins(series: list[int | None], horizon: int) -> list[tuple[int, int]]:
    run_start = 0
    run_starts = [0]
    for index, value in enumerate(series):
        if value is None:
            run_start = index + 1
        run_starts.append(run_start)
    return [
        (origin, run_starts[origin])
        for origin in range(MIN_TRAIN_DAYS, len(series) - horizon + 1, ORIGIN_STEP_DAYS)
        if origin - run_starts[origin] >= MIN_TRAIN_DAYS
        and None not in series[origin : origin + horizon]
    ]


def evaluate_horizon(series: list[int | None], horizon: int) -> dict[str, int | float | str | None]:
    absolute_error = squared_error = actual_total = smape_total = 0.0
    sample_count = window_count = smape_count = 0
    for origin, _ in eligible_origins(series, horizon):
        weekly_pattern = series[origin - SEASON_LENGTH : origin]
        for offset, actual in enumerate(series[origin : origin + horizon]):
            predicted = weekly_pattern[offset % SEASON_LENGTH]
            error = abs(actual - predicted)
            absolute_error += error
            squared_error += error * error
            actual_total += actual
            if actual + predicted:
                smape_total += 2 * error / (actual + predicted)
                smape_count += 1
        sample_count += horizon
        window_count += 1
    if not window_count:
        raise ValueError(f"no eligible fully observed rolling window for horizon {horizon}")
    return {
        "horizon_days": horizon,
        "status": "EVALUATED",
        "window_count": window_count,
        "sample_count": sample_count,
        "mae": round(absolute_error / sample_count, 4),
        "rmse": round(math.sqrt(squared_error / sample_count), 4),
        "wape": round(absolute_error / actual_total, 4) if actual_total else None,
        "smape": round(smape_total / smape_count, 4) if smape_count else 0.0,
    }


def build_report(path: Path) -> dict[str, object]:
    series, first_day, last_day, record_count, invalid_rows, digest = daily_counts(path)
    longest_run = max(len(observed) for _, observed in observed_segments(series))
    results = [
        evaluate_horizon(series, horizon) if eligible_origins(series, horizon) else {
            "horizon_days": horizon,
            "status": "INSUFFICIENT_HISTORY",
            "reason": "CALENDAR_TOO_SHORT" if len(series) < MIN_TRAIN_DAYS + horizon else "NO_FULLY_OBSERVED_WINDOW",
            "window_count": 0,
            "sample_count": 0,
            "required_calendar_days": MIN_TRAIN_DAYS + horizon,
            "available_calendar_days": len(series),
            "longest_observed_run_days": longest_run,
        }
        for horizon in HORIZONS
    ]
    evaluated_count = sum(result["status"] == "EVALUATED" for result in results)
    status = "EVALUATED" if evaluated_count == len(HORIZONS) else "PARTIAL" if evaluated_count else results[0]["status"]
    return {
        "report_version": "forecast-baseline.v2",
        "source_sha256": digest,
        "status": status,
        "method": "weekly_seasonal_naive",
        "scope": "total_daily_appeals_in_one_export",
        "first_date": first_day.isoformat(),
        "last_date": last_day.isoformat(),
        "record_count": record_count,
        "invalid_csv_or_date_row_count": invalid_rows,
        "calendar_days": len(series),
        "unobserved_calendar_days": series.count(None),
        "missing_day_policy": MISSING_DAY_POLICY,
        "longest_observed_run_days": longest_run,
        "minimum_train_days": MIN_TRAIN_DAYS,
        "origin_step_days": ORIGIN_STEP_DAYS,
        "season_length_days": SEASON_LENGTH,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.input)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        else:
            print(serialized, end="")
    except (OSError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "forecast baseline evaluation failed"}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
