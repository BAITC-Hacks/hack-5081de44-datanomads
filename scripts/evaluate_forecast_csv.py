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


HORIZONS = (30, 60, 90)
SEASON_LENGTH = 7
MIN_TRAIN_DAYS = 365
ORIGIN_STEP_DAYS = 30


def daily_counts(path: Path) -> tuple[list[int], date, date, int, int, str]:
    counts: Counter[date] = Counter()
    invalid_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
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

    if not counts:
        raise ValueError("CSV has no valid creation_date values")
    first_day, last_day = min(counts), max(counts)
    series = [counts[first_day + timedelta(days=offset)] for offset in range((last_day - first_day).days + 1)]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return series, first_day, last_day, sum(counts.values()), invalid_rows, digest.hexdigest()


def evaluate_horizon(series: list[int], horizon: int) -> dict[str, int | float]:
    absolute_error = squared_error = actual_total = smape_total = 0.0
    sample_count = window_count = smape_count = 0
    for origin in range(MIN_TRAIN_DAYS, len(series) - horizon + 1, ORIGIN_STEP_DAYS):
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
        raise ValueError(f"at least {MIN_TRAIN_DAYS + horizon} calendar days required for horizon {horizon}")
    return {
        "horizon_days": horizon,
        "window_count": window_count,
        "sample_count": sample_count,
        "mae": round(absolute_error / sample_count, 4),
        "rmse": round(math.sqrt(squared_error / sample_count), 4),
        "wape": round(absolute_error / actual_total, 4) if actual_total else 0.0,
        "smape": round(smape_total / smape_count, 4) if smape_count else 0.0,
    }


def build_report(path: Path) -> dict[str, object]:
    series, first_day, last_day, record_count, invalid_rows, digest = daily_counts(path)
    return {
        "source_sha256": digest,
        "method": "weekly_seasonal_naive",
        "scope": "total_daily_appeals_in_one_export",
        "first_date": first_day.isoformat(),
        "last_date": last_day.isoformat(),
        "record_count": record_count,
        "invalid_csv_or_date_row_count": invalid_rows,
        "calendar_days": len(series),
        "days_without_records": series.count(0),
        "minimum_train_days": MIN_TRAIN_DAYS,
        "origin_step_days": ORIGIN_STEP_DAYS,
        "season_length_days": SEASON_LENGTH,
        "results": [evaluate_horizon(series, horizon) for horizon in HORIZONS],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.input)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
