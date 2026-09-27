#!/usr/bin/env python3
"""Export every exploratory daily spike candidate for human review."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_forecast_csv import daily_counts
from scripts.evaluate_spike_csv import evaluate_series, score_series


def review_candidates(series: list[int | None], first_day: date) -> dict:
    evaluation = evaluate_series(series, first_day)
    scored = score_series(series, first_day)
    candidates: dict[int, dict] = {}
    for rule in evaluation["rules"]:
        previous_emitted = -rule["cooldown_days"]
        for row in scored:
            if row["count"] < rule["minimum_count"]:
                continue
            triggered = (
                row["count"] >= row["expected"] * rule["threshold"]
                if rule["method"] == "ratio"
                else row["robust_z"] >= rule["threshold"]
            )
            if not triggered:
                continue
            emitted = row["index"] - previous_emitted >= rule["cooldown_days"]
            if emitted:
                previous_emitted = row["index"]
            candidate = candidates.setdefault(row["index"], {
                "date": row["date"],
                "daily_count": row["count"],
                "weekday_median": row["expected"],
                "ratio": round(row["ratio"], 4) if row["ratio"] is not None else None,
                "robust_z": round(row["robust_z"], 4),
                "triggered_rules": [],
                "review_status": "PENDING",
            })
            candidate["triggered_rules"].append({
                "method": rule["method"],
                "threshold": rule["threshold"],
                "emitted_after_cooldown": emitted,
            })

    return {
        "status": evaluation["status"],
        "evaluated_days": evaluation["evaluated_days"],
        "excluded_unobserved_or_incomplete_history_days": evaluation["excluded_unobserved_or_incomplete_history_days"],
        "rules": evaluation["rules"],
        "review_item_count": len(candidates),
        "review_items": [candidates[index] for index in sorted(candidates)],
    }


def build_report(path: Path) -> dict:
    series, first_day, last_day, record_count, invalid_rows, digest = daily_counts(path)
    return {
        "report_version": "spike-review-candidates.v1",
        "source_sha256": digest,
        "scope": "total_daily_appeals_in_one_export",
        "unit": "one_region_total_per_day",
        "first_date": first_day.isoformat(),
        "last_date": last_day.isoformat(),
        "record_count": record_count,
        "invalid_csv_or_date_row_count": invalid_rows,
        "calendar_days": len(series),
        "unobserved_calendar_days": series.count(None),
        "missing_day_policy": "exclude_unobserved_days_and_same_weekday_incomplete_history",
        "candidate_policy": "union_of_all_raw_alert_days_across_six_exploratory_rules",
        "selection_status": "NO_REVIEWED_INCIDENT_LABELS",
        "incident_ground_truth_requirements": {
            "status": "MISSING",
            "independent_of_detector": True,
            "coverage": "all_evaluated_days_in_same_scope_including_non_alert_days",
            "required_incident_fields": ["incident_id", "onset_date", "end_date", "review_status"],
            "complete_coverage_assertion_required": True,
        },
        "precision": None,
        "recall": None,
        "false_alerts_per_period": None,
        "time_to_detect": None,
        "runtime_eligible": False,
        **review_candidates(series, first_day),
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
        print(json.dumps({"error": type(error).__name__, "message": "spike review export failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "review_item_count": report["review_item_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
