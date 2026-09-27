#!/usr/bin/env python3
"""Evaluate exploratory spike rules against a complete independent incident registry."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_spike_review import build_report


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _fields(value: object, expected: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _day(value: object, name: str) -> date:
    if type(value) is not str:
        raise ValueError(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date")
    return parsed


def evaluate_incident_registry(candidate: dict, registry: dict, candidate_report_sha256: str) -> dict:
    """Compute daily emitted-alert metrics only after complete registry validation."""
    if type(candidate) is not dict:
        raise ValueError("candidate report must be an object")
    if candidate.get("report_version") != "spike-review-candidates.v2":
        raise ValueError("candidate report version is unsupported")
    evaluated_dates = candidate.get("evaluated_dates")
    if type(evaluated_dates) is not list or not evaluated_dates:
        raise ValueError("candidate report has no evaluated dates")
    days = [_day(item, "evaluated date") for item in evaluated_dates]
    if days != sorted(set(days)) or candidate.get("evaluated_days") != len(days):
        raise ValueError("candidate evaluated dates are incomplete or duplicated")
    digest = hashlib.sha256("\n".join(evaluated_dates).encode("ascii")).hexdigest()
    if candidate.get("evaluated_date_sha256") != digest:
        raise ValueError("candidate evaluated date digest differs")
    for key in ("source_sha256", "scope", "unit"):
        if type(candidate.get(key)) is not str or not candidate[key]:
            raise ValueError(f"candidate {key} is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate["source_sha256"]):
        raise ValueError("candidate source SHA-256 is invalid")

    registry = _fields(registry, {
        "schema_version", "source_sha256", "scope", "unit", "evaluated_date_sha256", "candidate_report_sha256",
        "coverage_assertion", "daily_labels", "incidents",
    }, "incident registry")
    if registry["schema_version"] != "spike-incident-registry.v1":
        raise ValueError("incident registry version is unsupported")
    for key in ("source_sha256", "scope", "unit", "evaluated_date_sha256"):
        expected = digest if key == "evaluated_date_sha256" else candidate[key]
        if registry[key] != expected:
            raise ValueError(f"incident registry {key} differs from candidate report")
    if type(candidate_report_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", candidate_report_sha256) or registry["candidate_report_sha256"] != candidate_report_sha256:
        raise ValueError("incident registry candidate_report_sha256 differs from candidate file")

    assertion = _fields(registry["coverage_assertion"], {
        "all_evaluated_dates_reviewed", "independent_of_detector", "reviewer_id", "reviewed_at",
    }, "coverage assertion")
    if assertion["all_evaluated_dates_reviewed"] is not True or assertion["independent_of_detector"] is not True:
        raise ValueError("complete independent coverage has not been asserted")
    if type(assertion["reviewer_id"]) is not str or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", assertion["reviewer_id"]):
        raise ValueError("coverage assertion needs a reviewer ID")
    if type(assertion["reviewed_at"]) is not str:
        raise ValueError("coverage assertion needs a timestamp")
    try:
        reviewed_at = datetime.fromisoformat(assertion["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("coverage assertion needs an ISO timestamp") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("coverage assertion timestamp needs a timezone")

    incidents = registry["incidents"]
    if type(incidents) is not list:
        raise ValueError("incidents must be a list")
    incident_intervals: dict[str, tuple[date, date]] = {}
    evaluated_set = set(days)
    for incident in incidents:
        incident = _fields(incident, {"incident_id", "onset_date", "end_date", "review_status"}, "incident")
        incident_id = incident["incident_id"]
        if type(incident_id) is not str or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", incident_id):
            raise ValueError("incident ID must be an opaque identifier")
        if incident_id in incident_intervals or incident["review_status"] != "CONFIRMED":
            raise ValueError("incident IDs must be unique and confirmed")
        onset = _day(incident["onset_date"], "incident onset")
        end = _day(incident["end_date"], "incident end")
        if onset > end or any(onset + timedelta(days=offset) not in evaluated_set for offset in range((end - onset).days + 1)):
            raise ValueError("incident interval must be fully within evaluated dates")
        incident_intervals[incident_id] = (onset, end)
    intervals_by_onset = sorted(incident_intervals.values())
    if any(current[0] <= previous[1] for previous, current in zip(intervals_by_onset, intervals_by_onset[1:])):
        raise ValueError("incident intervals must not overlap in one-region daily scope")

    labels = registry["daily_labels"]
    if type(labels) is not list or len(labels) != len(days):
        raise ValueError("daily labels must cover every evaluated date")
    label_ids: dict[date, set[str]] = {}
    for label in labels:
        label = _fields(label, {"date", "incident_ids", "review_status"}, "daily label")
        day = _day(label["date"], "label date")
        ids = label["incident_ids"]
        if day not in evaluated_set or day in label_ids or label["review_status"] != "CONFIRMED":
            raise ValueError("daily labels have duplicate, extra, or unconfirmed dates")
        if type(ids) is not list or any(type(item) is not str for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("daily incident IDs must be a unique list")
        if not set(ids) <= incident_intervals.keys():
            raise ValueError("daily label refers to an unknown incident")
        label_ids[day] = set(ids)
    if set(label_ids) != evaluated_set:
        raise ValueError("daily labels do not cover every evaluated date")
    for day, actual in label_ids.items():
        expected = {incident_id for incident_id, (onset, end) in incident_intervals.items() if onset <= day <= end}
        if actual != expected:
            raise ValueError("daily labels contradict incident intervals")

    rules = candidate.get("rules")
    items = candidate.get("review_items")
    if type(rules) is not list or not rules or type(items) is not list or candidate.get("review_item_count") != len(items):
        raise ValueError("candidate report has incomplete rule or review items")
    rule_keys = set()
    for rule in rules:
        if type(rule) is not dict or rule.get("method") not in ("ratio", "weekday_robust_z"):
            raise ValueError("candidate rule is invalid")
        threshold = rule.get("threshold")
        if type(threshold) not in (int, float) or threshold <= 0:
            raise ValueError("candidate rule threshold is invalid")
        key = (rule["method"], threshold)
        if key in rule_keys or type(rule.get("raw_alert_count")) is not int or type(rule.get("cooldown_alert_count")) is not int:
            raise ValueError("candidate rules are duplicated or incomplete")
        rule_keys.add(key)
    alerts: dict[tuple[str, float], list[date]] = {key: [] for key in rule_keys}
    raw_counts = {key: 0 for key in rule_keys}
    seen_items: set[date] = set()
    for item in items:
        if type(item) is not dict or type(item.get("triggered_rules")) is not list:
            raise ValueError("candidate review item is invalid")
        day = _day(item.get("date"), "candidate date")
        if day not in evaluated_set or day in seen_items or not item["triggered_rules"]:
            raise ValueError("candidate dates are duplicated, extra, or untriggered")
        seen_items.add(day)
        seen_triggers = set()
        for trigger in item["triggered_rules"]:
            if type(trigger) is not dict:
                raise ValueError("candidate trigger is invalid")
            key = (trigger.get("method"), trigger.get("threshold"))
            if key not in rule_keys or key in seen_triggers or type(trigger.get("emitted_after_cooldown")) is not bool:
                raise ValueError("candidate trigger is duplicated or invalid")
            seen_triggers.add(key)
            raw_counts[key] += 1
            if trigger["emitted_after_cooldown"]:
                alerts[key].append(day)

    per_rule = []
    for rule in rules:
        key = (rule["method"], rule["threshold"])
        emitted = sorted(alerts[key])
        if raw_counts[key] != rule["raw_alert_count"] or len(emitted) != rule["cooldown_alert_count"]:
            raise ValueError("candidate alert counts do not match rule counts")
        true_alerts = [day for day in emitted if label_ids[day]]
        false_alerts = [day for day in emitted if not label_ids[day]]
        detection_delays = []
        for onset, end in incident_intervals.values():
            detected = next((day for day in emitted if onset <= day <= end), None)
            if detected is not None:
                detection_delays.append((detected - onset).days)
        per_rule.append({
            "method": rule["method"],
            "threshold": rule["threshold"],
            "minimum_count": rule["minimum_count"],
            "cooldown_days": rule["cooldown_days"],
            "emitted_alert_count": len(emitted),
            "true_alert_count": len(true_alerts),
            "false_alert_count": len(false_alerts),
            "false_alert_dates_for_review": [day.isoformat() for day in false_alerts],
            "precision": round(len(true_alerts) / len(emitted), 4) if emitted else None,
            "incident_count": len(incident_intervals),
            "detected_incident_count": len(detection_delays),
            "recall": round(len(detection_delays) / len(incident_intervals), 4) if incident_intervals else None,
            "false_alerts_per_30_evaluated_days": round(len(false_alerts) * 30 / len(days), 4),
            "time_to_detect_days": detection_delays,
            "mean_time_to_detect_days_for_detected_incidents": (
                round(sum(detection_delays) / len(detection_delays), 4) if detection_delays else None
            ),
        })

    return {
        "report_version": "spike-incident-evaluation.v1",
        "source_sha256": candidate["source_sha256"],
        "candidate_report_sha256": candidate_report_sha256,
        "scope": candidate["scope"],
        "unit": candidate["unit"],
        "evaluated_date_sha256": digest,
        "evaluated_days": len(days),
        "registry_reviewed_at": assertion["reviewed_at"],
        "registry_reviewer_id": assertion["reviewer_id"],
        "coverage_status": "COMPLETE_ASSERTED_AND_DATE_VERIFIED",
        "metric_policy": "daily_emitted_alerts_after_cooldown; incident_recall_by_confirmed_interval",
        "selection_status": "NO_RUNTIME_THRESHOLD_SELECTED",
        "runtime_eligible": False,
        "rules": per_rule,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("incident_registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate_bytes = args.candidate_report.read_bytes()
        candidate = json.loads(candidate_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
        registry = json.loads(args.incident_registry.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if candidate != build_report(args.source_csv):
            raise ValueError("candidate report differs from regenerated source evaluation")
        report = evaluate_incident_registry(candidate, registry, hashlib.sha256(candidate_bytes).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "spike incident evaluation failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["coverage_status"], "evaluated_days": report["evaluated_days"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
