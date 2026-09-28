"""Evaluate repeat windows using separately reviewed temporal evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE
from training.retrieval_baselines import _load_retrieval


EVIDENCE_FIELDS = {
    "evidence_version", "pair_id", "source_relation_sha256", "review_evidence_sha256",
    "temporal_source_sha256",
    "review_status", "reviewer_id", "reviewed_at", "query_created_at",
    "candidate_created_at", "candidate_resolved_at", "same_region", "same_object",
    "same_issue", "same_episode", "prior_episode_resolved",
}
FACTS = ("same_region", "same_object", "same_issue", "same_episode", "prior_episode_resolved")
TIME_FIELDS = ("query_created_at", "candidate_created_at", "candidate_resolved_at")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate temporal evidence or policy key")
        result[key] = value
    return result


def _timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} requires a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} requires a timezone-aware timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} requires a timezone-aware timestamp")
    return parsed


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    fields = {"policy_version", "window_days", "min_repeat_count", "min_nonrepeat_count",
              "min_predictions", "min_precision"}
    if not isinstance(policy, dict) or set(policy) != fields or policy["policy_version"] != "repeat-window.v1":
        raise ValueError("invalid repeat window policy")
    windows = policy["window_days"]
    if (not isinstance(windows, list) or not windows or
            any(type(value) is not int or value < 1 for value in windows) or
            windows != sorted(set(windows))):
        raise ValueError("repeat windows must be sorted unique positive days")
    if any(type(policy[key]) is not int or policy[key] < 1 for key in
           ("min_repeat_count", "min_nonrepeat_count", "min_predictions")):
        raise ValueError("repeat policy counts must be positive integers")
    precision = policy["min_precision"]
    if type(precision) not in (int, float) or not math.isfinite(precision) or not 0 < precision <= 1:
        raise ValueError("repeat minimum precision must be in (0, 1]")
    return policy


def _load_source(path: Path, pairs: dict[str, dict]) -> dict[str, dict]:
    source = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"temporal source line {line_number}: invalid JSON") from error
        pair_id = row.get("pair_id") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"pair_id", *TIME_FIELDS} or
                not isinstance(pair_id, str) or pair_id not in pairs or pair_id in source):
            raise ValueError(f"temporal source line {line_number}: invalid or duplicate pair")
        source[pair_id] = row
    if set(source) != set(pairs):
        raise ValueError("temporal source must cover every validation and test pair")
    return source


def _load_evidence(path: Path, source_path: Path, pairs: dict[str, dict]) -> dict[str, dict]:
    source = _load_source(source_path, pairs)
    source_sha = checksum(source_path)
    evidence = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise ValueError(f"temporal evidence line {line_number}: invalid JSON") from error
        pair_id = row.get("pair_id") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != EVIDENCE_FIELDS or
                not isinstance(pair_id, str) or pair_id not in pairs or pair_id in evidence or
                row["evidence_version"] != "repeat-temporal-evidence.v1" or
                row["review_status"] != "APPROVED" or
                not isinstance(row["reviewer_id"], str) or not ID_RE.fullmatch(row["reviewer_id"]) or
                row["source_relation_sha256"] != pairs[pair_id]["source_relation_sha256"] or
                row["review_evidence_sha256"] != pairs[pair_id]["review_evidence_sha256"] or
                row["temporal_source_sha256"] != source_sha or
                any(row[key] != source[pair_id][key] for key in TIME_FIELDS) or
                any(type(row[key]) is not bool for key in FACTS)):
            raise ValueError(f"temporal evidence line {line_number}: unapproved or mismatched pair")
        reviewed_time = _timestamp(row["reviewed_at"], "reviewed_at")
        query_time = _timestamp(row["query_created_at"], "query_created_at")
        candidate_time = _timestamp(row["candidate_created_at"], "candidate_created_at")
        resolved_value = row["candidate_resolved_at"]
        resolved_time = _timestamp(resolved_value, "candidate_resolved_at") if resolved_value is not None else None
        resolved_before_query = resolved_time is not None and resolved_time < query_time
        if (reviewed_time < query_time or candidate_time >= query_time or
                (resolved_time is not None and (resolved_time <= candidate_time or reviewed_time < resolved_time)) or
                row["prior_episode_resolved"] != resolved_before_query):
            raise ValueError(f"temporal evidence line {line_number}: invalid resolution chronology")
        label = pairs[pair_id]["relation_label"]
        same_core = all(row[key] for key in ("same_region", "same_object", "same_issue"))
        if (label == "REPEAT" and not (same_core and not row["same_episode"] and row["prior_episode_resolved"]) or
                label == "DUPLICATE" and not (same_core and row["same_episode"] and not row["prior_episode_resolved"]) or
                label in {"SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"} and same_core and
                (row["same_episode"] or row["prior_episode_resolved"])):
            raise ValueError(f"temporal evidence line {line_number}: relation semantics mismatch")
        evidence[pair_id] = {**row, "query_time": query_time, "resolved_time": resolved_time}
    if set(evidence) != set(pairs):
        raise ValueError("temporal evidence must cover every validation and test pair")
    return evidence


def _counts(rows: list[dict], evidence: dict[str, dict], window_days: int) -> dict:
    tp = fp = fn = tn = 0
    false_positive_pairs = []
    for row in rows:
        facts = evidence[row["pair_id"]]
        resolved = facts["resolved_time"]
        predicted = (facts["same_region"] and resolved is not None and
                     0 <= (facts["query_time"] - resolved).total_seconds() <= window_days * 86400)
        actual = row["relation_label"] == "REPEAT"
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
            false_positive_pairs.append("sha256:" + hashlib.sha256(row["pair_id"].encode()).hexdigest())
        elif actual:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_pair_hashes": false_positive_pairs}


def _hard_negative_count(rows: list[dict], evidence: dict[str, dict]) -> int:
    return sum(row["relation_label"] != "REPEAT" and evidence[row["pair_id"]]["same_region"] and
               evidence[row["pair_id"]]["prior_episode_resolved"] for row in rows)


def evaluate_repeat_windows(package: Path, source_path: Path, evidence_path: Path, policy_path: Path) -> dict:
    policy = load_policy(policy_path)
    manifest, _ = load_verified_classifier_package(package)
    splits = _load_retrieval(package, manifest.membership_sha256)
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if (sorted(row["pair_id"] for row in splits["test"]) != frozen["retrieval_pair_ids"] or
            sorted({row["relation_group"] for row in splits["test"]}) != frozen["retrieval_groups"]):
        raise ValueError("frozen retrieval membership mismatch")
    pairs = {row["pair_id"]: row for split in ("validation", "test") for row in splits[split]}
    evidence = _load_evidence(evidence_path, source_path, pairs)
    validation_rows = splits["validation"]
    repeat_count = sum(row["relation_label"] == "REPEAT" for row in validation_rows)
    nonrepeat_count = len(validation_rows) - repeat_count
    hard_negative_count = _hard_negative_count(validation_rows, evidence)
    curve = []
    selected = None
    if repeat_count >= policy["min_repeat_count"] and hard_negative_count >= policy["min_nonrepeat_count"]:
        for days in policy["window_days"]:
            point = {"window_days": days, **_counts(validation_rows, evidence, days)}
            curve.append(point)
        eligible = [point for point in curve if point["tp"] + point["fp"] >= policy["min_predictions"]
                    and point["precision"] is not None and point["precision"] >= policy["min_precision"]]
        selected = max(eligible, key=lambda point: (point["recall"], point["precision"], -point["window_days"])) if eligible else None
    test = _counts(splits["test"], evidence, selected["window_days"]) if selected else None
    status = "INSUFFICIENT_EVIDENCE"
    if test is not None:
        if (test["tp"] + test["fn"] < policy["min_repeat_count"] or
                _hard_negative_count(splits["test"], evidence) < policy["min_nonrepeat_count"] or
                test["tp"] + test["fp"] < policy["min_predictions"]):
            status = "INSUFFICIENT_TEST_EVIDENCE"
        elif test["precision"] is None or test["precision"] < policy["min_precision"]:
            status = "TEST_PRECISION_BELOW_POLICY"
        else:
            status = "PENDING_HUMAN_REVIEW"
    return {
        "report_version": "repeat-window-evaluation.v1", "status": status,
        "evidence_scope": "HUMAN_REVIEWED_RESOLUTION_ORACLE_ONLY",
        "runtime_window_status": "NOT_APPROVED",
        "dataset_version": manifest.dataset_version,
        "dataset_content_sha256": manifest.content_sha256,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "frozen_evaluation_sha256": manifest.frozen_evaluation_sha256,
        "synthetic": manifest.synthetic,
        "temporal_source_sha256": checksum(source_path),
        "temporal_evidence_sha256": checksum(evidence_path),
        "policy": policy, "policy_sha256": checksum(policy_path),
        "validation": {"repeat_count": repeat_count, "nonrepeat_count": nonrepeat_count,
                       "hard_negative_count": hard_negative_count,
                       "selected_window_days": selected["window_days"] if selected else None,
                       "curve": curve},
        "test": {**test, "hard_negative_count": _hard_negative_count(splits["test"], evidence)} if test else None,
    }
