"""Calibrate a provisional duplicate threshold on reviewed retrieval pairs."""

from __future__ import annotations

import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

from training.classifier_baselines import load_verified_classifier_package
from training.feedback_dataset import ID_RE
from training.retrieval_baselines import _e5_scores, _load_retrieval


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON policy key")
        result[key] = value
    return result


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    required = {"policy_version", "min_duplicate_count", "min_nonduplicate_count",
                "min_predictions", "min_precision"}
    if not isinstance(policy, dict) or set(policy) != required or policy["policy_version"] != "duplicate-threshold.v1":
        raise ValueError("invalid duplicate threshold policy")
    for key in ("min_duplicate_count", "min_nonduplicate_count", "min_predictions"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError("duplicate threshold policy counts must be positive integers")
    if type(policy["min_precision"]) not in (int, float) or not 0 < policy["min_precision"] <= 1:
        raise ValueError("duplicate threshold minimum precision must be in (0, 1]")
    return policy


def _counts(rows: list[dict], scores: list[float], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    false_positive_pairs = []
    for row, score in zip(rows, scores):
        predicted = score >= threshold
        actual = row["relation_label"] == "DUPLICATE"
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
            false_positive_pairs.append({
                "pair_sha256": "sha256:" + hashlib.sha256(row["pair_id"].encode("utf-8")).hexdigest(),
                "relation_label": row["relation_label"],
            })
        elif actual:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "repeat_false_positives": sum(item["relation_label"] == "REPEAT" for item in false_positive_pairs),
        "false_positive_pairs": false_positive_pairs,
    }


def calibrate_duplicate_threshold(rows: list[dict], scores: list[float], policy: dict) -> dict:
    if len(rows) != len(scores) or not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate evaluation pairs and scores do not match")
    if any(row.get("relation_label") not in {"DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "REPEAT", "UNRELATED"}
           for row in rows):
        raise ValueError("duplicate evaluation has an unsupported relation label")
    if any(not math.isfinite(score) or not -1 <= score <= 1 for score in scores):
        raise ValueError("duplicate similarity scores must be finite cosine values")
    positives = sum(row["relation_label"] == "DUPLICATE" for row in rows)
    negatives = len(rows) - positives
    if positives < policy["min_duplicate_count"] or negatives < policy["min_nonduplicate_count"]:
        return {"status": "INSUFFICIENT_EVIDENCE", "duplicate_count": positives,
                "nonduplicate_count": negatives, "selected_threshold": None, "curve": []}
    curve = []
    tp = fp = 0
    ranked = sorted(zip(scores, rows), key=lambda item: -item[0])
    for threshold, tied in groupby(ranked, key=lambda item: item[0]):
        for _, row in tied:
            if row["relation_label"] == "DUPLICATE":
                tp += 1
            else:
                fp += 1
        curve.append({"threshold": threshold, "tp": tp, "fp": fp,
                      "fn": positives - tp, "tn": negatives - fp,
                      "precision": tp / (tp + fp), "recall": tp / positives})
    eligible = [point for point in curve if point["tp"] + point["fp"] >= policy["min_predictions"]
                and point["precision"] is not None and point["precision"] >= policy["min_precision"]]
    selected = max(eligible, key=lambda point: (point["recall"], point["precision"], point["threshold"])) if eligible else None
    return {"status": "PENDING_HUMAN_REVIEW" if selected else "INSUFFICIENT_EVIDENCE",
            "duplicate_count": positives, "nonduplicate_count": negatives,
            "selected_threshold": selected["threshold"] if selected else None, "curve": curve}


def evaluate_duplicate_thresholds(package: Path, model: Path, policy_path: Path,
                                  model_version: str, batch_size: int = 16) -> dict:
    if not ID_RE.fullmatch(model_version):
        raise ValueError("model version is required")
    policy = load_policy(policy_path)
    manifest, _ = load_verified_classifier_package(package)
    splits = _load_retrieval(package, manifest.membership_sha256)
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if (sorted(row["pair_id"] for row in splits["test"]) != frozen["retrieval_pair_ids"] or
            sorted({row["relation_group"] for row in splits["test"]}) != frozen["retrieval_groups"]):
        raise ValueError("frozen retrieval membership mismatch")
    rows = splits["validation"] + splits["test"]
    model_manifest = model / "manifest.json"
    if model_manifest.is_file():
        recorded_version = json.loads(model_manifest.read_text(encoding="utf-8")).get("model_version")
        if recorded_version != model_version:
            raise ValueError("model version does not match local artifact manifest")
    scores, model_checksum = _e5_scores(rows, model, batch_size)
    if len(scores) != len(rows) or any(not math.isfinite(score) or not -1.000001 <= score <= 1.000001 for score in scores):
        raise ValueError("model returned invalid cosine similarity scores")
    scores = [max(-1.0, min(1.0, score)) for score in scores]
    validation_count = len(splits["validation"])
    validation = calibrate_duplicate_threshold(splits["validation"], scores[:validation_count], policy)
    threshold = validation["selected_threshold"]
    validation["selected_confusion"] = _counts(splits["validation"], scores[:validation_count], threshold) if threshold is not None else None
    test = None
    status = validation["status"]
    threshold_status = "NOT_SELECTED"
    if threshold is not None:
        test = _counts(splits["test"], scores[validation_count:], threshold)
        test["sample_count"] = len(splits["test"])
        test["duplicate_count"] = test["tp"] + test["fn"]
        test["nonduplicate_count"] = test["fp"] + test["tn"]
        if (test["duplicate_count"] < policy["min_duplicate_count"] or
                test["nonduplicate_count"] < policy["min_nonduplicate_count"] or
                test["tp"] + test["fp"] < policy["min_predictions"]):
            status = "INSUFFICIENT_TEST_EVIDENCE"
            threshold_status = "NOT_APPROVED"
        elif test["precision"] is None or test["precision"] < policy["min_precision"]:
            status = "TEST_PRECISION_BELOW_POLICY"
            threshold_status = "REJECTED"
        else:
            threshold_status = "PROVISIONAL"
    return {
        "report_version": "duplicate-threshold-evaluation.v1",
        "status": status,
        "threshold_status": threshold_status,
        "model_version": model_version,
        "model_identity_status": "VERSION_MATCHED" if model_manifest.is_file() else "EXPLICIT_UNVERIFIED",
        "model_artifact_sha256": model_checksum,
        "dataset_version": manifest.dataset_version,
        "dataset_content_sha256": manifest.content_sha256,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "frozen_evaluation_sha256": manifest.frozen_evaluation_sha256,
        "synthetic": manifest.synthetic,
        "policy": policy,
        "policy_sha256": "sha256:" + hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "validation": validation,
        "test": test,
        "repeat_calibration_status": "NOT_EVALUATED",
    }
