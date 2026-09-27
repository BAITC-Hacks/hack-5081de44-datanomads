"""Check that several classifier challengers share frozen and fresh evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from data.normalization.pii import scan_pii
from training.contracts import _checksum
from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE, TOPICS, _unique_object


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("challenger input report must be an object")
    return value


def _number(value: object, minimum: float, maximum: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def _topics(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(topic, str) and topic in TOPICS for topic in value)


def _offline_reference(report: dict) -> tuple:
    production = report["production"]
    return (
        report["dataset_version"], report["dataset_content_sha256"],
        report["frozen_evaluation_version"], report["frozen_evaluation_sha256"],
        report["sample_count"], report["sample_ids_sha256"],
        tuple(report["labels"]), report["synthetic"], report["policy_sha256"],
        json.dumps(report["policy"], sort_keys=True),
        production["model_version"], production["artifact_checksum"],
        json.dumps(production["metrics"], sort_keys=True),
    )


def _fresh_reference(report: dict) -> tuple:
    return (
        report["production_model_version"], report["promotion_policy_version"],
        report["window_start"], report["window_end"], report["sample_count"],
        report["sample_ids_sha256"], report["champion_reference_sha256"],
        report["policy_sha256"], json.dumps(report["policy"], sort_keys=True),
        json.dumps(report["origin_counts"], sort_keys=True),
        report["production_agreement"], report["production_correction_rate"],
        report["real_production_agreement"],
    )


def compare_challengers(inputs: list[tuple[Path, Path]]) -> dict:
    if len(inputs) < 2:
        raise ValueError("at least two challengers are required")
    frozen_reference = fresh_reference = None
    candidates = []
    seen_versions = set()
    for offline_path, shadow_path in inputs:
        offline = _read(offline_path)
        shadow = _read(shadow_path)
        try:
            candidate = offline["candidate"]
            version = candidate["model_version"]
            public_ids = (version, offline["production"]["model_version"],
                          offline["frozen_evaluation_version"])
            window_start = datetime.fromisoformat(shadow["window_start"])
            window_end = datetime.fromisoformat(shadow["window_end"])
            if (offline["report_version"] != "classifier-pair-evaluation.v1" or
                    shadow["report_version"] != "classifier-shadow-evaluation.v1" or
                    shadow["gate_population"] != "real_only.v1" or
                    any(not isinstance(value, str) or not ID_RE.fullmatch(value) or
                        scan_pii(value).detected for value in public_ids) or
                    version in seen_versions or
                    version == offline["production"]["model_version"] or
                    shadow["candidate_model_version"] != version or
                    shadow["production_model_version"] != offline["production"]["model_version"] or
                    window_start.tzinfo is None or window_end.tzinfo is None or
                    window_start >= window_end or
                    offline["policy"]["policy_version"] != "classifier-critical-regression.v1" or
                    shadow["policy"]["policy_version"] != "classifier-shadow-policy.v1" or
                    shadow["window_start"] != shadow["policy"]["window_start"] or
                    shadow["window_end"] != shadow["policy"]["window_end"] or
                    shadow["promotion_policy_version"] != shadow["policy"]["promotion_policy_version"] or
                    type(offline["synthetic"]) is not bool or
                    type(offline["policy"].get("min_total_samples")) is not int or
                    offline["policy"]["min_total_samples"] < 1 or
                    type(shadow["policy"].get("min_samples")) is not int or
                    shadow["policy"]["min_samples"] < 1 or
                    type(shadow["policy"].get("min_real_samples")) is not int or
                    shadow["policy"]["min_real_samples"] < 1 or
                    not _number(shadow["policy"].get("max_correction_rate_increase"), 0, 1) or
                    type(offline["sample_count"]) is not int or offline["sample_count"] < 1 or
                    type(shadow["sample_count"]) is not int or shadow["sample_count"] < 1 or
                    not _topics(offline["labels"]) or
                    not _topics(offline["regressed_critical_topics"]) or
                    not _topics(shadow["critical_regressions"]) or
                    set(shadow["origin_counts"]) - {"real", "synthetic"} or
                    offline["production"]["metrics"]["sample_count"] != offline["sample_count"] or
                    candidate["metrics"]["sample_count"] != offline["sample_count"] or
                    sum(shadow["origin_counts"].values()) != shadow["sample_count"] or
                    any(type(value) is not int or value < 0
                        for value in shadow["origin_counts"].values()) or
                    not _number(offline["production"]["metrics"]["macro_f1"], 0, 1) or
                    not _number(candidate["metrics"]["macro_f1"], 0, 1) or
                    not _number(offline["macro_f1_delta"], -1, 1) or
                    abs(offline["macro_f1_delta"] - round(
                        candidate["metrics"]["macro_f1"] -
                        offline["production"]["metrics"]["macro_f1"], 6)) > 0.000001 or
                    not _number(shadow["production_agreement"], 0, 1) or
                    not _number(shadow["candidate_agreement"], 0, 1) or
                    not _number(shadow["production_correction_rate"], 0, 1) or
                    not _number(shadow["correction_rate_delta"], -1, 1) or
                    abs(shadow["production_correction_rate"] -
                        round(1 - shadow["production_agreement"], 6)) > 0.000001 or
                    abs(shadow["correction_rate_delta"] - round(
                        shadow["production_agreement"] - shadow["candidate_agreement"], 6)) > 0.000001 or
                    (shadow["origin_counts"].get("real", 0) > 0 and (
                        not _number(shadow["real_production_agreement"], 0, 1) or
                        not _number(shadow["real_candidate_agreement"], 0, 1) or
                        not _number(shadow["real_correction_rate_delta"], -1, 1) or
                        abs(shadow["real_correction_rate_delta"] - round(
                            shadow["real_production_agreement"] -
                            shadow["real_candidate_agreement"], 6)) > 0.000001)) or
                    shadow["global_regression"] is not (
                        shadow["real_correction_rate_delta"] is not None and
                        shadow["real_correction_rate_delta"] >
                        shadow["policy"]["max_correction_rate_increase"])):
                raise ValueError("challenger reports have invalid identity or evidence")
            for value in (offline["dataset_content_sha256"], offline["frozen_evaluation_sha256"],
                          offline["sample_ids_sha256"], offline["policy_sha256"],
                          offline["production"]["artifact_checksum"], candidate["artifact_checksum"],
                          shadow["sample_ids_sha256"], shadow["champion_reference_sha256"],
                          shadow["policy_sha256"]):
                _checksum(value)
            frozen = _offline_reference(offline)
            fresh = _fresh_reference(shadow)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("challenger reports have invalid identity or evidence") from error
        if frozen_reference is None:
            frozen_reference, fresh_reference = frozen, fresh
        elif frozen != frozen_reference or fresh != fresh_reference:
            raise ValueError("challengers were not evaluated on identical frozen and fresh evidence")
        seen_versions.add(version)
        candidates.append({
            "model_version": version,
            "artifact_checksum": candidate["artifact_checksum"],
            "offline_synthetic": offline["synthetic"],
            "offline_decision": offline["decision"],
            "offline_macro_f1": candidate["metrics"]["macro_f1"],
            "offline_macro_f1_delta": offline["macro_f1_delta"],
            "fresh_decision": shadow["decision"],
            "fresh_candidate_agreement": shadow["candidate_agreement"],
            "fresh_correction_rate_delta": shadow["correction_rate_delta"],
            "fresh_real_correction_rate_delta": shadow["real_correction_rate_delta"],
            "offline_critical_regressions": offline["regressed_critical_topics"],
            "fresh_critical_regressions": shadow["critical_regressions"],
            "real_fresh_samples": shadow["origin_counts"].get("real", 0),
            "ready_for_human_review": (
                not offline["synthetic"] and shadow["status"] == "VALID" and
                offline["sample_count"] >= offline["policy"]["min_total_samples"] and
                shadow["sample_count"] >= shadow["policy"]["min_samples"] and
                shadow["origin_counts"].get("real", 0) >=
                shadow["policy"]["min_real_samples"] and
                offline["decision"] == "PENDING_HUMAN_REVIEW" and
                shadow["decision"] == "PENDING_HUMAN_REVIEW" and
                not shadow["global_regression"] and
                not offline["regressed_critical_topics"] and
                not shadow["critical_regressions"]
            ),
            "offline_report_sha256": checksum(offline_path),
            "shadow_report_sha256": checksum(shadow_path),
        })
    return {
        "report_version": "classifier-challengers.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPARABLE",
        "champion_model_version": frozen_reference[10],
        "champion_artifact_checksum": frozen_reference[11],
        "frozen_evaluation_version": frozen_reference[2],
        "frozen_evaluation_sha256": frozen_reference[3],
        "frozen_sample_count": frozen_reference[4],
        "frozen_sample_ids_sha256": frozen_reference[5],
        "fresh_window_start": fresh_reference[2],
        "fresh_window_end": fresh_reference[3],
        "fresh_sample_count": fresh_reference[4],
        "fresh_sample_ids_sha256": fresh_reference[5],
        "fresh_champion_reference_sha256": fresh_reference[6],
        "offline_policy_sha256": frozen_reference[8],
        "shadow_policy_sha256": fresh_reference[7],
        "candidates": sorted(candidates, key=lambda item: item["model_version"]),
        "automatic_promotion": False,
        "human_decision_required": True,
    }
