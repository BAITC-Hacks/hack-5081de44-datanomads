from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.challenger_eval import compare_challengers


CHECKSUM = "sha256:" + "a" * 64


def offline(version: str) -> dict:
    return {
        "report_version": "classifier-pair-evaluation.v1",
        "dataset_version": "reviewed_v1", "dataset_content_sha256": CHECKSUM,
        "frozen_evaluation_version": "eval_v1", "frozen_evaluation_sha256": CHECKSUM,
        "synthetic": False, "sample_count": 30, "sample_ids_sha256": CHECKSUM,
        "labels": ["roads", "water_supply"], "policy_sha256": CHECKSUM,
        "policy": {"policy_version": "classifier-critical-regression.v1", "min_total_samples": 30},
        "production": {"model_version": "champion_v1", "artifact_checksum": CHECKSUM,
                       "metrics": {"macro_f1": 0.8, "sample_count": 30}},
        "candidate": {"model_version": version, "artifact_checksum": CHECKSUM,
                      "metrics": {"macro_f1": 0.82, "sample_count": 30}},
        "macro_f1_delta": 0.02, "insufficient_critical_topics": [],
        "regressed_critical_topics": [],
        "decision": "PENDING_HUMAN_REVIEW",
    }


def shadow(version: str) -> dict:
    return {
        "report_version": "classifier-shadow-evaluation.v1",
        "gate_population": "real_only.v1",
        "cycle_id": f"cycle_{version}",
        "production_model_version": "champion_v1", "candidate_model_version": version,
        "promotion_policy_version": "policy-v1",
        "window_start": "2026-09-01T00:00:00+00:00",
        "window_end": "2026-09-20T00:00:00+00:00",
        "sample_count": 40, "sample_ids_sha256": CHECKSUM,
        "champion_reference_sha256": CHECKSUM,
        "policy_sha256": CHECKSUM,
        "policy": {"policy_version": "classifier-shadow-policy.v1",
                   "promotion_policy_version": "policy-v1",
                   "min_samples": 30, "min_real_samples": 30,
                   "max_correction_rate_increase": 0.05,
                   "window_start": "2026-09-01T00:00:00+00:00",
                   "window_end": "2026-09-20T00:00:00+00:00"},
        "origin_counts": {"real": 40},
        "production_agreement": 0.8, "production_correction_rate": 0.2,
        "candidate_agreement": 0.825, "correction_rate_delta": -0.025,
        "real_production_agreement": 0.8,
        "real_candidate_agreement": 0.825,
        "real_correction_rate_delta": -0.025,
        "global_regression": False,
        "status": "VALID", "decision": "PENDING_HUMAN_REVIEW",
        "insufficient_critical_topics": [], "critical_regressions": [],
    }


class ChallengerEvaluationTests(unittest.TestCase):
    def test_compares_two_candidates_only_on_identical_champion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for version in ("candidate_a", "candidate_b"):
                offline_path = root / f"{version}-offline.json"
                shadow_path = root / f"{version}-shadow.json"
                offline_path.write_text(json.dumps(offline(version)), encoding="utf-8")
                shadow_path.write_text(json.dumps(shadow(version)), encoding="utf-8")
                inputs.append((offline_path, shadow_path))
            report = compare_challengers(inputs)
            self.assertEqual(report["status"], "COMPARABLE")
            self.assertEqual(report["fresh_sample_count"], 40)
            self.assertEqual(len(report["candidates"]), 2)
            self.assertTrue(all(item["ready_for_human_review"] for item in report["candidates"]))
            self.assertEqual(report["candidates"][0]["fresh_real_correction_rate_delta"], -0.025)
            self.assertFalse(report["automatic_promotion"])
            self.assertNotIn("original_text", json.dumps(report))

            changed = shadow("candidate_b")
            changed["champion_reference_sha256"] = "sha256:" + "b" * 64
            inputs[1][1].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical frozen and fresh"):
                compare_challengers(inputs)

            inputs[1][0].write_text(json.dumps(offline("candidate_b")), encoding="utf-8")
            legacy_shadow = shadow("candidate_b")
            del legacy_shadow["gate_population"]
            inputs[1][1].write_text(json.dumps(legacy_shadow), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid identity or evidence"):
                compare_challengers(inputs)

            inputs[1][0].write_text(json.dumps(offline("123456789012")), encoding="utf-8")
            inputs[1][1].write_text(json.dumps(shadow("123456789012")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid identity"):
                compare_challengers(inputs)

            inputs[1][1].write_text(json.dumps(shadow("candidate_b")), encoding="utf-8")
            changed_offline = offline("candidate_b")
            changed_offline["sample_ids_sha256"] = "sha256:" + "b" * 64
            inputs[1][0].write_text(json.dumps(changed_offline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical frozen and fresh"):
                compare_challengers(inputs)

    def test_synthetic_or_insufficient_evidence_does_not_become_review_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for version in ("candidate_a", "candidate_b"):
                offline_data = offline(version)
                offline_data["synthetic"] = True
                offline_path = root / f"{version}-offline.json"
                shadow_path = root / f"{version}-shadow.json"
                offline_path.write_text(json.dumps(offline_data), encoding="utf-8")
                shadow_path.write_text(json.dumps(shadow(version)), encoding="utf-8")
                inputs.append((offline_path, shadow_path))
            report = compare_challengers(inputs)
            self.assertFalse(any(item["ready_for_human_review"] for item in report["candidates"]))
            changed = shadow("candidate_b")
            changed["status"] = "INSUFFICIENT_EVIDENCE"
            changed["decision"] = "INSUFFICIENT_EVIDENCE"
            changed["insufficient_critical_topics"] = ["roads"]
            inputs[1][1].write_text(json.dumps(changed), encoding="utf-8")
            report = compare_challengers(inputs)
            self.assertEqual(report["candidates"][1]["fresh_decision"], "INSUFFICIENT_EVIDENCE")
            self.assertFalse(report["candidates"][1]["ready_for_human_review"])

    def test_rejects_decisions_inconsistent_with_report_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for version in ("candidate_a", "candidate_b"):
                offline_path = root / f"{version}-offline.json"
                shadow_path = root / f"{version}-shadow.json"
                offline_path.write_text(json.dumps(offline(version)), encoding="utf-8")
                shadow_path.write_text(json.dumps(shadow(version)), encoding="utf-8")
                inputs.append((offline_path, shadow_path))

            changed_offline = offline("candidate_b")
            changed_offline["insufficient_critical_topics"] = ["roads"]
            inputs[1][0].write_text(json.dumps(changed_offline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid identity or evidence"):
                compare_challengers(inputs)

            inputs[1][0].write_text(json.dumps(offline("candidate_b")), encoding="utf-8")
            changed_shadow = shadow("candidate_b")
            changed_shadow["critical_regressions"] = ["roads"]
            inputs[1][1].write_text(json.dumps(changed_shadow), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid identity or evidence"):
                compare_challengers(inputs)


if __name__ == "__main__":
    unittest.main()
