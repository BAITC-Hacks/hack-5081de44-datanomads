from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.shadow_eval import evaluate_shadow


def shadow_row(index: int, topic: str, candidate_topic: str) -> dict:
    return {
        "contract_version": "classifier-shadow-input.v1",
        "feedback_id": f"feedback_{index}",
        "cycle_id": "cycle_1",
        "ticket_id": f"ticket_{index}",
        "source_dataset_version": "runtime_v1",
        "is_synthetic": False,
        "production_model_version": "production_v1",
        "candidate_model_version": "candidate_v1",
        "production_prediction": {"topic_id": topic, "confidence": 0.9},
        "candidate_shadow_prediction": {"topic_id": candidate_topic, "confidence": 0.8},
        "operator_confirmed_decision": {"decision_id": f"decision_{index}", "action": "confirm", "topic_id": topic},
        "accepted_or_corrected": "ACCEPTED",
        "feedback_created_at": f"2026-09-27T10:0{index}:00Z",
        "validation_status": "VALID",
    }


class ShadowEvaluationTests(unittest.TestCase):
    def test_paired_fresh_metrics_policy_and_insufficient_real_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "shadow.jsonl"
            policy_path = root / "policy.json"
            rows = [shadow_row(1, "roads", "roads"),
                    shadow_row(2, "roads", "electricity"),
                    shadow_row(3, "water_supply", "water_supply")]
            policy = {
                "policy_version": "classifier-shadow-policy.v1",
                "promotion_policy_version": "policy-v1",
                "window_start": "2026-09-27T10:00:00Z",
                "window_end": "2026-09-27T11:00:00Z",
                "min_samples": 3,
                "min_real_samples": 3,
                "critical_topics": ["roads"],
                "min_topic_support": 2,
                "max_topic_agreement_drop": 0.2,
                "max_correction_rate_increase": 0.4,
            }

            def evaluate() -> dict:
                input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                policy_path.write_text(json.dumps(policy), encoding="utf-8")
                return evaluate_shadow(input_path, policy_path, cycle_id="cycle_1",
                                       production_model_version="production_v1",
                                       candidate_model_version="candidate_v1")

            report = evaluate()
            self.assertEqual(report["status"], "VALID")
            self.assertEqual(report["decision"], "NO_GO_CRITICAL_REGRESSION")
            self.assertEqual(report["critical_regressions"], ["roads"])
            self.assertEqual(report["production_agreement"], 1.0)
            self.assertEqual(report["candidate_agreement"], 0.666667)
            self.assertEqual(report["correction_rate_delta"], 0.333333)
            self.assertEqual(report["by_topic"]["roads"]["sample_count"], 2)
            self.assertFalse(report["blind_ab_enabled"])
            self.assertIsNone(report["blind_ab_preference"])
            self.assertNotIn("ticket_1", json.dumps(report))
            original_reference = report["champion_reference_sha256"]
            rows.reverse()
            self.assertEqual(evaluate()["champion_reference_sha256"], original_reference)
            rows.reverse()

            policy["max_topic_agreement_drop"] = 1.0
            self.assertEqual(evaluate()["decision"], "PENDING_HUMAN_REVIEW")
            policy["max_correction_rate_increase"] = 0.2
            self.assertEqual(evaluate()["decision"], "NO_GO_CORRECTION_RATE")
            policy["max_correction_rate_increase"] = 0.4
            rows[0]["is_synthetic"] = True
            self.assertEqual(evaluate()["status"], "INSUFFICIENT_EVIDENCE")
            self.assertNotEqual(evaluate()["champion_reference_sha256"], original_reference)
            rows[0]["candidate_model_version"] = "another_candidate"
            with self.assertRaisesRegex(ValueError, "versions, window or identity"):
                evaluate()

            input_path.write_text("", encoding="utf-8")
            empty = evaluate_shadow(input_path, policy_path, cycle_id="cycle_1",
                                    production_model_version="production_v1",
                                    candidate_model_version="candidate_v1")
            self.assertEqual(empty["status"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(empty["sample_count"], 0)
            self.assertIsNone(empty["candidate_agreement"])


if __name__ == "__main__":
    unittest.main()
