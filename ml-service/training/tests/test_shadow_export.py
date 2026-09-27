from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from training.feedback_export import ReviewedFeedbackLink
from training.shadow_eval import ShadowPolicy, evaluate_shadow
from training.shadow_export import export_shadow
from scripts.record_classifier_shadow_report import evaluation_decision


START = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)
TEXT = "На дороге появилась яма рядом с остановкой."
CHECKSUM = "sha256:" + "a" * 64


def context() -> dict:
    return {
        "cycle_id": "cycle_1", "state": "EVALUATE",
        "production_model_version": "production_v1",
        "candidate_model_version": "candidate_v1",
        "candidate_artifact_checksum": CHECKSUM,
        "promotion_policy_version": "policy-v1",
        "evaluation_started_at": START,
        "evaluation_ends_at": START + timedelta(hours=1),
    }


def policy() -> ShadowPolicy:
    return ShadowPolicy.model_validate({
        "policy_version": "classifier-shadow-policy.v1",
        "promotion_policy_version": "policy-v1",
        "window_start": START,
        "window_end": START + timedelta(hours=1),
        "min_samples": 1, "min_real_samples": 1,
        "critical_topics": ["roads"], "min_topic_support": 1,
        "max_topic_agreement_drop": 0, "max_correction_rate_increase": 0,
    })


def row(*, synthetic: bool = True) -> dict:
    return {
        "db_ticket_id": 7, "ticket_created_at": START + timedelta(minutes=1),
        "original_text": TEXT,
        "production_model_version": "production_v1",
        "production_topic_id": "roads", "production_confidence": Decimal("0.90000"),
        "production_predicted_at": START + timedelta(minutes=2),
        "candidate_model_version": "candidate_v1",
        "candidate_artifact_checksum": CHECKSUM,
        "candidate_topic_id": "roads", "candidate_confidence": Decimal("0.80000"),
        "predicted_at": START + timedelta(minutes=3),
        "decision_id": 11, "decision_kind": "CONFIRMED",
        "confirmed_topic_id": "roads", "decision_created_at": START + timedelta(minutes=4),
        "prior_decisions": 0, "later_decisions": 1,
        "source_lineage": json.dumps([{
            "dataset_version": "source_v1", "is_synthetic": synthetic,
            "manifest_sha256": CHECKSUM, "content_sha256": CHECKSUM,
        }]),
    }


def review_link() -> ReviewedFeedbackLink:
    return ReviewedFeedbackLink.model_validate({
        "contract_version": "feedback-review-link.v1", "db_ticket_id": 7,
        "ticket_id": "source_ticket_7", "split_group": "incident_7",
        "source_dataset_version": "source_v1",
        "text_review_sha256": "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest(),
        "review_status": "APPROVED", "reviewer_id": "reviewer_1",
        "reviewed_at": START,
    })


class ShadowExportTests(unittest.TestCase):
    def test_export_can_be_evaluated_without_ticket_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "shadow.jsonl"
            result = export_shadow([row(synthetic=False)], context(), policy(), output,
                                   review_links={7: review_link()})
            self.assertEqual(result["accepted_count"], 1)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(TEXT, output.read_text(encoding="utf-8"))
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported["feedback_id"], "decision_11")
            self.assertFalse(exported["is_synthetic"])
            policy_path = root / "policy.json"
            policy_path.write_text(policy().model_dump_json(), encoding="utf-8")
            report = evaluate_shadow(output, policy_path, cycle_id="cycle_1",
                                     production_model_version="production_v1",
                                     candidate_model_version="candidate_v1")
            self.assertEqual(report["status"], "VALID")
            self.assertEqual(report["decision"], "PENDING_HUMAN_REVIEW")

    def test_rejects_unreviewed_real_text_and_ambiguous_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = row(synthetic=False)
            result = export_shadow([real], context(), policy(), root / "unreviewed.jsonl")
            self.assertEqual(result["rejected_counts"], {"REAL_TEXT_NOT_REVIEWED": 1})
            self.assertFalse((root / "unreviewed.jsonl").exists())
            bad_rows = [row() for _ in range(4)]
            bad_rows[0]["source_lineage"] = "[]"
            bad_rows[1]["prior_decisions"] = 1
            bad_rows[2]["decision_created_at"] = START + timedelta(hours=1)
            bad_rows[3]["candidate_artifact_checksum"] = "sha256:" + "b" * 64
            result = export_shadow(bad_rows, context(), policy(), root / "invalid.jsonl")
            self.assertEqual(result["accepted_count"], 0)
            self.assertEqual(result["rejected_counts"], {
                "DATASET_LINEAGE_UNVERIFIED": 1,
                "MODEL_LINEAGE_CHANGED": 1,
                "OPERATOR_DECISION_UNVERIFIED": 1,
                "OUTSIDE_PAIRED_WINDOW": 1,
            })
            self.assertFalse((root / "invalid.jsonl").exists())

    def test_requires_exact_cycle_window_and_review_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = context()
            changed["evaluation_started_at"] += timedelta(seconds=1)
            with self.assertRaisesRegex(ValueError, "context or window"):
                export_shadow([row()], changed, policy(), root / "invalid.jsonl")
            bad_review = review_link().model_copy(update={"text_review_sha256": CHECKSUM})
            result = export_shadow([row(synthetic=False)], context(), policy(), root / "real.jsonl",
                                   review_links={7: bad_review})
            self.assertEqual(result["rejected_counts"], {"REAL_TEXT_NOT_REVIEWED": 1})

    def test_report_only_becomes_reviewable_with_matching_offline_evidence(self) -> None:
        offline = {
            "report_version": "classifier-pair-evaluation.v1",
            "decision": "PENDING_HUMAN_REVIEW",
            "regressed_critical_topics": [],
            "candidate": {"model_version": "candidate_v1", "artifact_checksum": CHECKSUM},
            "production": {"model_version": "production_v1", "artifact_checksum": CHECKSUM},
        }
        report = {"decision": "PENDING_HUMAN_REVIEW"}
        self.assertEqual(evaluation_decision(offline, report, context(), CHECKSUM),
                         "READY_TO_REVIEW")
        self.assertEqual(evaluation_decision(offline, report, context(), None), "NOT_READY")
        report["decision"] = "INSUFFICIENT_EVIDENCE"
        self.assertEqual(evaluation_decision(offline, report, context(), CHECKSUM),
                         "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
