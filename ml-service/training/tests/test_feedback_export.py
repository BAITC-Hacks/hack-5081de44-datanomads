from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from training.feedback_export import RuntimeFeedbackLink, ReviewedFeedbackLink, export_feedback, load_review_links


CHECKSUM = "sha256:" + "a" * 64
TEXT = "На дороге появилась яма рядом с остановкой."
API_ID = "api-1790467200000000000"


def link() -> ReviewedFeedbackLink:
    return ReviewedFeedbackLink.model_validate({
        "contract_version": "feedback-review-link.v1",
        "db_ticket_id": 7,
        "ticket_id": "source_ticket_7",
        "split_group": "incident_7",
        "source_dataset_version": "source_v1",
        "text_review_sha256": "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest(),
        "review_status": "APPROVED",
        "reviewer_id": "reviewer_1",
        "reviewed_at": datetime(2026, 9, 27, 12, tzinfo=timezone.utc),
    })


def feedback_row() -> dict:
    return {
        "feedback_id": 13,
        "db_ticket_id": 7,
        "production_model_version": "production_v1",
        "production_prediction": json.dumps({"topic_id": "roads", "confidence": 0.8}),
        "operator_confirmed_decision": json.dumps({"decision_id": "9", "action": "correct", "topic_id": "electricity"}),
        "accepted_or_corrected": "CORRECTED",
        "validation_status": "VALID",
        "feedback_created_at": datetime(2026, 9, 27, tzinfo=timezone.utc),
        "original_text": TEXT,
        "language": "RU",
        "operator_decision_id": 9,
        "operator_decision_kind": "CORRECTED",
        "confirmed_topic_id": "electricity",
        "prediction_model_version": "production_v1",
        "prediction_topic_id": "roads",
        "prediction_confidence": Decimal("0.80000"),
        "source_lineage": json.dumps([{
            "dataset_version": "source_v1", "is_synthetic": False,
            "manifest_sha256": CHECKSUM, "content_sha256": CHECKSUM,
        }]),
    }


def runtime_link(*, synthetic: bool = False) -> RuntimeFeedbackLink:
    return RuntimeFeedbackLink.model_validate({
        "contract_version": "feedback-review-link.v2", "db_ticket_id": 7,
        "ticket_id": "runtime_ticket_7", "split_group": "incident_7",
        "source_kind": "RUNTIME_API", "source_system": "api",
        "external_ticket_id": API_ID, "is_synthetic": synthetic,
        "text_review_sha256": "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest(),
        "review_status": "APPROVED", "reviewer_id": "reviewer_1",
        "reviewed_at": datetime(2026, 9, 27, 12, tzinfo=timezone.utc),
    })


def runtime_feedback_row() -> dict:
    created = datetime(2026, 9, 27, tzinfo=timezone.utc)
    return {**feedback_row(), "source_lineage": "[]",
            "ticket_created_at": created, "created_in_pulse_at": created + timedelta(seconds=1),
            "source_system": "api", "external_ticket_id": API_ID,
            "api_create_audit_count": 1}


class FeedbackExportTests(unittest.TestCase):
    def test_runtime_origin_requires_exact_review_and_remains_separate_from_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for synthetic in (False, True):
                output = root / f"runtime-{synthetic}.jsonl"
                result = export_feedback([runtime_feedback_row()], {7: runtime_link(synthetic=synthetic)},
                                         output, cycle_id="cycle_1", production_model_version="production_v1")
                self.assertEqual(result["accepted_count"], 1)
                exported = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(exported["contract_version"], "learning-feedback-export.v2")
                self.assertEqual(exported["source_origin_kind"], "RUNTIME_API")
                self.assertNotIn("source_dataset_version", exported)
                self.assertEqual(exported["is_synthetic"], synthetic)
            base = runtime_feedback_row()
            invalid = [
                ({**base, "source_lineage": feedback_row()["source_lineage"]}, "RUNTIME_ORIGIN_UNVERIFIED"),
                ({**base, "external_ticket_id": "api-1790467200000000001"}, "RUNTIME_ORIGIN_UNVERIFIED"),
                ({**base, "source_system": "other"}, "RUNTIME_ORIGIN_UNVERIFIED"),
                ({**base, "ticket_created_at": datetime(2026, 9, 26, tzinfo=timezone.utc)}, "RUNTIME_ORIGIN_UNVERIFIED"),
                ({**base, "api_create_audit_count": 0}, "RUNTIME_ORIGIN_UNVERIFIED"),
                ({**base, "original_text": "ИИН 000000000000"}, "TEXT_MISSING_OR_PII"),
            ]
            for index, (candidate, reason) in enumerate(invalid):
                result = export_feedback([candidate], {7: runtime_link()}, root / f"bad-{index}.jsonl",
                                         cycle_id="cycle_1", production_model_version="production_v1")
                self.assertEqual(result["rejected_counts"], {reason: 1})
            result = export_feedback([base], {}, root / "unreviewed.jsonl",
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["rejected_counts"], {"NO_APPROVED_REVIEW_LINK": 1})
            bad_hash = runtime_link().model_copy(update={"text_review_sha256": CHECKSUM})
            result = export_feedback([base], {7: bad_hash}, root / "bad-hash.jsonl",
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["rejected_counts"], {"TEXT_NOT_REVIEWED": 1})

    def test_exports_verified_operator_label_without_db_or_reviewer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "feedback.jsonl"
            result = export_feedback([feedback_row()], {7: link()}, output,
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["accepted_count"], 1)
            self.assertEqual(result["rejected_counts"], {})
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported["operator_confirmed_decision"]["topic_id"], "electricity")
            self.assertEqual(exported["production_prediction"]["topic_id"], "roads")
            self.assertEqual(exported["split_group"], "incident_7")
            self.assertFalse(exported["is_synthetic"])
            self.assertNotIn("reviewer_id", exported)
            self.assertNotIn("db_ticket_id", exported)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                export_feedback([feedback_row()], {7: link()}, output,
                                cycle_id="cycle_1", production_model_version="production_v1")

    def test_rejects_unreviewed_text_lineage_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = [feedback_row() for _ in range(5)]
            rows[0]["db_ticket_id"] = 99
            rows[1]["original_text"] = "Обращение изменено после проверки."
            rows[2]["source_lineage"] = "[]"
            rows[3]["operator_decision_kind"] = "CONFIRMED"
            rows[4]["original_text"] = "ИИН 000000000000"
            output = Path(directory) / "feedback.jsonl"
            result = export_feedback(rows, {7: link()}, output,
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["status"], "INSUFFICIENT_FEEDBACK")
            self.assertEqual(result["rejected_counts"], {
                "DATASET_LINEAGE_UNVERIFIED": 1,
                "MISSING_DECISION_EVIDENCE": 1,
                "NO_APPROVED_REVIEW_LINK": 1,
                "TEXT_MISSING_OR_PII": 1,
                "TEXT_NOT_REVIEWED": 1,
            })
            self.assertFalse(output.exists())

    def test_rejects_invalid_review_file_and_unknown_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "links.jsonl"
            valid = link().model_dump(mode="json")
            path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_review_links(path)
            valid["review_status"] = "PENDING"
            path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_review_links(path)
            valid["review_status"] = "APPROVED"
            path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            self.assertEqual(load_review_links(path)[7].split_group, "incident_7")
            runtime = runtime_link().model_dump(mode="json")
            path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            self.assertIsInstance(load_review_links(path)[7], RuntimeFeedbackLink)
            runtime["is_synthetic"] = "false"
            path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_review_links(path)
            row = feedback_row()
            row["prediction_topic_id"] = "water_supply"
            result = export_feedback([row], {7: link()}, Path(directory) / "feedback.jsonl",
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["rejected_counts"], {"MISSING_PREDICTION_EVIDENCE": 1})
            row["prediction_topic_id"] = "roads"
            row["prediction_confidence"] = Decimal("0.70000")
            result = export_feedback([row], {7: link()}, Path(directory) / "feedback.jsonl",
                                     cycle_id="cycle_1", production_model_version="production_v1")
            self.assertEqual(result["rejected_counts"], {"MISSING_PREDICTION_EVIDENCE": 1})
