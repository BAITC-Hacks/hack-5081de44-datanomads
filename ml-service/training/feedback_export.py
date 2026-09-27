"""Export reviewed Core feedback into the offline candidate input contract."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from data.normalization.pii import scan_pii
from training.contracts import _checksum
from training.feedback_dataset import FeedbackRecord, ID_RE


FEEDBACK_QUERY = """
SELECT lf.id AS feedback_id, lf.ticket_id AS db_ticket_id,
       lf.production_model_version, lf.production_prediction::text,
       lf.operator_confirmed_decision::text, lf.accepted_or_corrected,
       lf.validation_status, lf.feedback_created_at,
       t.original_text, t.language,
       od.id AS operator_decision_id, od.decision AS operator_decision_kind,
       od.confirmed_topic_id,
       tp.model_version AS prediction_model_version, tp.topic_id AS prediction_topic_id,
       tp.confidence AS prediction_confidence,
       COALESCE((
           SELECT json_agg(json_build_object(
               'dataset_version', dv.dataset_version,
               'is_synthetic', dv.is_synthetic,
               'manifest_sha256', dv.manifest_sha256,
               'content_sha256', dv.content_sha256
           ) ORDER BY dv.dataset_version)::text
           FROM dataset_ticket_links dtl
           JOIN dataset_versions dv ON dv.dataset_version = dtl.dataset_version
           WHERE dtl.ticket_id = lf.ticket_id
       ), '[]') AS source_lineage
FROM learning_feedback lf
JOIN learning_cycles lc ON lc.id = lf.cycle_id
JOIN tickets t ON t.id = lf.ticket_id
LEFT JOIN operator_decisions od
    ON od.ticket_id = lf.ticket_id
   AND od.id::text = lf.operator_confirmed_decision->>'decision_id'
LEFT JOIN LATERAL (
    SELECT model_version, topic_id, confidence
    FROM ticket_predictions
    WHERE ticket_id = lf.ticket_id AND model_version = lf.production_model_version
      AND created_at <= lf.feedback_created_at
    ORDER BY created_at DESC, id DESC LIMIT 1
) tp ON TRUE
WHERE lc.cycle_id = $1 OR lc.id::text = $1
ORDER BY lf.id
"""


class ReviewedFeedbackLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["feedback-review-link.v1"]
    db_ticket_id: int = Field(gt=0)
    ticket_id: str
    split_group: str
    source_dataset_version: str
    text_review_sha256: str
    review_status: Literal["APPROVED"]
    reviewer_id: str
    reviewed_at: AwareDatetime

    @field_validator("ticket_id", "split_group", "source_dataset_version", "reviewer_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not ID_RE.fullmatch(value) or scan_pii(value).detected:
            raise ValueError("review identifier is invalid")
        return value

    @field_validator("text_review_sha256")
    @classmethod
    def valid_checksum(cls, value: str) -> str:
        return _checksum(value)


def load_review_links(path: Path) -> dict[int, ReviewedFeedbackLink]:
    links = {}
    export_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            json.loads(line, object_pairs_hook=_unique_object)
            link = ReviewedFeedbackLink.model_validate_json(line)
            if link.db_ticket_id in links or link.ticket_id in export_ids:
                raise ValueError("duplicate reviewed ticket link")
            links[link.db_ticket_id] = link
            export_ids.add(link.ticket_id)
    if not links:
        raise ValueError("review link file is empty")
    return links


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_value(value: str | dict | list) -> object:
    return json.loads(value, object_pairs_hook=_unique_object) if isinstance(value, str) else value


def _export_row(row: dict, link: ReviewedFeedbackLink, cycle_id: str, production_model_version: str) -> tuple[dict | None, str | None]:
    text = row["original_text"]
    if not isinstance(text, str) or not text.strip() or scan_pii(text).detected:
        return None, "TEXT_MISSING_OR_PII"
    if "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest() != link.text_review_sha256:
        return None, "TEXT_NOT_REVIEWED"
    try:
        lineage = _json_value(row["source_lineage"])
    except ValueError:
        return None, "DATASET_LINEAGE_UNVERIFIED"
    if not isinstance(lineage, list) or len(lineage) != 1:
        return None, "DATASET_LINEAGE_UNVERIFIED"
    source = lineage[0]
    if (not isinstance(source, dict) or source.get("dataset_version") != link.source_dataset_version or
            not isinstance(source.get("is_synthetic"), bool)):
        return None, "DATASET_LINEAGE_UNVERIFIED"
    try:
        _checksum(source["content_sha256"])
        _checksum(source["manifest_sha256"])
    except (KeyError, TypeError, ValueError):
        return None, "DATASET_LINEAGE_UNVERIFIED"
    if row["validation_status"] != "VALID":
        return None, "INVALID_FEEDBACK_STATUS"
    if row["production_model_version"] != production_model_version:
        return None, "WRONG_PRODUCTION_MODEL"
    try:
        prediction = _json_value(row["production_prediction"])
        decision = _json_value(row["operator_confirmed_decision"])
    except ValueError:
        return None, "MISSING_DECISION_EVIDENCE"
    if not isinstance(prediction, dict) or not isinstance(decision, dict):
        return None, "MISSING_DECISION_EVIDENCE"
    action = decision.get("action")
    expected_decision = "CONFIRMED" if action == "confirm" else "CORRECTED" if action == "correct" else None
    expected_feedback = "ACCEPTED" if action == "confirm" else "CORRECTED" if action == "correct" else None
    if (expected_decision is None or row["operator_decision_id"] is None or
            str(row["operator_decision_id"]) != decision.get("decision_id") or
            row["operator_decision_kind"] != expected_decision or
            row["confirmed_topic_id"] != decision.get("topic_id") or
            row["accepted_or_corrected"] != expected_feedback):
        return None, "MISSING_DECISION_EVIDENCE"
    if (row["prediction_model_version"] != production_model_version or
            row["prediction_topic_id"] != prediction.get("topic_id")):
        return None, "MISSING_PREDICTION_EVIDENCE"
    try:
        confidence = prediction["confidence"]
        if (isinstance(confidence, bool) or row["prediction_confidence"] is None or
                not math.isclose(float(confidence), float(row["prediction_confidence"]),
                                 rel_tol=0, abs_tol=0.00002)):
            return None, "MISSING_PREDICTION_EVIDENCE"
    except (KeyError, TypeError, ValueError):
        return None, "MISSING_PREDICTION_EVIDENCE"
    try:
        record = FeedbackRecord.model_validate({
            "contract_version": "learning-feedback-export.v1",
            "feedback_id": f"feedback_{row['feedback_id']}",
            "cycle_id": cycle_id,
            "ticket_id": link.ticket_id,
            "split_group": link.split_group,
            "source_dataset_version": link.source_dataset_version,
            "is_synthetic": source["is_synthetic"],
            "original_text": text,
            "language": row["language"],
            "production_model_version": production_model_version,
            "production_prediction": prediction,
            "operator_confirmed_decision": decision,
            "accepted_or_corrected": row["accepted_or_corrected"],
            "feedback_created_at": row["feedback_created_at"],
            "validation_status": row["validation_status"],
        })
    except (ValueError, TypeError):
        return None, "INVALID_EXPORT_SCHEMA"
    return record.model_dump(mode="json"), None


def export_feedback(rows: list[dict], links: dict[int, ReviewedFeedbackLink], output: Path,
                    *, cycle_id: str, production_model_version: str) -> dict:
    if (not ID_RE.fullmatch(cycle_id) or not ID_RE.fullmatch(production_model_version) or
            scan_pii(cycle_id).detected or scan_pii(production_model_version).detected):
        raise ValueError("cycle or model version is invalid")
    if output.exists() or output.is_symlink():
        raise FileExistsError("feedback export already exists")
    rejected: Counter = Counter()
    records = []
    for row in rows:
        link = links.get(row["db_ticket_id"])
        if link is None:
            rejected["NO_APPROVED_REVIEW_LINK"] += 1
            continue
        record, reason = _export_row(row, link, cycle_id, production_model_version)
        if reason:
            rejected[reason] += 1
        else:
            records.append(record)
    result = {
        "status": "COMPLETED" if records else "INSUFFICIENT_FEEDBACK",
        "cycle_id": cycle_id,
        "production_model_version": production_model_version,
        "input_count": len(rows),
        "accepted_count": len(records),
        "rejected_counts": dict(sorted(rejected.items())),
    }
    if records:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return result
