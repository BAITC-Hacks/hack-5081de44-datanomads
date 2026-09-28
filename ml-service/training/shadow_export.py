"""Export paired fresh predictions with verified operator and dataset lineage."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from training.contracts import _checksum
from training.feedback_dataset import ID_RE, _unique_object
from training.feedback_export import ReviewLink, ReviewedFeedbackLink, RuntimeFeedbackLink, runtime_origin_valid
from training.shadow_eval import ShadowPolicy, ShadowRecord


SHADOW_CONTEXT_QUERY = """
SELECT lc.id, lc.cycle_id, lc.state, lc.production_model_version,
       lc.candidate_model_version, lc.promotion_policy_version,
       lc.evaluation_started_at, lc.evaluation_ends_at,
       mv.artifact_checksum AS candidate_artifact_checksum
FROM learning_cycles lc
JOIN model_versions mv ON mv.model_version = lc.candidate_model_version
WHERE lc.cycle_id = $1
"""

SHADOW_ROWS_QUERY = """
SELECT sp.id AS shadow_id, sp.ticket_id AS db_ticket_id,
       t.created_at AS ticket_created_at, t.created_in_pulse_at,
       t.source_system, t.external_ticket_id, t.original_text,
       (SELECT COUNT(*)::int FROM audit_log al
        WHERE al.action = 'CREATE_TICKET' AND al.entity_type = 'ticket'
          AND al.entity_id = t.id::text AND al.metadata->>'source' = t.source_system
          AND al.actor_id IS NOT NULL AND al.request_id IS NOT NULL
          AND al.created_at >= t.created_in_pulse_at) AS api_create_audit_count,
       tp.model_version AS production_model_version,
       tp.topic_id AS production_topic_id,
       tp.confidence AS production_confidence,
       tp.created_at AS production_predicted_at,
       sp.candidate_model_version, sp.candidate_artifact_checksum,
       sp.topic_id AS candidate_topic_id,
       sp.confidence AS candidate_confidence,
       sp.predicted_at,
       od.id AS decision_id, od.decision AS decision_kind,
       od.confirmed_topic_id, od.created_at AS decision_created_at,
       (SELECT COUNT(*) FROM operator_decisions prior
        WHERE prior.ticket_id = sp.ticket_id AND prior.created_at <= sp.predicted_at) AS prior_decisions,
       (SELECT COUNT(*) FROM operator_decisions decisions
        WHERE decisions.ticket_id = sp.ticket_id AND decisions.created_at > sp.predicted_at) AS later_decisions,
       COALESCE((
           SELECT json_agg(json_build_object(
               'dataset_version', dv.dataset_version,
               'is_synthetic', dv.is_synthetic,
               'manifest_sha256', dv.manifest_sha256,
               'content_sha256', dv.content_sha256
           ) ORDER BY dv.dataset_version)::text
           FROM dataset_ticket_links dtl
           JOIN dataset_versions dv ON dv.dataset_version = dtl.dataset_version
           WHERE dtl.ticket_id = sp.ticket_id
       ), '[]') AS source_lineage
FROM classifier_shadow_predictions sp
JOIN tickets t ON t.id = sp.ticket_id
JOIN ticket_predictions tp ON tp.id = sp.production_prediction_id AND tp.ticket_id = sp.ticket_id
LEFT JOIN LATERAL (
    SELECT id, decision, confirmed_topic_id, created_at
    FROM operator_decisions
    WHERE ticket_id = sp.ticket_id AND created_at > sp.predicted_at
    ORDER BY created_at, id LIMIT 1
) od ON TRUE
WHERE sp.cycle_id = $1
ORDER BY sp.id
"""


def validate_context(context: dict, policy: ShadowPolicy) -> None:
    if (context["state"] not in {"EVALUATE", "DECISION"} or
            not all(isinstance(context[key], str) and ID_RE.fullmatch(context[key])
                    for key in ("cycle_id", "production_model_version", "candidate_model_version")) or
            context["production_model_version"] == context["candidate_model_version"] or
            context["promotion_policy_version"] != policy.promotion_policy_version or
            context["evaluation_started_at"] != policy.window_start or
            (context["evaluation_ends_at"] is not None and
             context["evaluation_ends_at"] < policy.window_end) or
            policy.window_end > datetime.now(timezone.utc)):
        raise ValueError("shadow evaluation context or window is invalid")
    _checksum(context["candidate_artifact_checksum"])


def _export_row(row: dict, context: dict, policy: ShadowPolicy,
                review_links: dict[int, ReviewLink]) -> tuple[dict | None, str | None]:
    try:
        lineage = json.loads(row["source_lineage"], object_pairs_hook=_unique_object)
    except (KeyError, TypeError, ValueError):
        return None, "DATASET_LINEAGE_UNVERIFIED"
    review = review_links.get(row["db_ticket_id"])
    if lineage == []:
        if not isinstance(review, RuntimeFeedbackLink):
            return None, "DATASET_LINEAGE_UNVERIFIED"
        if not runtime_origin_valid(row, review, lineage):
            return None, "RUNTIME_ORIGIN_UNVERIFIED"
        source_dataset_version = None
        source_origin_kind = "RUNTIME_API"
        is_synthetic = review.is_synthetic
    else:
        if not isinstance(lineage, list) or len(lineage) != 1 or isinstance(review, RuntimeFeedbackLink):
            return None, "DATASET_LINEAGE_UNVERIFIED"
        source = lineage[0]
        if (not isinstance(source, dict) or not isinstance(source.get("is_synthetic"), bool) or
                not isinstance(source.get("dataset_version"), str)):
            return None, "DATASET_LINEAGE_UNVERIFIED"
        try:
            _checksum(source["manifest_sha256"])
            _checksum(source["content_sha256"])
        except (KeyError, TypeError, ValueError):
            return None, "DATASET_LINEAGE_UNVERIFIED"
        source_dataset_version = source["dataset_version"]
        source_origin_kind = "IMPORTED_DATASET"
        is_synthetic = source["is_synthetic"]
        if not is_synthetic:
            text = row["original_text"]
            if (not isinstance(review, ReviewedFeedbackLink) or
                    review.source_dataset_version != source_dataset_version or
                    not isinstance(text, str) or not text.strip() or
                    "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest() != review.text_review_sha256):
                return None, "REAL_TEXT_NOT_REVIEWED"
    if (row["prior_decisions"] != 0 or row["later_decisions"] != 1 or
            row["decision_id"] is None or row["confirmed_topic_id"] is None or
            row["decision_kind"] not in {"CONFIRMED", "CORRECTED"} or
            (row["decision_kind"] == "CONFIRMED" and
             row["confirmed_topic_id"] != row["production_topic_id"])):
        return None, "OPERATOR_DECISION_UNVERIFIED"
    if (row["production_model_version"] != context["production_model_version"] or
            row["candidate_model_version"] != context["candidate_model_version"] or
            row["candidate_artifact_checksum"] != context["candidate_artifact_checksum"]):
        return None, "MODEL_LINEAGE_CHANGED"
    start, end = policy.window_start, policy.window_end
    if (not start <= row["ticket_created_at"] <= row["production_predicted_at"] <
            row["predicted_at"] < row["decision_created_at"] < end):
        return None, "OUTSIDE_PAIRED_WINDOW"
    action = "confirm" if row["decision_kind"] == "CONFIRMED" else "correct"
    try:
        record_data = {
            "contract_version": "classifier-shadow-input.v2" if source_origin_kind == "RUNTIME_API" else "classifier-shadow-input.v1",
            "feedback_id": f"decision_{row['decision_id']}",
            "cycle_id": context["cycle_id"],
            "ticket_id": f"ticket_{row['db_ticket_id']}",
            "is_synthetic": is_synthetic,
            "production_model_version": context["production_model_version"],
            "candidate_model_version": context["candidate_model_version"],
            "production_prediction": {"topic_id": row["production_topic_id"],
                                      "confidence": float(row["production_confidence"])},
            "candidate_shadow_prediction": {"topic_id": row["candidate_topic_id"],
                                            "confidence": float(row["candidate_confidence"])},
            "operator_confirmed_decision": {"decision_id": f"decision_{row['decision_id']}",
                                            "action": action,
                                            "topic_id": row["confirmed_topic_id"]},
            "accepted_or_corrected": "ACCEPTED" if action == "confirm" else "CORRECTED",
            "feedback_created_at": row["decision_created_at"],
            "validation_status": "VALID",
        }
        if source_origin_kind == "RUNTIME_API":
            record_data["source_origin_kind"] = source_origin_kind
        else:
            record_data["source_dataset_version"] = source_dataset_version
        record = ShadowRecord.model_validate(record_data)
    except (KeyError, TypeError, ValueError):
        return None, "INVALID_PAIRED_PREDICTION"
    return record.model_dump(mode="json", exclude_unset=True), None


def export_shadow(rows: list[dict], context: dict, policy: ShadowPolicy, output: Path,
                  *, review_links: dict[int, ReviewLink] | None = None) -> dict:
    validate_context(context, policy)
    if output.exists() or output.is_symlink():
        raise FileExistsError("shadow export already exists")
    rejected: Counter = Counter()
    records = []
    for row in rows:
        record, reason = _export_row(row, context, policy, review_links or {})
        if reason:
            rejected[reason] += 1
        else:
            records.append(record)
    result = {
        "status": "COMPLETED" if records else "INSUFFICIENT_EVIDENCE",
        "cycle_id": context["cycle_id"],
        "production_model_version": context["production_model_version"],
        "candidate_model_version": context["candidate_model_version"],
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
