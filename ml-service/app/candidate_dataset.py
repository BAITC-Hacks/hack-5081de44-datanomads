"""Build immutable candidate datasets from validated feedback references."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from contracts import validate_document


_SAFE_VERSION_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _cycle_suffix(cycle_id: str) -> str:
    safe_value = _SAFE_VERSION_PART.sub("-", cycle_id).strip("-.")
    if not safe_value:
        raise RuntimeError("INVALID_CYCLE_ID")
    return safe_value[:80]


def _date_time(value: Any) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise RuntimeError("INVALID_FEEDBACK_TIMESTAMP")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _prediction(value: Any, fallback_model_version: str) -> dict[str, Any]:
    source = _json_object(value)
    return {
        "topic_id": source.get("topic_id"),
        "service_id": source.get("service_id"),
        "priority": source.get("priority"),
        "confidence": source.get("confidence"),
        "confidence_state": source.get("confidence_state"),
        "model_version": str(source.get("model_version") or fallback_model_version),
    }


def _confirmed_decision(value: Any, prediction: dict[str, Any], accepted: bool) -> dict[str, Any]:
    source = _json_object(value)
    topic_id = (
        source.get("topic_id")
        or source.get("confirmed_topic_id")
        or (prediction.get("topic_id") if accepted else None)
    )
    service_id = source.get("service_id") or source.get("confirmed_service_id") or source.get("service")
    priority = source.get("priority") or source.get("confirmed_priority")
    decision = {
        "topic_id": topic_id,
        "service_id": service_id,
        "priority": priority,
    }
    if not any(item is not None for item in decision.values()):
        raise RuntimeError("FEEDBACK_LABELS_UNAVAILABLE")
    return decision


async def export_validated_feedback(
    pool: Any,
    request_payload: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Write an ID-only LearningFeedbackExport and return the builder request."""

    cycle_id = str(request_payload.get("cycle_id") or "")
    production_model_version = str(request_payload.get("production_model_version") or "")
    feedback_ids = [str(value) for value in request_payload.get("feedback_ids", [])]
    evaluation_dataset_version = request_payload.get("frozen_evaluation_dataset_version")
    evaluation_ticket_ids = [
        str(value) for value in request_payload.get("frozen_evaluation_ticket_ids", [])
    ]
    if not production_model_version:
        raise RuntimeError("PRODUCTION_BASELINE_NOT_CONFIGURED")
    if not evaluation_dataset_version or not evaluation_ticket_ids:
        raise RuntimeError("FROZEN_EVALUATION_SET_NOT_CONFIGURED")
    if not feedback_ids:
        raise RuntimeError("VALIDATED_FEEDBACK_UNAVAILABLE")

    async with pool.acquire() as connection:
        cycle_db_id = await connection.fetchval(
            "SELECT id FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1 LIMIT 1",
            cycle_id,
        )
        if cycle_db_id is None:
            raise RuntimeError("LEARNING_CYCLE_NOT_FOUND")
        rows = await connection.fetch(
            """
            SELECT lf.id::text AS feedback_id,
                   lf.ticket_id::text AS ticket_id,
                   lf.production_model_version,
                   lf.production_prediction,
                   lf.operator_confirmed_decision,
                   lf.accepted_or_corrected,
                   lf.validation_status,
                   lf.feedback_created_at,
                   COALESCE(BOOL_OR(dv.is_synthetic), FALSE) AS source_is_synthetic,
                   COUNT(DISTINCT dtl.dataset_version)::int AS source_version_count
            FROM learning_feedback lf
            LEFT JOIN dataset_ticket_links dtl ON dtl.ticket_id = lf.ticket_id
            LEFT JOIN dataset_versions dv ON dv.dataset_version = dtl.dataset_version
            WHERE lf.cycle_id = $1
              AND lf.id = ANY($2::bigint[])
              AND lf.validation_status = 'VALID'
            GROUP BY lf.id
            ORDER BY lf.id
            """,
            int(cycle_db_id),
            [int(value) for value in feedback_ids],
        )

    if len(rows) != len(set(feedback_ids)):
        raise RuntimeError("VALIDATED_FEEDBACK_CHANGED")

    record_payloads: list[dict[str, Any]] = []
    synthetic = False
    timestamps: list[str] = []
    for row in rows:
        source_model_version = row["production_model_version"]
        if source_model_version and str(source_model_version) != production_model_version:
            raise RuntimeError("PRODUCTION_BASELINE_MISMATCH")
        feedback_id = str(row["feedback_id"])
        ticket_id = str(row["ticket_id"])
        is_accepted = str(row["accepted_or_corrected"]).upper() == "ACCEPTED"
        prediction = _prediction(row["production_prediction"], production_model_version)
        decision = _confirmed_decision(
            row["operator_confirmed_decision"], prediction, is_accepted
        )
        timestamp = _date_time(row["feedback_created_at"])
        timestamps.append(timestamp)
        synthetic = synthetic or bool(row["source_is_synthetic"]) or int(
            row["source_version_count"]
        ) == 0
        record_payloads.append(
            {
                "feedback_id": feedback_id,
                "ticket_id": ticket_id,
                "production_prediction": prediction,
                "operator_confirmed_decision": decision,
                "accepted_or_corrected": is_accepted,
                "validation_status": "VALIDATED",
                "feedback_created_at": timestamp,
            }
        )

    export = {
        "schema_version": "learning-feedback-export.v1",
        "cycle_id": cycle_id,
        "dataset_version": f"feedback-export-{_cycle_suffix(cycle_id)}",
        "production_model_version": production_model_version,
        "exported_at": max(timestamps),
        "record_count": len(record_payloads),
        "records": record_payloads,
        "synthetic": synthetic,
    }
    validate_document(export, "LearningFeedbackExport")
    export_bytes = _canonical_json(export)
    export_sha256 = _sha256(export_bytes)
    export_path = artifact_dir / "exports" / f"{_cycle_suffix(cycle_id)}.json"
    _write_atomically(export_path, export_bytes)

    request = {
        "schema_version": "candidate-dataset-build-request.v1",
        "cycle_id": cycle_id,
        "candidate_model_version": str(request_payload.get("candidate_model_version") or ""),
        "production_model_version": production_model_version,
        "feedback_ids": feedback_ids,
        "feedback_export_uri": _file_uri(export_path),
        "feedback_export_sha256": export_sha256,
        "frozen_evaluation_dataset_version": str(evaluation_dataset_version),
        "frozen_evaluation_ticket_ids": evaluation_ticket_ids,
    }
    validate_document(request, "CandidateDatasetBuildRequest")
    return request


async def build_candidate_dataset(
    pool: Any,
    request: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Resolve feedback IDs into a checksummed training artifact."""

    validate_document(request, "CandidateDatasetBuildRequest")
    export_path = Path(request["feedback_export_uri"].removeprefix("file://"))
    export_bytes = export_path.read_bytes()
    if _sha256(export_bytes) != request["feedback_export_sha256"]:
        raise RuntimeError("FEEDBACK_EXPORT_CHECKSUM_MISMATCH")
    feedback_export = json.loads(export_bytes)
    validate_document(feedback_export, "LearningFeedbackExport")
    if (
        feedback_export["cycle_id"] != request["cycle_id"]
        or feedback_export["production_model_version"]
        != request["production_model_version"]
        or sorted(request["feedback_ids"])
        != sorted(record["feedback_id"] for record in feedback_export["records"])
    ):
        raise RuntimeError("FEEDBACK_EXPORT_LINEAGE_MISMATCH")

    evaluation_ticket_ids = set(request["frozen_evaluation_ticket_ids"])
    feedback_ticket_ids = sorted(
        {record["ticket_id"] for record in feedback_export["records"]}
    )
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT t.id::text AS ticket_id,
                   t.original_text,
                   t.language,
                   COALESCE(
                       ARRAY_REMOVE(ARRAY_AGG(DISTINCT dtl.dataset_version), NULL),
                       ARRAY[]::text[]
                   ) AS source_dataset_versions,
                   COALESCE(BOOL_OR(dv.is_synthetic), FALSE) AS source_is_synthetic
            FROM tickets t
            LEFT JOIN dataset_ticket_links dtl ON dtl.ticket_id = t.id
            LEFT JOIN dataset_versions dv ON dv.dataset_version = dtl.dataset_version
            WHERE t.id = ANY($1::bigint[])
            GROUP BY t.id
            ORDER BY t.id
            """,
            [int(value) for value in feedback_ticket_ids],
        )

    tickets_by_id = {str(row["ticket_id"]): row for row in rows}
    if len(tickets_by_id) != len(feedback_ticket_ids):
        raise RuntimeError("SOURCE_TICKET_UNAVAILABLE")

    training_samples: list[dict[str, Any]] = []
    training_ticket_ids: list[str] = []
    source_dataset_versions: set[str] = set()
    synthetic = bool(feedback_export["synthetic"])
    for record in feedback_export["records"]:
        ticket_id = record["ticket_id"]
        if ticket_id in evaluation_ticket_ids:
            continue
        row = tickets_by_id[ticket_id]
        text = str(row["original_text"] or "").strip()
        if not text:
            raise RuntimeError("SOURCE_TICKET_TEXT_UNAVAILABLE")
        label = record["operator_confirmed_decision"].get("topic_id")
        if not label:
            raise RuntimeError("FEEDBACK_LABELS_UNAVAILABLE")
        training_samples.append(
            {
                "text": text,
                "label": str(label),
                "topic_id": str(label),
                "language": str(row["language"] or "UNKNOWN"),
            }
        )
        training_ticket_ids.append(ticket_id)
        row_versions = {str(value) for value in row["source_dataset_versions"]}
        source_dataset_versions.update(row_versions)
        synthetic = synthetic or bool(row["source_is_synthetic"]) or not row_versions

    if not training_samples:
        raise RuntimeError("INSUFFICIENT_CANDIDATE_DATASET")

    content = b"".join(_canonical_json(sample) + b"\n" for sample in training_samples)
    content_sha256 = _sha256(content)
    dataset_version = (
        f"candidate-dataset-{_cycle_suffix(request['cycle_id'])}-{content_sha256[:16]}"
    )
    dataset_path = artifact_dir / "datasets" / f"{dataset_version}.jsonl"
    _write_atomically(dataset_path, content)
    artifact_uri = _file_uri(dataset_path)

    manifest = {
        "schema_version": "candidate-dataset-manifest.v1",
        "dataset_version": dataset_version,
        "artifact_uri": artifact_uri,
        "content_sha256": content_sha256,
        "record_count": len(training_samples),
        "feedback_ids": [record["feedback_id"] for record in feedback_export["records"]],
        "feedback_export_uri": request["feedback_export_uri"],
        "feedback_export_sha256": request["feedback_export_sha256"],
        "candidate_model_version": request["candidate_model_version"],
        "production_model_version": request["production_model_version"],
        "frozen_evaluation_dataset_version": request[
            "frozen_evaluation_dataset_version"
        ],
        "frozen_evaluation_ticket_ids": sorted(evaluation_ticket_ids),
        "source_dataset_versions": sorted(source_dataset_versions),
        "training_ticket_ids": training_ticket_ids,
        "synthetic": synthetic,
        "created_at": feedback_export["exported_at"],
    }
    validate_document(manifest, "CandidateDatasetManifest")
    manifest_bytes = _canonical_json(manifest)
    manifest_path = artifact_dir / "datasets" / f"{dataset_version}.manifest.json"
    _write_atomically(manifest_path, manifest_bytes)
    return {
        "dataset_version": dataset_version,
        "artifact_uri": artifact_uri,
        "manifest_uri": _file_uri(manifest_path),
        "manifest_sha256": _sha256(manifest_bytes),
        "content_sha256": content_sha256,
        "record_count": len(training_samples),
        "feedback_ids": manifest["feedback_ids"],
        "training_ticket_ids": training_ticket_ids,
        "source_dataset_versions": sorted(source_dataset_versions),
        "synthetic": synthetic,
        "production_model_version": request["production_model_version"],
        "candidate_model_version": request["candidate_model_version"],
        "frozen_evaluation_dataset_version": request[
            "frozen_evaluation_dataset_version"
        ],
        "frozen_evaluation_ticket_ids": sorted(evaluation_ticket_ids),
    }
