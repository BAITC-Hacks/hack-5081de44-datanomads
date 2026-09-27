"""Source-row to UnifiedTicket normalization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Optional

from data.quarantine.records import QuarantineRecord
from data.schemas.taxonomy import (
    canonical_language,
    canonical_priority,
    canonical_region_id,
    canonical_source_system,
    canonical_status,
    canonical_topic_id,
)
from data.schemas.unified_ticket import SchemaValidationError, UnifiedTicket
from .pii import PIIReport, minimize_text, scan_pii


@dataclass(frozen=True)
class NormalizationResult:
    ticket: Optional[UnifiedTicket] = None
    quarantine: Optional[QuarantineRecord] = None

    @property
    def valid(self) -> bool:
        return self.ticket is not None


def _parse_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            raise SchemaValidationError("MISSING_REQUIRED_FIELD", f"{field} is required", field)
        parsed_text = text.replace("Z", "+00:00")
        # A few government exports use a space instead of T; fromisoformat
        # accepts both and keeps the implementation dependency-free.
        try:
            parsed = datetime.fromisoformat(parsed_text)
        except ValueError as exc:
            # Do not guess ambiguous day/month values.  They belong in the
            # quarantine so a source-specific parser can be configured later.
            raise SchemaValidationError("INVALID_DATE", f"invalid {field}", field) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _service_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower(), flags=re.IGNORECASE).strip("_")
    # Keep a stable, non-empty identifier even for Cyrillic-only service names.
    return slug or "service_other"


def _quarantine(
    source_system: str,
    row_number: int,
    row: Mapping[str, Any],
    error: SchemaValidationError,
) -> NormalizationResult:
    return NormalizationResult(
        quarantine=QuarantineRecord(
            source_system=source_system,
            row_number=row_number,
            reason=error.code,
            detail=str(error),
            field=error.field,
            row=row,
        )
    )


def normalize_row(
    row: Mapping[str, Any],
    *,
    source_system: str,
    row_number: int = 1,
) -> NormalizationResult:
    """Normalize one source-specific canonical row.

    ``row`` is expected to have canonical keys after a source importer has
    resolved aliases.  Extra source columns are ignored and never copied into
    the unified record.
    """

    canonical_source = canonical_source_system(source_system) or str(source_system).strip()
    if not isinstance(row, Mapping):
        return _quarantine(
            canonical_source,
            row_number,
            {"row": str(row)},
            SchemaValidationError("UNKNOWN_SCHEMA", "source row must be an object"),
        )
    required = {
        "external_ticket_id": row.get("external_ticket_id"),
        "region_id": row.get("region_id"),
        "created_at": row.get("created_at"),
        "original_text": row.get("original_text"),
    }
    for field_name, value in required.items():
        if value is None or not str(value).strip():
            return _quarantine(
                canonical_source,
                row_number,
                row,
                SchemaValidationError("MISSING_REQUIRED_FIELD", f"{field_name} is required", field_name),
            )

    if scan_pii(row.get("external_ticket_id")).detected:
        return _quarantine(
            canonical_source,
            row_number,
            row,
            SchemaValidationError("PII_REVIEW", "source identifier may contain PII", "external_ticket_id"),
        )

    try:
        region_id = canonical_region_id(row.get("region_id"))
        if not region_id:
            raise SchemaValidationError("INVALID_VALUE", "unknown region", "region_id")
        created_at = _parse_datetime(row.get("created_at"), "created_at")
        closed_at = _parse_datetime(row.get("closed_at"), "closed_at") if row.get("closed_at") else None
        deadline_at = _parse_datetime(row.get("deadline_at"), "deadline_at") if row.get("deadline_at") else None
        text, pii_report = minimize_text(row.get("original_text"))
        if not text:
            raise SchemaValidationError("MISSING_REQUIRED_FIELD", "original_text is empty", "original_text")
        language = canonical_language(row.get("language"), text)
        safe_text = {}
        for field_name in ("topic_raw", "service_raw", "district", "object", "channel", "resolution_text", "official_response"):
            value = row.get(field_name)
            safe_value, report = minimize_text(value) if value else ("", PIIReport())
            safe_text[field_name] = safe_value or None
            pii_report = PIIReport(
                tuple(dict.fromkeys(pii_report.categories + report.categories)),
                pii_report.redaction_count + report.redaction_count,
            )
        topic_raw = safe_text["topic_raw"] or "UNKNOWN"
        topic_id = canonical_topic_id(row.get("topic_id") or topic_raw)
        service_raw = safe_text["service_raw"]
        address = row.get("address")
        if address:
            # Exact addresses are not needed by the safe layer.  Keep an
            # explicit marker so downstream consumers know why it is absent.
            address = "[ADDRESS_REDACTED]"
        ticket = UnifiedTicket.from_mapping(
            {
                "external_ticket_id": str(row.get("external_ticket_id")).strip(),
                "source_system": canonical_source,
                "region_id": region_id,
                "created_at": created_at,
                "original_text": text,
                "language": language,
                "topic_raw": topic_raw,
                "topic_id": topic_id,
                "service_raw": service_raw,
                "service_id": row.get("service_id") or _service_id(service_raw),
                "priority": canonical_priority(row.get("priority")),
                "status": canonical_status(row.get("status")),
                "district": safe_text["district"],
                "address": address,
                # Precise coordinates can identify a household; this layer
                # only needs the region and optional district.
                "coordinates": None,
                "object": safe_text["object"],
                "channel": safe_text["channel"],
                "closed_at": closed_at,
                "deadline_at": deadline_at,
                "resolution_text": safe_text["resolution_text"],
                "official_response": safe_text["official_response"],
                # Attachments are metadata-only in the unified layer.  Their
                # contents must never be copied to ML/analytics-safe data.
                "attachments": (),
                "assignment_history": (),
                "pulse_prediction": row.get("pulse_prediction"),
                "operator_confirmed_decision": row.get("operator_confirmed_decision"),
                "model_versions": row.get("model_versions") or {},
                "needs_review": bool(row.get("needs_review", False)),
                "embedding_ref": row.get("embedding_ref"),
                "duplicate_feedback": row.get("duplicate_feedback"),
                "repeat_feedback": row.get("repeat_feedback"),
                "text_redaction_count": pii_report.redaction_count,
            }
        )
        # A residual match means the source contains a PII form this safe
        # masker does not understand.  Keep the row visible in quarantine for
        # a source-specific rule rather than silently leaking it downstream.
        residual_categories = [category for value in (
            ticket.original_text, ticket.topic_raw, ticket.service_raw, ticket.district,
            ticket.object, ticket.channel, ticket.resolution_text, ticket.official_response,
        ) for category in scan_pii(value).categories]
        if residual_categories:
            raise SchemaValidationError(
                "PII_REVIEW",
                "PII remains after minimization: " + ", ".join(dict.fromkeys(residual_categories)),
            )
        return NormalizationResult(ticket=ticket)
    except SchemaValidationError as error:
        return _quarantine(canonical_source, row_number, row, error)
    except (TypeError, ValueError) as error:
        return _quarantine(
            canonical_source,
            row_number,
            row,
            SchemaValidationError("INVALID_VALUE", "invalid source value"),
        )


__all__ = ["NormalizationResult", "normalize_row"]
