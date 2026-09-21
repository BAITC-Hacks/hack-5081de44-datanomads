"""UnifiedTicket schema and validation.

This is the safe boundary between source-specific rows and PostgreSQL.  The
schema intentionally contains no raw source payload and no dedicated PII
fields.  ``original_text`` means the operator-readable, PII-minimized text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import re
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = "unified-ticket.v1"
UNKNOWN = "UNKNOWN"
ALLOWED_LANGUAGES = {"RU", "KZ", "OTHER", UNKNOWN}
ALLOWED_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL", "OTHER"}
ALLOWED_STATUSES = {
    "OPEN",
    "IN_PROGRESS",
    "RESOLVED",
    "CLOSED",
    "CANCELLED",
    "OTHER",
    UNKNOWN,
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


class SchemaValidationError(ValueError):
    """A row cannot cross the UnifiedTicket contract boundary."""

    def __init__(self, code: str, message: str, field: Optional[str] = None):
        self.code = code
        self.field = field
        super().__init__(message)


def _optional_text(value: Any, max_length: int = 10_000) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length:
        raise SchemaValidationError("INVALID_VALUE", f"value exceeds {max_length} characters")
    return text


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            raise SchemaValidationError("MISSING_REQUIRED_FIELD", "created_at is required", "created_at")
        parsed_text = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(parsed_text)
        except ValueError as exc:
            raise SchemaValidationError("INVALID_DATE", f"invalid created_at: {text}", "created_at") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class UnifiedTicket:
    """Versioned, normalized and PII-minimized ticket record.

    Required fields follow the master-plan contract.  Optional fields are
    represented as ``None`` instead of fabricated values when a source does
    not provide them.
    """

    external_ticket_id: str
    source_system: str
    region_id: str
    created_at: datetime
    original_text: str
    language: str = UNKNOWN
    topic_raw: str = UNKNOWN
    topic_id: str = "unknown"
    service_raw: Optional[str] = None
    service_id: Optional[str] = None
    priority: Optional[str] = None
    status: str = UNKNOWN
    district: Optional[str] = None
    address: Optional[str] = None
    coordinates: Optional[Mapping[str, float]] = None
    object: Optional[str] = None
    channel: Optional[str] = None
    closed_at: Optional[datetime] = None
    deadline_at: Optional[datetime] = None
    resolution_text: Optional[str] = None
    official_response: Optional[str] = None
    attachments: Tuple[str, ...] = ()
    assignment_history: Tuple[Mapping[str, Any], ...] = ()
    pulse_prediction: Optional[Mapping[str, Any]] = None
    operator_confirmed_decision: Optional[Mapping[str, Any]] = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    needs_review: bool = False
    embedding_ref: Optional[str] = None
    duplicate_feedback: Optional[str] = None
    repeat_feedback: Optional[str] = None
    created_in_pulse_at: Optional[datetime] = None
    updated_in_pulse_at: Optional[datetime] = None
    text_redaction_count: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("external_ticket_id", "source_system", "region_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError("MISSING_REQUIRED_FIELD", f"{field_name} is required", field_name)
            if not _ID_RE.fullmatch(value):
                raise SchemaValidationError("INVALID_VALUE", f"invalid {field_name}", field_name)
        if not isinstance(self.original_text, str) or not self.original_text.strip():
            raise SchemaValidationError("MISSING_REQUIRED_FIELD", "original_text is required", "original_text")
        if len(self.original_text) > 100_000:
            raise SchemaValidationError("INVALID_VALUE", "original_text is too long", "original_text")
        for field_name in ("created_at", "closed_at", "deadline_at", "created_in_pulse_at", "updated_in_pulse_at"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, datetime):
                raise SchemaValidationError("INVALID_DATE", f"{field_name} must be datetime", field_name)
            if value.tzinfo is None:
                object.__setattr__(self, field_name, value.replace(tzinfo=timezone.utc))
            else:
                object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        language = str(self.language or UNKNOWN).upper()
        if language not in ALLOWED_LANGUAGES:
            raise SchemaValidationError("INVALID_VALUE", f"unsupported language: {language}", "language")
        object.__setattr__(self, "language", language)
        if self.priority is not None and self.priority not in ALLOWED_PRIORITIES:
            raise SchemaValidationError("INVALID_VALUE", f"unsupported priority: {self.priority}", "priority")
        if self.status not in ALLOWED_STATUSES:
            raise SchemaValidationError("INVALID_VALUE", f"unsupported status: {self.status}", "status")
        if self.text_redaction_count < 0:
            raise SchemaValidationError("INVALID_VALUE", "text_redaction_count cannot be negative")
        if self.duplicate_feedback not in {None, "CONFIRMED", "REJECTED"}:
            raise SchemaValidationError("INVALID_VALUE", "invalid duplicate_feedback")
        if self.repeat_feedback not in {None, "CONFIRMED", "REJECTED"}:
            raise SchemaValidationError("INVALID_VALUE", "invalid repeat_feedback")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "UnifiedTicket":
        """Build a ticket from already-normalized values.

        Importers should use ``data.normalization.normalize_row`` first.  This
        constructor still accepts ISO date strings for direct API and fixture
        use and raises typed errors for quarantine handling.
        """

        if not isinstance(values, Mapping):
            raise SchemaValidationError("UNKNOWN_SCHEMA", "ticket must be an object")
        data = dict(values)
        allowed_fields = {item.name for item in fields(cls)}
        unknown_fields = sorted(set(data) - allowed_fields)
        if unknown_fields:
            raise SchemaValidationError(
                "UNKNOWN_SCHEMA",
                "unsupported UnifiedTicket fields: " + ", ".join(unknown_fields),
            )
        required = ("external_ticket_id", "source_system", "region_id", "created_at", "original_text")
        for field_name in required:
            if data.get(field_name) is None or not str(data.get(field_name)).strip():
                raise SchemaValidationError("MISSING_REQUIRED_FIELD", f"{field_name} is required", field_name)
        for field_name in ("created_at", "closed_at", "deadline_at", "created_in_pulse_at", "updated_in_pulse_at"):
            if data.get(field_name) is not None:
                data[field_name] = _parse_datetime(data[field_name])
        data["external_ticket_id"] = str(data["external_ticket_id"]).strip()
        data["source_system"] = str(data["source_system"]).strip()
        data["region_id"] = str(data["region_id"]).strip()
        data["original_text"] = str(data["original_text"]).strip()
        data["topic_raw"] = str(data.get("topic_raw") or UNKNOWN).strip() or UNKNOWN
        data["topic_id"] = str(data.get("topic_id") or "unknown").strip() or "unknown"
        data["language"] = str(data.get("language") or UNKNOWN).upper()
        data["status"] = str(data.get("status") or UNKNOWN).upper()
        if data.get("priority") is not None:
            data["priority"] = str(data["priority"]).upper()
        for field_name in (
            "service_raw",
            "service_id",
            "district",
            "address",
            "object",
            "channel",
            "resolution_text",
            "official_response",
            "embedding_ref",
        ):
            data[field_name] = _optional_text(data.get(field_name))
        if data.get("attachments") is None:
            data["attachments"] = ()
        elif isinstance(data["attachments"], list):
            data["attachments"] = tuple(str(item) for item in data["attachments"])
        if data.get("assignment_history") is None:
            data["assignment_history"] = ()
        elif isinstance(data["assignment_history"], list):
            data["assignment_history"] = tuple(data["assignment_history"])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""

        return _json_safe(asdict(self))


__all__ = ["SCHEMA_VERSION", "SchemaValidationError", "UnifiedTicket"]
