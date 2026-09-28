"""Typed quarantine records with value-free row summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from data.normalization.pii import minimize_mapping, minimize_text


QUARANTINE_REASONS = {
    "BAD_CSV_STRUCTURE",
    "INVALID_DATE",
    "MISSING_REQUIRED_FIELD",
    "UNKNOWN_SCHEMA",
    "PII_REVIEW",
    "INVALID_VALUE",
}


@dataclass(frozen=True)
class QuarantineRecord:
    source_system: str
    row_number: int
    reason: str
    detail: str
    row: Mapping[str, Any]
    field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reason not in QUARANTINE_REASONS:
            raise ValueError(f"unsupported quarantine reason: {self.reason}")
        if self.row_number < 1:
            raise ValueError("row_number must be positive")

    def to_dict(self) -> Mapping[str, Any]:
        # Source headers can themselves contain identifiers.  Keep only counts
        # so an unknown PII pattern cannot escape through a quarantine snapshot.
        _, pii_report = minimize_mapping(self.row)
        safe_detail, _ = minimize_text(self.detail)
        return {
            "source_system": self.source_system,
            "row_number": self.row_number,
            "reason": self.reason,
            "detail": safe_detail,
            "field": self.field,
            "row": {
                "column_count": len(self.row),
                "nonempty_count": sum(value is not None and bool(str(value).strip()) for value in self.row.values()),
            },
            "pii_categories": list(pii_report.categories),
        }


def write_jsonl(records: list[QuarantineRecord], path: Path) -> None:
    """Write a deterministic, inspectable quarantine artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["QUARANTINE_REASONS", "QuarantineRecord", "write_jsonl"]
