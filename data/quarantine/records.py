"""Typed quarantine records with PII-safe row snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from data.normalization.pii import minimize_mapping


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
        safe_row, pii_report = minimize_mapping(self.row)
        return {
            "source_system": self.source_system,
            "row_number": self.row_number,
            "reason": self.reason,
            "detail": self.detail,
            "field": self.field,
            "row": safe_row,
            "pii_categories": list(pii_report.categories),
        }


def write_jsonl(records: list[QuarantineRecord], path: Path) -> None:
    """Write a deterministic, inspectable quarantine artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["QUARANTINE_REASONS", "QuarantineRecord", "write_jsonl"]
