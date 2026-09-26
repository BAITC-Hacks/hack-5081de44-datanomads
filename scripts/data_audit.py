#!/usr/bin/env python3
"""Build a quality report from UnifiedTicket JSONL or a raw 109 CSV export."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_COLUMNS = {
    "application_number", "creation_date", "closing_date", "region", "district",
    "street", "full_name", "applicant_number", "application_type",
    "submittal_channel", "category", "service", "contractor", "com_exp",
    "result", "status", "status_1", "operator",
}


def _rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_parse_error": f"line {line_number}: {exc.msg}"}
                continue
            if isinstance(row, Mapping):
                yield row
            else:
                yield {"_schema_error": f"line {line_number}: expected object"}


def build_normalized_report(path: Path) -> Dict[str, Any]:
    records = list(_rows(path))
    valid = [row for row in records if "external_ticket_id" in row and "_parse_error" not in row and "_schema_error" not in row]
    created = []
    invalid_dates = 0
    for row in valid:
        try:
            created.append(datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")))
        except (KeyError, TypeError, ValueError):
            invalid_dates += 1
    ids = [str(row["external_ticket_id"]) for row in valid]
    report = {
        "path": str(path),
        "record_count": len(records),
        "valid_record_count": len(valid),
        "parse_or_schema_error_count": len(records) - len(valid),
        "duplicate_external_id_count": len(ids) - len(set(ids)),
        "invalid_date_count": invalid_dates,
        "time_range": {
            "min": min(created).isoformat() if created else None,
            "max": max(created).isoformat() if created else None,
        },
        "by_region": dict(sorted(Counter(str(row.get("region_id", "UNKNOWN")) for row in valid).items())),
        "by_source_system": dict(sorted(Counter(str(row.get("source_system", "UNKNOWN")) for row in valid).items())),
        "by_topic": dict(sorted(Counter(str(row.get("topic_id", "unknown")) for row in valid).items())),
        "by_language": dict(sorted(Counter(str(row.get("language", "UNKNOWN")) for row in valid).items())),
        "by_status": dict(sorted(Counter(str(row.get("status", "UNKNOWN")) for row in valid).items())),
    }
    return report


def build_109_csv_report(path: Path) -> Dict[str, Any]:
    """Count only structural properties; never serialize source field values."""
    required = {"application_number", "creation_date", "category", "service", "com_exp"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not required.issubset(headers):
            raise ValueError("CSV is missing required 109 export columns")

        nonempty = Counter()
        categories = Counter()
        years = Counter()
        seen_ids = set()
        row_count = invalid_date_count = duplicate_id_count = malformed_row_count = 0
        missing_id_count = long_region_value_count = 0
        first_date = last_date = None
        for row in reader:
            row_count += 1
            if None in row or any(value is None for value in row.values()):
                malformed_row_count += 1
                continue
            for column in headers:
                if column not in AUDIT_COLUMNS:
                    continue
                if row[column].strip():
                    nonempty[column] += 1
            identifier = row["application_number"].strip()
            if identifier:
                if identifier in seen_ids:
                    duplicate_id_count += 1
                seen_ids.add(identifier)
            else:
                missing_id_count += 1
            try:
                created = datetime.strptime(row["creation_date"].strip(), "%d.%m.%Y %H:%M:%S")
            except ValueError:
                invalid_date_count += 1
            else:
                first_date = created if first_date is None else min(first_date, created)
                last_date = created if last_date is None else max(last_date, created)
                years[str(created.year)] += 1
            category = row["category"].strip()
            if category:
                categories[category] += 1
            if "region" in row and len(row["region"].strip()) > 120:
                long_region_value_count += 1

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "format": "raw_109_csv",
        "sha256": digest.hexdigest(),
        "record_count": row_count,
        "malformed_row_count": malformed_row_count,
        "missing_external_id_count": missing_id_count,
        "invalid_date_count": invalid_date_count,
        "duplicate_external_id_count": duplicate_id_count,
        "time_range": {
            "min": first_date.isoformat() if first_date else None,
            "max": last_date.isoformat() if last_date else None,
        },
        "by_year": dict(sorted(years.items())),
        "nonempty_by_column": {
            column: nonempty[column] for column in headers if column in AUDIT_COLUMNS
        },
        "unknown_column_count": sum(column not in AUDIT_COLUMNS for column in headers),
        "category_count": len(categories),
        "largest_category_count": max(categories.values(), default=0),
        "region_values_over_120_chars": long_region_value_count,
        "has_original_text_column": "original_text" in headers,
    }


def build_report(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".csv":
        return build_109_csv_report(path)
    return build_normalized_report(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.input)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
