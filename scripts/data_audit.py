#!/usr/bin/env python3
"""Build a compact data-quality report from normalized UnifiedTicket JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def build_report(path: Path) -> Dict[str, Any]:
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
