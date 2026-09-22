#!/usr/bin/env python3
"""Import a source export into PII-safe UnifiedTicket JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.importers import get_importer
from data.quarantine import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source system name, e.g. iKOMEK109")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="normalized JSONL destination")
    parser.add_argument("--quarantine", type=Path, required=True, help="quarantine JSONL destination")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PULSE_CORE_URL"),
        help="Core URL for durable PostgreSQL/Qdrant import (or PULSE_CORE_URL)",
    )
    parser.add_argument("--api-role", default=os.environ.get("PULSE_ROLE", "ADMIN"))
    parser.add_argument("--api-user", default=os.environ.get("PULSE_USER", "data-import"))
    parser.add_argument("--dataset-version")
    parser.add_argument("--manifest-uri")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    importer = get_importer(args.source)
    result = importer.import_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for ticket in result.tickets:
            handle.write(json.dumps(ticket.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    write_jsonl(result.quarantine, args.quarantine)
    summary = dict(result.to_summary())
    if args.api_url:
        payload = {
            "source_system": result.source_system,
            "source_uri": str(args.input),
            "dataset_version": args.dataset_version,
            "manifest_uri": args.manifest_uri,
            "manifest_sha256": args.manifest_sha256,
            "is_synthetic": args.synthetic,
            "tickets": [ticket.to_dict() for ticket in result.tickets],
            "quarantine": [record.to_dict() for record in result.quarantine],
        }
        endpoint = args.api_url.rstrip("/") + "/api/v1/import"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Pulse-Role": args.api_role,
                "X-User-Id": args.api_user,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                imported = json.load(response)
        except Exception as error:
            print(json.dumps({"error": f"Core import failed: {error}", **summary}, ensure_ascii=False))
            return 1
        summary["persistence"] = imported
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not result.quarantine else 2


if __name__ == "__main__":
    raise SystemExit(main())
