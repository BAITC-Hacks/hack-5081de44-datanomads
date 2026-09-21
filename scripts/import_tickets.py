#!/usr/bin/env python3
"""Import a source export into PII-safe UnifiedTicket JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    args = parser.parse_args()

    importer = get_importer(args.source)
    result = importer.import_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for ticket in result.tickets:
            handle.write(json.dumps(ticket.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    write_jsonl(result.quarantine, args.quarantine)
    print(json.dumps(result.to_summary(), ensure_ascii=False, sort_keys=True))
    return 0 if not result.quarantine else 2


if __name__ == "__main__":
    raise SystemExit(main())
