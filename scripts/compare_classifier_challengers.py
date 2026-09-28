#!/usr/bin/env python3
"""Compare several candidate reports against one champion and shared slices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.challenger_eval import compare_challengers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-report", type=Path, action="append", required=True)
    parser.add_argument("--shadow-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.offline_report) != len(args.shadow_report) or len(args.offline_report) < 2:
            raise ValueError("provide at least two aligned offline and shadow report pairs")
        inputs = list(zip(args.offline_report, args.shadow_report))
        if args.output.resolve() in {path.resolve() for pair in inputs for path in pair}:
            raise ValueError("challenger report must not overwrite its inputs")
        report = compare_challengers(inputs)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "CHALLENGER_COMPARISON_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"],
                      "candidate_count": len(report["candidates"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
