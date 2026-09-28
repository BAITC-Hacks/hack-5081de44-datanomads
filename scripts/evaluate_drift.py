#!/usr/bin/env python3
"""Compare two PII-free drift snapshots under a versioned policy."""

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

from training.drift import evaluate_drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() in {args.baseline.resolve(), args.recent.resolve(), args.policy.resolve()}:
            raise ValueError("drift report must not overwrite its inputs")
        report = evaluate_drift(args.baseline, args.recent, args.policy)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "DRIFT_EVALUATION_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "drifted_signals": report["drifted_signals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
