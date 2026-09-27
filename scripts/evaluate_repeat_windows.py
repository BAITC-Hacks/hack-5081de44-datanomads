#!/usr/bin/env python3
"""Evaluate REPEAT time windows on approved retrieval and temporal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.repeat_windows import evaluate_repeat_windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--temporal-source", type=Path, required=True)
    parser.add_argument("--temporal-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if (output.is_relative_to(args.dataset.resolve()) or
                output in (args.temporal_source.resolve(), args.temporal_evidence.resolve(), args.policy.resolve())):
            raise ValueError("repeat report must be outside immutable inputs")
        report = evaluate_repeat_windows(args.dataset, args.temporal_source,
                                         args.temporal_evidence, args.policy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "error_code": "REPEAT_WINDOW_EVALUATION_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"],
                      "selected_window_days": report["validation"]["selected_window_days"],
                      "runtime_window_status": report["runtime_window_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
