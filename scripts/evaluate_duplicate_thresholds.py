#!/usr/bin/env python3
"""Evaluate a provisional duplicate threshold on reviewed retrieval pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.duplicate_thresholds import evaluate_duplicate_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="local E5-compatible model directory")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if (output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(args.model.resolve()) or
                output in (args.policy.resolve(),)):
            raise ValueError("duplicate threshold report must be outside immutable inputs")
        report = evaluate_duplicate_thresholds(args.dataset, args.model, args.policy,
                                               args.model_version, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "error_code": "DUPLICATE_THRESHOLD_EVALUATION_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "threshold_status": report["threshold_status"],
                      "validation_pairs": sum(report["validation"][key] for key in
                                              ("duplicate_count", "nonduplicate_count"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
