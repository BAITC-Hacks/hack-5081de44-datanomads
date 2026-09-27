#!/usr/bin/env python3
"""Recommend one of four comparable validation-only classifier artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ml-service") not in sys.path:
    sys.path.insert(0, str(ROOT / "ml-service"))

from training.classifier_input_strategy_comparison import compare_input_strategies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="validation-only classifier artifact; pass once per strategy")
    parser.add_argument("--output", type=Path, help="new JSON report path")
    args = parser.parse_args()
    try:
        if args.output is not None and any(
                args.output.resolve().is_relative_to(artifact.resolve()) for artifact in args.artifact):
            raise ValueError("comparison report must be outside immutable model artifacts")
        report = compare_input_strategies(args.artifact)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__,
                          "message": "classifier input-strategy comparison failed"}), file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
