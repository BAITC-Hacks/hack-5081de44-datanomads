#!/usr/bin/env python3
"""Evaluate paired production/candidate shadow predictions without ticket text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.shadow_eval import evaluate_shadow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--production-model-version", required=True)
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() in (args.input.resolve(), args.policy.resolve()):
            raise ValueError("shadow report must not overwrite its inputs")
        report = evaluate_shadow(args.input, args.policy, cycle_id=args.cycle_id,
                                 production_model_version=args.production_model_version,
                                 candidate_model_version=args.candidate_model_version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "error_code": "SHADOW_EVALUATION_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "sample_count": report["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
