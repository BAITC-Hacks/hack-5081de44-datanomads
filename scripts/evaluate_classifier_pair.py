#!/usr/bin/env python3
"""Compare two local classifiers on the same frozen test and regression policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ml-service") not in sys.path:
    sys.path.insert(0, str(ROOT / "ml-service"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.classifier_pair_eval import compare_classifiers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.output.resolve().is_relative_to(args.dataset.resolve()) or
                args.output.resolve().is_relative_to(args.production.resolve()) or
                args.output.resolve().is_relative_to(args.candidate.resolve())):
            raise ValueError("pair evaluation report must be outside immutable inputs")
        report = compare_classifiers(args.dataset, args.production, args.candidate, args.policy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "classifier pair evaluation failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"production_model_version": report["production"]["model_version"],
                      "candidate_model_version": report["candidate"]["model_version"],
                      "decision": report["decision"], "sample_count": report["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
