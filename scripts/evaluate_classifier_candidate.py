#!/usr/bin/env python3
"""Evaluate one local reviewed classifier against TF-IDF on frozen test."""

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

from training.classifier_candidate_eval import evaluate_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="reviewed dataset package")
    parser.add_argument("--model", type=Path, required=True, help="local classifier artifact directory")
    parser.add_argument("--baseline-report", type=Path, required=True, help="matching TF-IDF baseline JSON")
    parser.add_argument("--output", type=Path, required=True, help="new aggregate evaluation JSON path")
    args = parser.parse_args()
    try:
        if (args.output.resolve().is_relative_to(args.dataset.resolve()) or
                args.output.resolve().is_relative_to(args.model.resolve())):
            raise ValueError("evaluation report must be outside immutable dataset and model directories")
        report = evaluate_candidate(args.dataset, args.model, args.baseline_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "classifier candidate evaluation failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"model_version": report["model_version"], "decision": report["decision"],
                      "sample_count": report["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
