#!/usr/bin/env python3
"""Build an offline candidate dataset from a validated learning-feedback export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.feedback_dataset import build_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--frozen-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/feedback")
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--production-model-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--min-feedback-count", type=int, required=True)
    args = parser.parse_args()
    try:
        result = build_candidate(
            args.feedback, args.frozen_from, args.output_root,
            cycle_id=args.cycle_id,
            production_model_version=args.production_model_version,
            dataset_version=args.dataset_version,
            min_feedback_count=args.min_feedback_count,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "feedback candidate build failed"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
