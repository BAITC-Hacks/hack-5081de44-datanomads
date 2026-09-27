#!/usr/bin/env python3
"""Train a local classifier candidate from a verified feedback dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.feedback_trainer import CandidateTrainingError, train_feedback_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen-from", type=Path, required=True)
    parser.add_argument("--production-model", type=Path, required=True)
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=109)
    args = parser.parse_args()
    try:
        result = train_feedback_candidate(
            args.dataset, args.frozen_from, args.production_model, args.output,
            candidate_model_version=args.candidate_model_version, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
        )
    except (CandidateTrainingError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        code = str(error) if isinstance(error, CandidateTrainingError) else "TRAINING_FAILED"
        print(json.dumps({"status": "FAILED", "error_code": code,
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
