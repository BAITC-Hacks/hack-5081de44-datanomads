#!/usr/bin/env python3
"""Build or verify an offline E5-compatible retrieval candidate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.embedder_candidate import train_embedder_candidate, verify_embedder_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--base-model", type=Path, required=True)
    build.add_argument("--base-model-id", required=True)
    build.add_argument("--baseline-report", type=Path, required=True)
    build.add_argument("--model-version", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--epochs", type=int, default=3)
    build.add_argument("--batch-size", type=int, default=2)
    build.add_argument("--learning-rate", type=float, default=2e-5)
    build.add_argument("--seed", type=int, default=109)
    verify = commands.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = train_embedder_candidate(args.dataset, args.base_model, args.baseline_report,
                                              args.output, model_version=args.model_version,
                                              base_model_id=args.base_model_id, epochs=args.epochs,
                                              batch_size=args.batch_size, learning_rate=args.learning_rate,
                                              seed=args.seed)
        else:
            manifest = verify_embedder_candidate(args.artifact)
            result = {"status": "VERIFIED", "model_version": manifest.model_version,
                      "artifact_checksum": manifest.artifact_checksum}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error_code": "EMBEDDER_CANDIDATE_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
