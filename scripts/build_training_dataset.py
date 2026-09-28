#!/usr/bin/env python3
"""Build a versioned synthetic training package from human-reviewed inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.dataset_builder import build_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, required=True, help="approved classifier JSONL")
    parser.add_argument("--classifier-candidates", type=Path, required=True, help="original pending classifier candidates JSONL")
    parser.add_argument("--classifier-review", type=Path, required=True, help="complete classifier review queue JSONL")
    parser.add_argument("--retrieval", type=Path, required=True, help="approved relation-pair JSONL")
    parser.add_argument("--retrieval-review", type=Path, required=True, help="complete relation review queue JSONL")
    parser.add_argument("--scenario-source", type=Path, required=True, help="versioned source scenarios")
    parser.add_argument("--relation-source", type=Path, required=True, help="versioned relation evidence")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--frozen-evaluation-version", required=True)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--frozen-from", type=Path, help="existing immutable package for candidate retraining")
    args = parser.parse_args()
    try:
        manifest = build_package(
            args.classifier, args.retrieval, args.scenario_source, args.relation_source,
            args.output_root, args.dataset_version,
            args.frozen_evaluation_version, args.seed,
            classifier_candidates=args.classifier_candidates,
            classifier_review=args.classifier_review,
            retrieval_review=args.retrieval_review,
            frozen_from=args.frozen_from,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "dataset build failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "dataset_version": manifest.dataset_version,
        "record_count": manifest.record_count,
        "retrieval_pair_count": manifest.retrieval_pair_count,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "content_sha256": manifest.content_sha256,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
