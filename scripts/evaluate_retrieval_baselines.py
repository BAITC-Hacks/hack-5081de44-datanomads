#!/usr/bin/env python3
"""Write lexical and optional local E5 retrieval baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.retrieval_baselines import evaluate_retrieval_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e5-model", type=Path, help="local pretrained multilingual-e5-base directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("retrieval report must be outside the immutable dataset package")
        report = evaluate_retrieval_baselines(args.dataset, args.e5_model, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "retrieval evaluation failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"dataset_version": report["dataset_version"], "e5_model_status": report["e5_model_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
