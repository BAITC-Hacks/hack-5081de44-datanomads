#!/usr/bin/env python3
"""Write aggregate token-length evidence for a reviewed classifier package."""

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

from training.classifier_token_audit import audit_token_lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="reviewed dataset package")
    parser.add_argument("--tokenizer", type=Path, required=True, help="local tokenizer artifact directory")
    parser.add_argument("--output", type=Path, required=True, help="new aggregate report JSON path")
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("token audit report must be outside the immutable dataset package")
        report = audit_token_lengths(args.dataset, args.tokenizer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "token audit failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"dataset_version": report["dataset_version"], "report_version": report["report_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
