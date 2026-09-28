#!/usr/bin/env python3
"""Build or verify a self-contained classifier candidate handoff bundle."""

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

from training.classifier_bundle import build_classifier_bundle, verify_classifier_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a new immutable bundle")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--baseline-report", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="check every bundle checksum and lineage field")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            manifest = build_classifier_bundle(args.dataset, args.model, args.baseline_report, args.output)
        else:
            manifest = verify_classifier_bundle(args.bundle)
    except (OSError, ValueError, KeyError, TypeError) as error:
        message = str(error) if type(error) is ValueError else "classifier bundle failed validation or file access"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"model_version": manifest.model_version, "status": manifest.status,
                      "evaluation_version": manifest.evaluation_version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
