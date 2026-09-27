#!/usr/bin/env python3
"""Create an unreviewed, detector-blind incident registry template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_spike_incidents import _unique_object
from scripts.export_spike_review import build_report


def prepare_template(source_csv: Path, candidate_report: Path, output: Path) -> dict:
    if output.resolve() in {source_csv.resolve(), candidate_report.resolve()}:
        raise ValueError("template output must differ from its inputs")
    candidate_bytes = candidate_report.read_bytes()
    candidate = json.loads(candidate_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    if candidate != build_report(source_csv):
        raise ValueError("candidate report differs from regenerated source evaluation")
    dates = candidate["evaluated_dates"]
    if not dates:
        raise ValueError("candidate report has no evaluated dates")
    template = {
        "schema_version": "spike-incident-registry.v1",
        "source_sha256": candidate["source_sha256"],
        "scope": candidate["scope"],
        "unit": candidate["unit"],
        "evaluated_date_sha256": candidate["evaluated_date_sha256"],
        "candidate_report_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "coverage_assertion": {
            "all_evaluated_dates_reviewed": False,
            "independent_of_detector": False,
            "reviewer_id": None,
            "reviewed_at": None,
        },
        "daily_labels": [
            {"date": day, "incident_ids": None, "review_status": "PENDING"}
            for day in dates
        ],
        "incidents": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        template = prepare_template(args.source_csv, args.candidate_report, args.output)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "incident review template failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PENDING", "daily_label_count": len(template["daily_labels"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
