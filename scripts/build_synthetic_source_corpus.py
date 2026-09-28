#!/usr/bin/env python3
"""Build an untrainable, PII-minimized UnifiedTicket package from synthetic sources."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.importers import get_importer
from data.normalization.pii import scan_pii
from data.schemas.unified_ticket import SCHEMA_VERSION, UnifiedTicket
from scripts.audit_synthetic_sources import DEFAULT_MANIFEST, build_report
from scripts.strict_json import unique_object


DEFAULT_QUALITY_REPORT = ROOT / "data/reports/synthetic_source_quality_v1.json"
DEFAULT_OUTPUT = ROOT / "data/processed/synthetic_source_v1"


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_corpus(raw_manifest_path: Path, quality_report_path: Path, output_dir: Path) -> dict:
    quality_bytes = quality_report_path.read_bytes()
    quality = json.loads(quality_bytes.decode("utf-8"), object_pairs_hook=unique_object)
    audited = build_report(raw_manifest_path)
    expected_bytes = (json.dumps(audited, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if quality != audited or quality_bytes != expected_bytes:
        raise ValueError("quality report differs from audited synthetic raw fixtures")

    root = raw_manifest_path.parent
    tickets = []
    quarantine = []
    seen_ids = set()
    for source in quality["sources"]:
        source_system = source["source_system"]
        importer = get_importer(source_system)
        for file in source["files"].values():
            result = importer.import_file(root / file["path"])
            if (result.valid_count != file["valid_record_count"] or
                    result.quarantine_count != file["quarantine_count"]):
                raise ValueError("source import differs from quality report")
            for ticket in result.tickets:
                key = (source_system, ticket.external_ticket_id)
                if key in seen_ids:
                    raise ValueError("source ticket ID repeats across fixture files")
                seen_ids.add(key)
                row = ticket.to_dict()
                if (row["schema_version"] != SCHEMA_VERSION or
                        UnifiedTicket.from_mapping(row).to_dict() != row or
                        scan_pii(row["original_text"]).detected):
                    raise ValueError("normalized ticket failed schema or PII validation")
                tickets.append(row)
            quarantine.extend(record.to_dict() for record in result.quarantine)

    if (len(tickets) != quality["totals"]["valid_ticket_count"] or
            len(quarantine) != quality["totals"]["quarantine_count"]):
        raise ValueError("normalized counts differ from quality report")
    tickets.sort(key=lambda row: (row["source_system"], row["external_ticket_id"]))
    quarantine.sort(key=lambda row: (row["source_system"], row["reason"], row["row_number"]))
    files = {
        "tickets.jsonl": _jsonl(tickets),
        "quarantine.jsonl": _jsonl(quarantine),
        "audit.json": quality_bytes,
    }
    manifest = {
        "dataset_version": "synthetic-source-normalized-v1",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "profile_status": "SYNTHETIC_TEST_ONLY",
        "approved_for_training": False,
        "purpose": "source normalization and privacy validation only; not customer appeal data",
        "raw_fixture_manifest_sha256": quality["fixture_manifest_sha256"],
        "source_systems": sorted(source["source_system"] for source in quality["sources"]),
        "record_count": len(tickets),
        "quarantine_count": len(quarantine),
        "languages": dict(sorted(Counter(row["language"] for row in tickets).items())),
        "topics": dict(sorted(Counter(row["topic_id"] for row in tickets).items())),
        "regions": dict(sorted(Counter(row["region_id"] for row in tickets).items())),
        "files": {name: {"sha256": _sha256(content), "bytes": len(content)} for name, content in files.items()},
        "content_sha256": _sha256(files["tickets.jsonl"]),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(content)
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = build_corpus(args.raw_manifest, args.quality_report, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "synthetic source corpus build failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"dataset_version": manifest["dataset_version"], "record_count": manifest["record_count"],
                      "quarantine_count": manifest["quarantine_count"], "approved_for_training": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
