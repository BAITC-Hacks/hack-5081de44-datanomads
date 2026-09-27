#!/usr/bin/env python3
"""Audit every synthetic raw fixture through its source-specific importer."""

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

from data.schemas.taxonomy import CANONICAL_SOURCE_SYSTEMS
from scripts.data_audit import build_source_report
from scripts.strict_json import unique_object


DEFAULT_MANIFEST = ROOT / "data/synthetic_raw/v1/manifest.json"
EXPECTED_RESULTS = {
    "primary": (4, {"INVALID_DATE": 1, "INVALID_VALUE": 1, "MISSING_REQUIRED_FIELD": 1}),
    "alternate": (1, {}),
    "malformed": (0, {"BAD_CSV_STRUCTURE": 1}),
    "unknown": (0, {"UNKNOWN_SCHEMA": 1}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (not isinstance(manifest, dict) or manifest.get("fixture_version") != "synthetic-sources.v1" or
            manifest.get("synthetic") is not True or not isinstance(manifest.get("sources"), list)):
        raise ValueError("unsupported synthetic fixture manifest")
    sources = manifest["sources"]
    if len(sources) != len(CANONICAL_SOURCE_SYSTEMS):
        raise ValueError("synthetic fixture manifest lacks source profiles")

    root = manifest_path.parent.resolve()
    seen_sources = set()
    totals = Counter()
    quarantine_reasons = Counter()
    distributions = {name: Counter() for name in ("by_region", "by_source_system", "by_topic", "by_language", "by_status")}
    field_presence = Counter()
    first_date = last_date = None
    source_reports = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid synthetic source profile")
        source_system = source.get("source_system")
        if (not isinstance(source_system, str) or source_system not in CANONICAL_SOURCE_SYSTEMS or
                source_system in seen_sources or
                source.get("profile_status") != "SYNTHETIC_TEST_ONLY" or
                not isinstance(source.get("files"), dict) or set(source["files"]) != set(EXPECTED_RESULTS) or
                not isinstance(source.get("schema_fingerprints"), dict)):
            raise ValueError("invalid synthetic source profile")
        seen_sources.add(source_system)
        files = {}
        for kind, expected in EXPECTED_RESULTS.items():
            entry = source["files"][kind]
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("invalid synthetic fixture entry")
            path = root / entry["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("synthetic fixture path escapes manifest directory")
            if sha256(path) != entry.get("sha256"):
                raise ValueError(f"synthetic fixture checksum mismatch: {source_system}/{kind}")
            audit = build_source_report(path, source_system, synthetic=True)
            if (audit["source_system"] != source_system or
                    audit["profile_version"] != source.get("profile_version") or
                    audit["profile_status"] != "SYNTHETIC_TEST_ONLY" or
                    audit["valid_record_count"] != expected[0] or
                    audit["quarantine_reasons"] != expected[1]):
                raise ValueError(f"synthetic fixture import contract changed: {source_system}/{kind}")
            if kind in {"primary", "alternate"} and audit["schema_fingerprints"] != [source["schema_fingerprints"].get(kind)]:
                raise ValueError(f"synthetic schema fingerprint mismatch: {source_system}/{kind}")
            files[kind] = {
                "path": entry["path"],
                **{key: audit[key] for key in (
                    "source_sha256", "source_format", "schema_fingerprints", "observed_record_count",
                    "valid_record_count", "quarantine_count", "quarantine_reasons", "duplicate_external_id_count",
                    "invalid_date_count", "malformed_csv_count", "pii_redacted_ticket_count",
                )},
            }
            totals["file_count"] += 1
            totals["observed_record_count"] += audit["observed_record_count"]
            totals["valid_ticket_count"] += audit["valid_record_count"]
            totals["quarantine_count"] += audit["quarantine_count"]
            totals["duplicate_external_id_count"] += audit["duplicate_external_id_count"]
            totals["pii_redacted_ticket_count"] += audit["pii_redacted_ticket_count"]
            quarantine_reasons.update(audit["quarantine_reasons"])
            field_presence.update(audit["valid_field_presence"])
            for name, counts in distributions.items():
                counts.update(audit[name])
            minimum = audit["time_range"]["min"]
            maximum = audit["time_range"]["max"]
            if minimum is not None:
                first_date = minimum if first_date is None else min(first_date, minimum)
            if maximum is not None:
                last_date = maximum if last_date is None else max(last_date, maximum)
        source_reports.append({
            "source_system": source_system,
            "profile_version": source["profile_version"],
            "profile_status": "SYNTHETIC_TEST_ONLY",
            "files": files,
        })
    if seen_sources != set(CANONICAL_SOURCE_SYSTEMS):
        raise ValueError("synthetic fixture manifest lacks source profiles")
    return {
        "report_version": "synthetic-source-quality.v1",
        "synthetic": True,
        "purpose": "importer and quality-gate validation only; not customer or training evidence",
        "fixture_manifest_sha256": sha256(manifest_path),
        "source_count": len(source_reports),
        "totals": {
            **dict(sorted(totals.items())),
            "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
            "valid_field_presence": dict(sorted(field_presence.items())),
            "time_range": {"min": first_date, "max": last_date},
            "duplicate_repeat_gold_set_available": False,
            "label_ground_truth": "UNSUITABLE_SYNTHETIC",
            **{name: dict(sorted(counts.items())) for name, counts in distributions.items()},
        },
        "sources": source_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": "synthetic source audit failed"}), file=sys.stderr)
        return 2
    print(json.dumps({"source_count": report["source_count"], **report["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
