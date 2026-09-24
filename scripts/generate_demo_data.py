#!/usr/bin/env python3
"""Generate the deterministic, synthetic Pulse 109 data foundation fixture."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.normalization.pipeline import normalize_row
from data.schemas.taxonomy import (
    REGION_DEFINITIONS,
    SOURCE_DISPLAY_NAMES,
    TOPIC_DEFINITIONS,
    canonical_topic_id,
)


DATASET_VERSION = "demo-2026-09-21.v1"
GENERATED_AT = "2026-09-21T00:00:00Z"
SEED = 109
SOURCE_SYSTEMS = tuple(SOURCE_DISPLAY_NAMES)


def _topic_text(topic: Mapping[str, str], language: str, index: int) -> str:
    if language == "KZ":
        return f"{topic['name_kk']} бойынша өтініш: мәселені тексеру қажет, үлгі жазба {index:03d}."
    return f"{topic['name_ru']}: необходимо проверить проблему, демонстрационная запись {index:03d}."


def _raw_rows() -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for index in range(1, 161):
        region = REGION_DEFINITIONS[(index - 1) % len(REGION_DEFINITIONS)]
        topic = TOPIC_DEFINITIONS[((index - 1) * 5) % len(TOPIC_DEFINITIONS)]
        language = "RU" if index % 2 else "KZ"
        source_system = SOURCE_SYSTEMS[(index - 1) % len(SOURCE_SYSTEMS)]
        # Keep source labels intentionally varied so the importer contract is
        # exercised instead of merely re-reading canonical JSON.
        if language == "RU":
            region_label = region["name_ru"]
        else:
            region_label = region["name_kk"]
        rows.append(
            {
                "external_ticket_id": f"demo-{index:04d}",
                "source_system": source_system,
                "region_id": region_label,
                "created_at": f"2026-{((index - 1) % 12) + 1:02d}-{((index - 1) % 27) + 1:02d}T{index % 24:02d}:00:00Z",
                "original_text": _topic_text(topic, language, index),
                "language": language,
                "topic_raw": topic["name_kk"] if language == "KZ" else topic["name_ru"],
                "service_raw": "Городская служба" if language == "RU" else "Қалалық қызмет",
                "priority": ("HIGH" if index % 11 == 0 else "MEDIUM"),
                "status": ("CLOSED" if index % 5 == 0 else "OPEN"),
                "channel": "demo",
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output_dir: Path) -> Mapping[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _raw_rows()

    regions_path = output_dir / "regions.json"
    regions_path.write_text(
        json.dumps(
            [
                {key: region[key] for key in ("id", "name_ru", "name_kk", "name_en")}
                for region in REGION_DEFINITIONS
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    topics_path = output_dir / "topics.json"
    topics_path.write_text(
        json.dumps(
            [
                {key: topic[key] for key in ("id", "name_ru", "name_kk")}
                for topic in TOPIC_DEFINITIONS
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    normalized: List[Mapping[str, Any]] = []
    quarantined: List[Mapping[str, Any]] = []
    for index, row in enumerate(raw_rows, start=1):
        result = normalize_row(row, source_system=row["source_system"], row_number=index)
        if result.ticket is not None:
            normalized.append(result.ticket.to_dict())
        elif result.quarantine is not None:
            quarantined.append(result.quarantine.to_dict())

    tickets_path = output_dir / "tickets.jsonl"
    _write_jsonl(tickets_path, normalized)
    raw_path = output_dir / "raw_tickets.jsonl"
    _write_jsonl(raw_path, raw_rows)

    quarantine_path = output_dir / "quarantine.jsonl"
    _write_jsonl(quarantine_path, quarantined)

    files = []
    for path, kind, count in (
        (regions_path, "regions", len(REGION_DEFINITIONS)),
        (topics_path, "topics", len(TOPIC_DEFINITIONS)),
        (raw_path, "raw_rows", len(raw_rows)),
        (tickets_path, "normalized_tickets", len(normalized)),
        (quarantine_path, "quarantine_rows", len(quarantined)),
    ):
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "kind": kind,
                "records": count,
                "sha256": _sha256(path),
            }
        )
    region_ids = sorted({row["region_id"] for row in normalized})
    topic_ids = sorted({row["topic_id"] for row in normalized})
    languages = sorted({row["language"] for row in normalized})
    sources = sorted({row["source_system"] for row in normalized})
    manifest: Dict[str, Any] = {
        "manifest_version": "dataset-manifest.v1",
        "dataset_id": "pulse109-demo",
        "dataset_version": DATASET_VERSION,
        "schema_version": "unified-ticket.v1",
        "synthetic": True,
        "purpose": "deterministic demo and pipeline smoke tests; not real service metrics",
        "seed": SEED,
        "generated_at": GENERATED_AT,
        "record_count": len(normalized),
        "quarantine_record_count": len(quarantined),
        "coverage": {
            "region_count": len(region_ids),
            "topic_count": len(topic_ids),
            "language_counts": {language: sum(row["language"] == language for row in normalized) for language in languages},
            "source_system_count": len(sources),
        },
        "region_ids": region_ids,
        "topic_ids": topic_ids,
        "languages": languages,
        "source_systems": sources,
        "pii_policy": {
            "raw_pii_in_normalized_fixture": False,
            "text_redaction_tokens": ["[EMAIL]", "[PHONE]", "[IIN]", "[NAME]", "[ADDRESS]"],
            "vector_payload_fields": ["ticket_id", "region_id", "topic_id", "created_at"],
        },
        "files": files,
    }
    manifest_path = manifest_dir / "demo-2026-09-21.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "demo")
    parser.add_argument("--check", action="store_true", help="generate and assert the required coverage")
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    if args.check:
        coverage = manifest["coverage"]
        checks = {
            "regions": coverage["region_count"] >= 20,
            "topics": coverage["topic_count"] >= 10,
            "languages": set(manifest["languages"]) >= {"RU", "KZ"},
            "valid": manifest["record_count"] > 0,
            "quarantine": manifest["quarantine_record_count"] == 0,
        }
        if not all(checks.values()):
            raise SystemExit(f"demo coverage check failed: {checks}")
    print(json.dumps({"dataset_version": manifest["dataset_version"], "records": manifest["record_count"], "coverage": manifest["coverage"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
