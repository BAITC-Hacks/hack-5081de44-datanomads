#!/usr/bin/env python3
"""Prepare and validate human taxonomy reviews without auto-approving source labels."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.normalization.pii import scan_pii
from data.schemas.taxonomy import CANONICAL_SOURCE_SYSTEMS, TOPIC_DEFINITIONS
from scripts.strict_json import unique_object


CATALOG = ROOT / "data/catalogs/almaty_2025_taxonomy_review.json"
TOPIC_IDS = {topic["id"] for topic in TOPIC_DEFINITIONS}
PROPOSAL_STATUSES = {"CANDIDATE", "MANUAL_REVIEW", "INSUFFICIENT_LABEL", "OUT_OF_SCOPE"}
DECISIONS = {"PENDING", "APPROVED", "REJECTED", "DEFERRED"}
REVIEW_REASONS = {"SOURCE_CONTEXT_VERIFIED", "AMBIGUOUS_LABEL", "INSUFFICIENT_CONTEXT", "OUT_OF_SCOPE"}
SUBTOPIC_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
REVIEWER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*\Z")
FIELDS = {
    "proposal_id", "source_profile", "source_system", "source_profile_verified",
    "source_checksum", "catalog_checksum", "raw_direction", "raw_subdirection",
    "observed_count", "proposal_status", "suggested_topic_id", "suggested_subtopic_id",
    "decision", "canonical_topic_id", "canonical_subtopic_id", "reviewer",
    "reviewed_at", "evidence_ref", "review_reason", "approved_for_training",
    "approved_for_routing", "source_text_available",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_almaty(catalog_path: Path, output_path: Path) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (not isinstance(catalog, dict) or not isinstance(catalog.get("pairs"), list) or
            not isinstance(catalog.get("source"), dict) or
            not isinstance(catalog["source"].get("sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", catalog["source"]["sha256"])):
        raise ValueError("catalog has no valid pairs or source checksum")
    catalog_checksum = sha256(catalog_path)
    rows = []
    seen_keys = set()
    for index, pair in enumerate(catalog["pairs"], start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"catalog pair {index} must be an object")
        direction = pair.get("source_category")
        subdirection = pair.get("source_service")
        if (not isinstance(direction, str) or not direction.strip() or
                not isinstance(subdirection, str) or not subdirection.strip() or
                scan_pii(direction).detected or scan_pii(subdirection).detected):
            raise ValueError(f"catalog pair {index} has invalid or sensitive labels")
        key = (direction, subdirection)
        if key in seen_keys:
            raise ValueError(f"catalog pair {index} is duplicated")
        seen_keys.add(key)
        proposal_status = pair.get("proposal_status")
        if not isinstance(proposal_status, str) or proposal_status not in PROPOSAL_STATUSES or pair.get("approved_for_training") is not False or pair.get("approved_for_routing") is not False:
            raise ValueError(f"catalog pair {index} has unsupported review state")
        suggested_topic = pair.get("suggested_topic_id")
        if suggested_topic is not None and (not isinstance(suggested_topic, str) or suggested_topic not in TOPIC_IDS):
            raise ValueError(f"catalog pair {index} suggests an unknown topic")
        proposal_id = hashlib.sha256(json.dumps([catalog_checksum, direction, subdirection], ensure_ascii=False).encode("utf-8")).hexdigest()
        rows.append({
            "proposal_id": proposal_id,
            "source_profile": "almaty_2025_unverified",
            "source_system": None,
            "source_profile_verified": False,
            "source_checksum": catalog["source"]["sha256"],
            "catalog_checksum": catalog_checksum,
            "raw_direction": direction,
            "raw_subdirection": subdirection,
            "observed_count": pair.get("count"),
            "proposal_status": pair["proposal_status"],
            "suggested_topic_id": suggested_topic,
            "suggested_subtopic_id": pair.get("suggested_subtopic_id"),
            "decision": "PENDING",
            "canonical_topic_id": None,
            "canonical_subtopic_id": None,
            "reviewer": None,
            "reviewed_at": None,
            "evidence_ref": None,
            "review_reason": None,
            "approved_for_training": False,
            "approved_for_routing": False,
            "source_text_available": False,
        })
    if not rows:
        raise ValueError("catalog contains no pairs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _validate_row(row: dict, line_number: int) -> None:
    if set(row) != FIELDS:
        raise ValueError(f"review line {line_number}: unexpected fields")
    if (row["source_profile"] != "almaty_2025_unverified" or
            not isinstance(row["proposal_status"], str) or row["proposal_status"] not in PROPOSAL_STATUSES or
            not isinstance(row["decision"], str) or row["decision"] not in DECISIONS):
        raise ValueError(f"review line {line_number}: invalid profile or status")
    if (not isinstance(row["proposal_id"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["proposal_id"]) or
            not isinstance(row["source_checksum"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["source_checksum"]) or
            not isinstance(row["catalog_checksum"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["catalog_checksum"])):
        raise ValueError(f"review line {line_number}: invalid checksums")
    if any(not isinstance(row[field], str) or not row[field].strip() or scan_pii(row[field]).detected for field in ("raw_direction", "raw_subdirection")):
        raise ValueError(f"review line {line_number}: invalid source labels")
    expected_id = hashlib.sha256(json.dumps([row["catalog_checksum"], row["raw_direction"], row["raw_subdirection"]], ensure_ascii=False).encode("utf-8")).hexdigest()
    if row["proposal_id"] != expected_id or type(row["observed_count"]) is not int or row["observed_count"] < 1:
        raise ValueError(f"review line {line_number}: proposal identity or count changed")
    if row["suggested_topic_id"] is not None and (not isinstance(row["suggested_topic_id"], str) or row["suggested_topic_id"] not in TOPIC_IDS):
        raise ValueError(f"review line {line_number}: invalid suggested topic")
    if row["approved_for_training"] is not False or row["approved_for_routing"] is not False or row["source_text_available"] is not False:
        raise ValueError(f"review line {line_number}: metadata-only catalog cannot approve training or routing")
    if row["decision"] == "PENDING":
        if any(row[field] is not None for field in ("canonical_topic_id", "canonical_subtopic_id", "reviewer", "reviewed_at", "evidence_ref", "review_reason")):
            raise ValueError(f"review line {line_number}: pending row contains a decision")
        return
    if (not isinstance(row["reviewer"], str) or not REVIEWER_RE.fullmatch(row["reviewer"]) or
            not isinstance(row["reviewed_at"], str) or
            not isinstance(row["evidence_ref"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", row["evidence_ref"]) or
            not isinstance(row["review_reason"], str) or row["review_reason"] not in REVIEW_REASONS):
        raise ValueError(f"review line {line_number}: review provenance is incomplete")
    try:
        reviewed_at = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"review line {line_number}: invalid reviewed_at") from error
    if reviewed_at.tzinfo is None:
        raise ValueError(f"review line {line_number}: reviewed_at needs timezone")
    if row["decision"] == "APPROVED":
        if (not isinstance(row["source_system"], str) or row["source_system"] not in CANONICAL_SOURCE_SYSTEMS or
                row["source_profile_verified"] is not True or
                not isinstance(row["canonical_topic_id"], str) or row["canonical_topic_id"] not in TOPIC_IDS or
                row["review_reason"] != "SOURCE_CONTEXT_VERIFIED"):
            raise ValueError(f"review line {line_number}: approved mapping lacks verified source or canonical topic")
        subtopic = row["canonical_subtopic_id"]
        if subtopic is not None and (not isinstance(subtopic, str) or not SUBTOPIC_RE.fullmatch(subtopic)):
            raise ValueError(f"review line {line_number}: invalid canonical subtopic")
    elif row["canonical_topic_id"] is not None or row["canonical_subtopic_id"] is not None:
        raise ValueError(f"review line {line_number}: non-approved row has canonical labels")


def read_reviews(path: Path, catalog_path: Path = CATALOG) -> list[dict]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (not isinstance(catalog, dict) or not isinstance(catalog.get("pairs"), list) or
            not isinstance(catalog.get("source"), dict) or
            not isinstance(catalog["source"].get("sha256"), str)):
        raise ValueError("catalog has no valid pairs or source checksum")
    catalog_checksum = sha256(catalog_path)
    expected_pairs = {}
    for index, pair in enumerate(catalog["pairs"], start=1):
        if not isinstance(pair, dict) or not isinstance(pair.get("source_category"), str) or not isinstance(pair.get("source_service"), str):
            raise ValueError(f"catalog pair {index} has invalid labels")
        key = (pair["source_category"], pair["source_service"])
        if key in expected_pairs:
            raise ValueError(f"catalog pair {index} is duplicated")
        expected_pairs[key] = pair
    rows = []
    ids = set()
    mapping_keys = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=unique_object)
        except ValueError as error:
            raise ValueError(f"review line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"review line {line_number}: expected object")
        _validate_row(row, line_number)
        pair = expected_pairs.get((row["raw_direction"], row["raw_subdirection"]))
        if (pair is None or row["catalog_checksum"] != catalog_checksum or
                row["source_checksum"] != catalog["source"]["sha256"] or
                row["observed_count"] != pair["count"] or
                row["proposal_status"] != pair["proposal_status"] or
                row["suggested_topic_id"] != pair["suggested_topic_id"] or
                row["suggested_subtopic_id"] != pair["suggested_subtopic_id"]):
            raise ValueError(f"review line {line_number}: source proposal changed")
        if row["proposal_id"] in ids:
            raise ValueError(f"review line {line_number}: duplicate proposal")
        ids.add(row["proposal_id"])
        if row["decision"] == "APPROVED":
            key = (row["source_system"], row["raw_direction"], row["raw_subdirection"])
            if key in mapping_keys:
                raise ValueError(f"review line {line_number}: duplicate mapping key")
            mapping_keys.add(key)
        rows.append(row)
    if len(rows) != len(expected_pairs):
        raise ValueError("review queue is incomplete or has duplicate source labels")
    return rows


def export_approved(review_path: Path, output_path: Path, catalog_path: Path = CATALOG) -> int:
    rows = read_reviews(review_path, catalog_path)
    approved = [row for row in rows if row["decision"] == "APPROVED"]
    if not approved:
        raise ValueError("no human-approved mappings to export")
    mappings = [{
        "source_system": row["source_system"],
        "raw_direction": row["raw_direction"],
        "raw_subdirection": row["raw_subdirection"],
        "canonical_topic_id": row["canonical_topic_id"],
        "canonical_subtopic_id": row["canonical_subtopic_id"],
        "source_checksum": row["source_checksum"],
        "reviewer": row["reviewer"],
        "reviewed_at": row["reviewed_at"],
        "evidence_ref": row["evidence_ref"],
        "approved_for_training": False,
        "approved_for_routing": False,
    } for row in approved]
    export = {
        "contract_version": "taxonomy-mapping.v1",
        "review_queue_sha256": sha256(review_path),
        "mapping_key_fields": ["source_system", "raw_direction", "raw_subdirection"],
        "mappings": mappings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(export, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return len(mappings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create an unapproved queue from the Almaty proposal catalog")
    prepare.add_argument("--catalog", type=Path, default=CATALOG)
    prepare.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate review decisions and provenance")
    validate.add_argument("input", type=Path)
    validate.add_argument("--catalog", type=Path, default=CATALOG)
    export = commands.add_parser("export", help="export only human-approved mappings")
    export.add_argument("input", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            count = prepare_almaty(args.catalog, args.output)
        elif args.command == "validate":
            count = len(read_reviews(args.input, args.catalog))
        else:
            count = export_approved(args.input, args.output, args.catalog)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        message = str(error) if isinstance(error, ValueError) else "review file could not be read or written"
        print(json.dumps({"error": type(error).__name__, "message": message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"command": args.command, "records": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
