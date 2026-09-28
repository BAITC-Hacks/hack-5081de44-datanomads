#!/usr/bin/env python3
"""Review synthetic retrieval pairs against immutable relation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
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
from scripts.pulse_sdg import LANGUAGES
from scripts.strict_json import unique_object


ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
SOURCE_FIELDS = {
    "pair_id", "relation_group", "query_id", "candidate_id", "query_text",
    "candidate_text", "query_language", "candidate_language", "query_context",
    "candidate_context", "synthetic",
}
REVIEW_FIELDS = {
    "source", "source_relation_sha256", "decision", "relation_label",
    "reviewer_id", "reviewed_at", "review_reason", "checks", "same_region",
    "same_object", "same_issue", "same_episode", "prior_episode_resolved",
}
CHECKS = ("facts_consistent", "language_correct", "relation_supported", "no_sensitive_details")
LABELS = {"DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED", "REPEAT"}
REQUIRED_LABELS = {"DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"}
DECISIONS = {"PENDING", "APPROVED", "REJECTED", "DEFERRED"}
REASONS = {
    "VERIFIED", "INSUFFICIENT_CONTEXT", "SOURCE_INCONSISTENT",
    "WRONG_LANGUAGE", "SENSITIVE_CONTENT", "OTHER_QUALITY",
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=unique_object)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            except ValueError as error:
                raise ValueError(f"line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            rows.append(row)
    if not rows:
        raise ValueError("input is empty")
    return rows


def read_source(path: Path) -> tuple[dict[str, dict], str]:
    rows = _read_jsonl(path)
    pairs = {}
    entities = {}
    candidates_per_query: Counter[str] = Counter()
    query_candidates = set()
    for line_number, row in enumerate(rows, start=1):
        if set(row) != SOURCE_FIELDS or row.get("synthetic") is not True:
            raise ValueError(f"source line {line_number}: invalid relation source fields")
        if any(not isinstance(row[key], str) or not ID_RE.fullmatch(row[key])
               for key in ("pair_id", "relation_group", "query_id", "candidate_id")):
            raise ValueError(f"source line {line_number}: invalid relation identifiers")
        if row["query_id"] == row["candidate_id"] or row["pair_id"] in pairs:
            raise ValueError(f"source line {line_number}: duplicate pair or self match")
        for role in ("query", "candidate"):
            language = row[f"{role}_language"]
            text = row[f"{role}_text"]
            context = row[f"{role}_context"]
            if (not isinstance(language, str) or language not in LANGUAGES or
                    not isinstance(text, str) or not text.strip() or scan_pii(text).detected or
                    not isinstance(context, str) or not context.strip() or scan_pii(context).detected):
                raise ValueError(f"source line {line_number}: invalid or sensitive text/context")
            entity_id = row[f"{role}_id"]
            entity = (row["relation_group"], text, language, context)
            if entity_id in entities and entities[entity_id] != entity:
                raise ValueError(f"source line {line_number}: entity crosses groups or changes content")
            entities[entity_id] = entity
        query_candidate = (row["query_id"], row["candidate_id"])
        if query_candidate in query_candidates:
            raise ValueError(f"source line {line_number}: duplicate query candidate")
        query_candidates.add(query_candidate)
        candidates_per_query[row["query_id"]] += 1
        pairs[row["pair_id"]] = row
    if any(count < 2 for count in candidates_per_query.values()):
        raise ValueError("each query needs at least two distinct candidates")
    return pairs, checksum(path)


def prepare(source_path: Path, output_path: Path) -> int:
    pairs, source_sha = read_source(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for source in pairs.values():
            row = {
                "source": source,
                "source_relation_sha256": source_sha,
                "decision": "PENDING",
                "relation_label": None,
                "reviewer_id": None,
                "reviewed_at": None,
                "review_reason": None,
                "checks": {check: None for check in CHECKS},
                "same_region": None,
                "same_object": None,
                "same_issue": None,
                "same_episode": None,
                "prior_episode_resolved": None,
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(pairs)


def read_reviews(review_path: Path, source_path: Path) -> list[dict]:
    pairs, source_sha = read_source(source_path)
    reviews = _read_jsonl(review_path)
    seen = set()
    for line_number, row in enumerate(reviews, start=1):
        source = row.get("source")
        pair_id = source.get("pair_id") if isinstance(source, dict) else None
        if (set(row) != REVIEW_FIELDS or not isinstance(pair_id, str) or
                pair_id not in pairs or source != pairs[pair_id] or
                row["source_relation_sha256"] != source_sha or pair_id in seen):
            raise ValueError(f"review line {line_number}: source pair changed, duplicated or missing")
        seen.add(pair_id)
        decision = row["decision"]
        checks = row["checks"]
        if (not isinstance(decision, str) or decision not in DECISIONS or
                not isinstance(checks, dict) or set(checks) != set(CHECKS)):
            raise ValueError(f"review line {line_number}: invalid decision or checks")
        relation_fields = (
            "relation_label", "same_region", "same_object", "same_issue",
            "same_episode", "prior_episode_resolved",
        )
        if decision == "PENDING":
            if (any(row[key] is not None for key in (*relation_fields, "reviewer_id", "reviewed_at", "review_reason")) or
                    any(value is not None for value in checks.values())):
                raise ValueError(f"review line {line_number}: pending row contains review data")
            continue
        reviewer = row["reviewer_id"]
        reviewed_at = row["reviewed_at"]
        reason = row["review_reason"]
        if (not isinstance(reviewer, str) or not ID_RE.fullmatch(reviewer) or
                not isinstance(reviewed_at, str) or not isinstance(reason, str) or reason not in REASONS or
                any(value is not None and type(value) is not bool for value in checks.values()) or
                any(row[key] is not None and type(row[key]) is not bool for key in relation_fields[1:])):
            raise ValueError(f"review line {line_number}: incomplete review provenance")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"review line {line_number}: invalid reviewed_at") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"review line {line_number}: reviewed_at needs timezone")
        if decision == "APPROVED":
            label = row["relation_label"]
            if (not isinstance(label, str) or label not in LABELS or reason != "VERIFIED" or
                    any(value is not True for value in checks.values()) or
                    any(type(row[key]) is not bool for key in relation_fields[1:5])):
                raise ValueError(f"review line {line_number}: incomplete approved relation")
            if label == "DUPLICATE" and not (
                all(row[key] is True for key in relation_fields[1:5]) and
                row["prior_episode_resolved"] is False
            ):
                raise ValueError(f"review line {line_number}: duplicate needs same region, object, issue and episode")
            if label == "REPEAT" and not (
                all(row[key] is True for key in ("same_region", "same_object", "same_issue")) and
                row["same_episode"] is False and row["prior_episode_resolved"] is True
            ):
                raise ValueError(f"review line {line_number}: repeat needs a resolved prior episode")
            if (label in {"SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"} and
                    all(row[key] is True for key in ("same_region", "same_object", "same_issue")) and
                    (row["same_episode"] is True or row["prior_episode_resolved"] is True)):
                raise ValueError(f"review line {line_number}: same episode or resolved recurrence needs duplicate/repeat label")
        elif row["relation_label"] is not None or reason == "VERIFIED":
            raise ValueError(f"review line {line_number}: non-approved row has a relation label")
    if len(seen) != len(pairs):
        raise ValueError("review queue is incomplete")
    return reviews


def approved_records(review_path: Path, source_path: Path) -> list[dict]:
    reviews = read_reviews(review_path, source_path)
    if not any(row["decision"] == "APPROVED" for row in reviews):
        raise ValueError("no human-approved retrieval pairs to export")
    if any(row["decision"] in {"PENDING", "DEFERRED"} for row in reviews):
        raise ValueError("retrieval source has unresolved pair reviews")
    review_sha = checksum(review_path)
    source_sha = checksum(source_path)
    approved = []
    candidate_counts: Counter[str] = Counter()
    for row in reviews:
        if row["decision"] != "APPROVED":
            continue
        source = row["source"]
        candidate_counts[source["query_id"]] += 1
        approved.append({
            "pair_id": source["pair_id"],
            "relation_group": source["relation_group"],
            "query_id": source["query_id"],
            "candidate_id": source["candidate_id"],
            "query_text": source["query_text"],
            "candidate_text": source["candidate_text"],
            "query_language": source["query_language"],
            "candidate_language": source["candidate_language"],
            "synthetic": True,
            "relation_label": row["relation_label"],
            "source_relation_sha256": source_sha,
            "review_status": "APPROVED",
            "reviewer_id": row["reviewer_id"],
            "reviewed_at": row["reviewed_at"],
            "review_evidence_sha256": review_sha,
        })
    if any(count < 2 for count in candidate_counts.values()):
        raise ValueError("approved retrieval query needs at least two reviewed candidates")
    if not REQUIRED_LABELS.issubset({row["relation_label"] for row in approved}):
        raise ValueError("approved retrieval source lacks required relation labels")
    return approved


def export_approved(review_path: Path, source_path: Path, output_path: Path) -> int:
    approved = approved_records(review_path, source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in approved:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(approved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "export"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("review", type=Path)
            if name == "export":
                command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            count = prepare(args.source, args.output)
        elif args.command == "validate":
            count = len(read_reviews(args.review, args.source))
        else:
            count = export_approved(args.review, args.source, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"command": args.command, "records": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
