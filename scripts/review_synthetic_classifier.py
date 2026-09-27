#!/usr/bin/env python3
"""Record human decisions on generated classifier candidates before export."""

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
from data.schemas.taxonomy import TOPIC_DEFINITIONS
from scripts.pulse_sdg import DEFAULT_SEEDS, LANGUAGES, STYLES, clean_text, read_seeds, source_checksum


TOPICS = {topic["id"] for topic in TOPIC_DEFINITIONS}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
CANDIDATE_FIELDS = {
    "variant_id", "scenario_id", "split_group", "language", "style", "text",
    "topic_id", "subtopic_id", "synthetic", "generator_model", "prompt_version",
    "generator_seed", "source_scenario_sha256", "review_status",
}
CHECKS = (
    "scenario_facts_valid", "facts_preserved", "no_invented_facts",
    "topic_correct", "language_correct", "style_correct",
)
DECISIONS = {"PENDING", "APPROVED", "REJECTED", "DEFERRED"}
REASONS = {
    "VERIFIED", "INVALID_SCENARIO", "FACT_MISMATCH", "INVENTED_FACT",
    "WRONG_TOPIC", "WRONG_LANGUAGE", "WRONG_STYLE", "SENSITIVE_CONTENT",
    "OTHER_QUALITY", "INSUFFICIENT_EVIDENCE",
}
REVIEW_FIELDS = {
    "candidate", "scenario_facts_ru", "scenario_critical_facts",
    "scenario_forbidden_invented_facts", "scenario_source_provenance",
    "scenario_review_status", "scenario_context", "candidate_source_sha256", "decision",
    "reviewer_id", "reviewed_at", "review_reason", "checks",
}
SCENARIO_CONTEXT_FIELDS = ("object_type", "region_constraints", "time_context", "duplicate_group", "repeat_group")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            rows.append(row)
    if not rows:
        raise ValueError("input is empty")
    return rows


def _candidates(path: Path, scenario_path: Path) -> tuple[dict[str, dict], dict[str, dict], str]:
    seeds = read_seeds(scenario_path)
    scenario_sha = source_checksum(scenario_path)
    candidates = {}
    seen_texts = set()
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        candidate_id = row.get("variant_id")
        scenario_id = row.get("scenario_id")
        seed = seeds.get(scenario_id) if isinstance(scenario_id, str) else None
        if (set(row) != CANDIDATE_FIELDS or
                not isinstance(candidate_id, str) or not ID_RE.fullmatch(candidate_id) or
                seed is None or not ID_RE.fullmatch(scenario_id) or
                row["split_group"] != scenario_id or
                not isinstance(row["topic_id"], str) or row["topic_id"] not in TOPICS or
                (row["topic_id"], row["subtopic_id"]) != (seed["topic_id"], seed["subtopic_id"]) or
                not isinstance(row["language"], str) or row["language"] not in LANGUAGES or
                not isinstance(row["style"], str) or row["style"] not in STYLES or
                row["synthetic"] is not True or row["review_status"] != "PENDING" or
                row["source_scenario_sha256"] != scenario_sha or
                not isinstance(row["text"], str) or not row["text"].strip() or
                clean_text(row["text"]) != row["text"] or scan_pii(row["text"]).detected or
                any(not isinstance(row[key], str) or not row[key].strip()
                    for key in ("generator_model", "prompt_version")) or
                type(row["generator_seed"]) is not int or row["generator_seed"] < 0):
            raise ValueError(f"candidate line {line_number}: invalid origin, label, text or provenance")
        digest = hashlib.sha256(
            f"{scenario_id}\0{row['language']}\0{row['style']}\0{row['text']}".encode("utf-8")
        ).hexdigest()[:16]
        if candidate_id != f"{scenario_id}_{digest}":
            raise ValueError(f"candidate line {line_number}: variant_id does not match content")
        if candidate_id in candidates:
            raise ValueError(f"candidate line {line_number}: duplicate variant_id")
        normalized_text = row["text"].casefold()
        if normalized_text in seen_texts:
            raise ValueError(f"candidate line {line_number}: duplicate text")
        seen_texts.add(normalized_text)
        candidates[candidate_id] = row
    return candidates, seeds, source_checksum(path)


def prepare(candidate_path: Path, scenario_path: Path, output_path: Path) -> int:
    candidates, seeds, candidate_sha = _candidates(candidate_path, scenario_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates.values():
            row = {
                "candidate": candidate,
                "scenario_facts_ru": seeds[candidate["scenario_id"]]["facts_ru"],
                "scenario_critical_facts": seeds[candidate["scenario_id"]]["critical_facts"],
                "scenario_forbidden_invented_facts": seeds[candidate["scenario_id"]]["forbidden_invented_facts"],
                "scenario_source_provenance": seeds[candidate["scenario_id"]]["source_provenance"],
                "scenario_review_status": seeds[candidate["scenario_id"]]["review_status"],
                "scenario_context": {key: seeds[candidate["scenario_id"]][key]
                                     for key in SCENARIO_CONTEXT_FIELDS if key in seeds[candidate["scenario_id"]]},
                "candidate_source_sha256": candidate_sha,
                "decision": "PENDING",
                "reviewer_id": None,
                "reviewed_at": None,
                "review_reason": None,
                "checks": {check: None for check in CHECKS},
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(candidates)


def read_reviews(review_path: Path, candidate_path: Path, scenario_path: Path) -> list[dict]:
    candidates, seeds, candidate_sha = _candidates(candidate_path, scenario_path)
    reviews = _read_jsonl(review_path)
    seen = set()
    for line_number, row in enumerate(reviews, start=1):
        candidate = row.get("candidate")
        candidate_id = candidate.get("variant_id") if isinstance(candidate, dict) else None
        expected = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
        seed = seeds[expected["scenario_id"]] if expected is not None else None
        if (set(row) != REVIEW_FIELDS or expected is None or candidate != expected or
                row["scenario_facts_ru"] != seed["facts_ru"] or
                row["scenario_critical_facts"] != seed["critical_facts"] or
                row["scenario_forbidden_invented_facts"] != seed["forbidden_invented_facts"] or
                row["scenario_source_provenance"] != seed["source_provenance"] or
                row["scenario_review_status"] != seed["review_status"] or
                row["scenario_context"] != {key: seed[key] for key in SCENARIO_CONTEXT_FIELDS if key in seed} or
                row["candidate_source_sha256"] != candidate_sha or
                candidate_id in seen):
            raise ValueError(f"review line {line_number}: candidate or scenario changed, duplicated or missing")
        seen.add(candidate_id)
        decision = row["decision"]
        checks = row["checks"]
        if not isinstance(decision, str) or decision not in DECISIONS or not isinstance(checks, dict) or set(checks) != set(CHECKS):
            raise ValueError(f"review line {line_number}: invalid decision or checks")
        if decision == "PENDING":
            if (any(row[key] is not None for key in ("reviewer_id", "reviewed_at", "review_reason")) or
                    any(value is not None for value in checks.values())):
                raise ValueError(f"review line {line_number}: pending row contains review data")
            continue
        reviewer = row["reviewer_id"]
        reviewed_at = row["reviewed_at"]
        reason = row["review_reason"]
        if (not isinstance(reviewer, str) or not ID_RE.fullmatch(reviewer) or
                not isinstance(reviewed_at, str) or not isinstance(reason, str) or reason not in REASONS or
                any(value is not None and type(value) is not bool for value in checks.values())):
            raise ValueError(f"review line {line_number}: incomplete review provenance")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"review line {line_number}: invalid reviewed_at") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"review line {line_number}: reviewed_at needs timezone")
        if decision == "APPROVED":
            if reason != "VERIFIED" or any(value is not True for value in checks.values()):
                raise ValueError(f"review line {line_number}: approval requires every human check")
        elif reason == "VERIFIED":
            raise ValueError(f"review line {line_number}: non-approved row has verified reason")
    if len(seen) != len(candidates):
        raise ValueError("review queue is incomplete")
    return reviews


def approved_records(review_path: Path, candidate_path: Path, scenario_path: Path) -> list[dict]:
    reviews = read_reviews(review_path, candidate_path, scenario_path)
    review_sha = source_checksum(review_path)
    approved = []
    for row in reviews:
        if row["decision"] != "APPROVED":
            continue
        candidate = {key: value for key, value in row["candidate"].items() if key != "review_status"}
        approved.append({
            **candidate,
            "review_status": "APPROVED",
            "reviewer_id": row["reviewer_id"],
            "reviewed_at": row["reviewed_at"],
            "review_evidence_sha256": review_sha,
        })
    if not approved:
        raise ValueError("no human-approved candidates to export")
    return approved


def export_approved(review_path: Path, candidate_path: Path, scenario_path: Path, output_path: Path) -> int:
    approved = approved_records(review_path, candidate_path, scenario_path)
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
        command.add_argument("--candidates", type=Path, required=True)
        command.add_argument("--scenarios", type=Path, default=DEFAULT_SEEDS)
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("review", type=Path)
            if name == "export":
                command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            count = prepare(args.candidates, args.scenarios, args.output)
        elif args.command == "validate":
            count = len(read_reviews(args.review, args.candidates, args.scenarios))
        else:
            count = export_approved(args.review, args.candidates, args.scenarios, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"command": args.command, "records": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
