#!/usr/bin/env python3
"""Export PII-free drift counts for one PostgreSQL time window."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.dataset_builder import checksum
from training.drift import (CHAR_LIMITS, CONFIDENCE_LIMITS, LANGUAGES,
                            TOKENIZER_FILES, TOKEN_LIMITS, DriftSnapshot)
from training.feedback_dataset import TOPICS


TICKET_QUERY = """
SELECT t.original_text, t.language, tp.topic_id, tp.confidence
FROM tickets t
LEFT JOIN LATERAL (
    SELECT topic_id, confidence FROM ticket_predictions
    WHERE ticket_id = t.id AND model_version = $3
    ORDER BY created_at, id LIMIT 1
) tp ON TRUE
WHERE t.created_at >= $1 AND t.created_at < $2
ORDER BY t.id
"""

DECISION_QUERY = """
WITH first_decision AS (
    SELECT DISTINCT ON (ticket_id) ticket_id, decision, created_at
    FROM operator_decisions
    ORDER BY ticket_id, created_at, id
)
SELECT COUNT(*)::int AS decision_count,
       COUNT(*) FILTER (WHERE od.decision = 'CORRECTED')::int AS corrected_count
FROM first_decision od
JOIN LATERAL (
    SELECT model_version FROM ticket_predictions
    WHERE ticket_id = od.ticket_id AND created_at < od.created_at
    ORDER BY created_at DESC, id DESC LIMIT 1
) tp ON TRUE
WHERE od.created_at >= $1 AND od.created_at < $2 AND tp.model_version = $3
"""


def _bucket(value: int | float, limits: tuple[int | float, ...]) -> int:
    return next((index for index, limit in enumerate(limits) if value < limit), len(limits))


def _tokenizer(directory: Path):
    if (directory.is_symlink() or not directory.is_dir() or
            any(not (directory / name).is_file() or (directory / name).is_symlink()
                for name in TOKENIZER_FILES)):
        raise ValueError("tokenizer must be a complete local directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
    checksums = {name: checksum(directory / name) for name in sorted(TOKENIZER_FILES)}
    return tokenizer, checksums


async def build_snapshot(database_url: str, start: datetime, end: datetime,
                         model_version: str, tokenizer_dir: Path | None = None) -> dict:
    import asyncpg

    if start.tzinfo is None or end.tzinfo is None or start >= end or end > datetime.now(timezone.utc):
        raise ValueError("snapshot window must be closed and timezone aware")
    tokenizer, tokenizer_checksums = _tokenizer(tokenizer_dir) if tokenizer_dir else (None, None)
    char_counts = [0] * 6
    token_counts = [0] * 6 if tokenizer is not None else None
    confidence_counts = [0] * 5
    languages: Counter = Counter()
    topics: Counter = Counter()
    ticket_count = prediction_count = topic_prediction_count = 0
    connection = await asyncpg.connect(database_url)
    try:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            async for row in connection.cursor(TICKET_QUERY, start, end, model_version, prefetch=100):
                value = row["original_text"]
                if not isinstance(value, str):
                    raise ValueError("ticket text is missing")
                ticket_count += 1
                char_counts[_bucket(len(value), CHAR_LIMITS)] += 1
                if token_counts is not None:
                    length = len(tokenizer.encode(value, add_special_tokens=True, truncation=False))
                    token_counts[_bucket(length, TOKEN_LIMITS)] += 1
                language = row["language"]
                languages[language if language in LANGUAGES else "UNKNOWN"] += 1
                topic = row["topic_id"]
                if topic is not None:
                    topics[topic if topic in TOPICS else "unknown"] += 1
                    topic_prediction_count += 1
                confidence = row["confidence"]
                if confidence is not None:
                    numeric = float(confidence)
                    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                        raise ValueError("prediction confidence is invalid")
                    confidence_counts[_bucket(numeric, CONFIDENCE_LIMITS)] += 1
                    prediction_count += 1
            decision = await connection.fetchrow(DECISION_QUERY, start, end, model_version)
            similarity_count = await connection.fetchval(
                "SELECT COUNT(*)::int FROM similarity_feedback WHERE created_at >= $1 AND created_at < $2",
                start, end,
            )
    finally:
        await connection.close()
    snapshot = DriftSnapshot.model_validate({
        "snapshot_version": "pulse-drift-snapshot.v1", "source": "postgres",
        "model_version": model_version, "window_start": start, "window_end": end,
        "ticket_count": ticket_count, "character_lengths": char_counts,
        "token_lengths": token_counts, "tokenizer_file_checksums": tokenizer_checksums,
        "language_counts": dict(languages), "topic_counts": dict(topics),
        "topic_prediction_count": topic_prediction_count,
        "confidence_counts": confidence_counts, "prediction_count": prediction_count,
        "decision_count": decision["decision_count"], "corrected_count": decision["corrected_count"],
        "similarity_feedback_count": similarity_count,
    })
    return snapshot.model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "FAILED", "error": "DATABASE_URL_REQUIRED"}), file=sys.stderr)
        return 2
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("snapshot output already exists")
        start = datetime.fromisoformat(args.window_start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.window_end.replace("Z", "+00:00"))
        result = asyncio.run(build_snapshot(database_url, start, end, args.model_version,
                                            args.tokenizer))
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "DRIFT_SNAPSHOT_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "COMPLETED", "ticket_count": result["ticket_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
