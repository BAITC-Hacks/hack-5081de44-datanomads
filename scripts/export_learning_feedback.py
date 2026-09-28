#!/usr/bin/env python3
"""Read Core feedback from PostgreSQL into a reviewed offline JSONL export."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.feedback_export import FEEDBACK_QUERY, export_feedback, load_review_links


async def read_feedback(database_url: str, cycle_id: str) -> list[dict]:
    import asyncpg

    connection = await asyncpg.connect(database_url)
    try:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            return [dict(row) for row in await connection.fetch(FEEDBACK_QUERY, cycle_id)]
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--production-model-version", required=True)
    parser.add_argument("--review-links", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "FAILED", "error": "DATABASE_URL_REQUIRED"}), file=sys.stderr)
        return 2
    try:
        links = load_review_links(args.review_links)
    except Exception:
        print(json.dumps({"status": "FAILED", "error": "INVALID_REVIEW_LINKS"}), file=sys.stderr)
        return 2
    try:
        rows = asyncio.run(read_feedback(database_url, args.cycle_id))
    except Exception:
        print(json.dumps({"status": "FAILED", "error": "DATABASE_READ_FAILED"}), file=sys.stderr)
        return 2
    try:
        result = export_feedback(rows, links, args.output, cycle_id=args.cycle_id,
                                 production_model_version=args.production_model_version)
    except Exception:
        print(json.dumps({"status": "FAILED", "error": "EXPORT_FAILED"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
