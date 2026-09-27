#!/usr/bin/env python3
"""Export reviewed, paired shadow evidence from PostgreSQL without ticket text."""

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

from training.feedback_export import load_review_links
from training.shadow_eval import ShadowPolicy
from training.shadow_export import SHADOW_CONTEXT_QUERY, SHADOW_ROWS_QUERY, export_shadow


async def read_shadow(database_url: str, cycle_id: str) -> tuple[dict, list[dict]]:
    import asyncpg

    connection = await asyncpg.connect(database_url)
    try:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            context = await connection.fetchrow(SHADOW_CONTEXT_QUERY, cycle_id)
            if context is None:
                raise ValueError("learning cycle not found")
            rows = await connection.fetch(SHADOW_ROWS_QUERY, context["id"])
            return dict(context), [dict(row) for row in rows]
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--review-links", type=Path,
                        help="approved text checksums required for real-origin rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "FAILED", "error": "DATABASE_URL_REQUIRED"}), file=sys.stderr)
        return 2
    try:
        policy = ShadowPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
        links = load_review_links(args.review_links) if args.review_links else {}
        context, rows = asyncio.run(read_shadow(database_url, args.cycle_id))
        result = export_shadow(rows, context, policy, args.output, review_links=links)
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "SHADOW_EXPORT_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
