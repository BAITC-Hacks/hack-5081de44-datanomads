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

from training.feedback_export import export_feedback, load_review_links


FEEDBACK_QUERY = """
SELECT lf.id AS feedback_id, lf.ticket_id AS db_ticket_id,
       lf.production_model_version, lf.production_prediction::text,
       lf.operator_confirmed_decision::text, lf.accepted_or_corrected,
       lf.validation_status, lf.feedback_created_at,
       t.original_text, t.language,
       od.id AS operator_decision_id, od.decision AS operator_decision_kind,
       od.confirmed_topic_id,
       tp.model_version AS prediction_model_version, tp.topic_id AS prediction_topic_id,
       tp.confidence AS prediction_confidence,
       COALESCE((
           SELECT json_agg(json_build_object(
               'dataset_version', dv.dataset_version,
               'is_synthetic', dv.is_synthetic,
               'manifest_sha256', dv.manifest_sha256,
               'content_sha256', dv.content_sha256
           ) ORDER BY dv.dataset_version)::text
           FROM dataset_ticket_links dtl
           JOIN dataset_versions dv ON dv.dataset_version = dtl.dataset_version
           WHERE dtl.ticket_id = lf.ticket_id
       ), '[]') AS source_lineage
FROM learning_feedback lf
JOIN learning_cycles lc ON lc.id = lf.cycle_id
JOIN tickets t ON t.id = lf.ticket_id
LEFT JOIN operator_decisions od
    ON od.ticket_id = lf.ticket_id
   AND od.id::text = lf.operator_confirmed_decision->>'decision_id'
LEFT JOIN LATERAL (
    SELECT model_version, topic_id, confidence
    FROM ticket_predictions
    WHERE ticket_id = lf.ticket_id AND model_version = lf.production_model_version
      AND created_at <= lf.feedback_created_at
    ORDER BY created_at DESC, id DESC LIMIT 1
) tp ON TRUE
WHERE lc.cycle_id = $1
ORDER BY lf.id
"""


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
