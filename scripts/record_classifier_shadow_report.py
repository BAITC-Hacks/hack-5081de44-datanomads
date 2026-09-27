#!/usr/bin/env python3
"""Compute paired shadow metrics from PostgreSQL and attach them to a candidate."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "ml-service"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.feedback_export import load_review_links
from training.shadow_eval import ShadowPolicy, evaluate_shadow
from training.shadow_export import SHADOW_CONTEXT_QUERY, SHADOW_ROWS_QUERY, export_shadow


def evaluation_decision(offline: dict, report: dict, context: dict,
                        production_checksum: str | None) -> str:
    offline_ready = (
        offline.get("report_version") == "classifier-pair-evaluation.v1" and
        offline.get("decision") == "PENDING_HUMAN_REVIEW" and
        offline.get("regressed_critical_topics") == [] and
        offline.get("candidate", {}).get("model_version") == context["candidate_model_version"] and
        offline.get("candidate", {}).get("artifact_checksum") == context["candidate_artifact_checksum"] and
        offline.get("production", {}).get("model_version") == context["production_model_version"] and
        offline.get("production", {}).get("artifact_checksum") == production_checksum
    )
    if report["decision"] == "PENDING_HUMAN_REVIEW":
        return "READY_TO_REVIEW" if offline_ready else "NOT_READY"
    return report["decision"]


async def record_report(database_url: str, cycle_id: str, policy_path: Path,
                        review_links_path: Path | None) -> dict:
    import asyncpg

    policy_bytes = policy_path.read_bytes()
    policy = ShadowPolicy.model_validate_json(policy_bytes)
    links = load_review_links(review_links_path) if review_links_path else {}
    connection = await asyncpg.connect(database_url)
    try:
        async with connection.transaction(isolation="repeatable_read"):
            context_row = await connection.fetchrow(SHADOW_CONTEXT_QUERY, cycle_id)
            if context_row is None:
                raise ValueError("learning cycle not found")
            context = dict(context_row)
            candidate = context["candidate_model_version"]
            evaluation = await connection.fetchrow(
                "SELECT evaluation_id, metrics_json, shadow_metrics_json FROM model_evaluations "
                "WHERE model_version = $1 ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE",
                candidate,
            )
            if evaluation is None or evaluation["evaluation_id"] != f"offline-{candidate}":
                raise ValueError("candidate offline evaluation is missing")
            production_checksum = await connection.fetchval(
                "SELECT artifact_checksum FROM model_versions WHERE model_version = $1",
                context["production_model_version"],
            )
            rows = [dict(row) for row in await connection.fetch(SHADOW_ROWS_QUERY, context["id"])]
            with tempfile.TemporaryDirectory(prefix="pulse-shadow-") as directory:
                input_path = Path(directory) / "shadow.jsonl"
                frozen_policy_path = Path(directory) / "policy.json"
                frozen_policy_path.write_bytes(policy_bytes)
                export = export_shadow(rows, context, policy, input_path, review_links=links)
                if not input_path.exists():
                    input_path.touch(mode=0o600)
                report = evaluate_shadow(
                    input_path, frozen_policy_path, cycle_id=context["cycle_id"],
                    production_model_version=context["production_model_version"],
                    candidate_model_version=candidate,
                )
            report["export"] = export
            offline = evaluation["metrics_json"]
            existing = evaluation["shadow_metrics_json"]
            offline = json.loads(offline) if isinstance(offline, str) else offline
            existing = json.loads(existing) if isinstance(existing, str) else existing
            if not isinstance(offline, dict) or not isinstance(existing, dict):
                raise ValueError("candidate evaluation is malformed")
            if existing and (existing.get("input_sha256") != report["input_sha256"] or
                             existing.get("policy_sha256") != report["policy_sha256"]):
                raise ValueError("candidate shadow evidence has changed")
            decision = evaluation_decision(offline, report, context, production_checksum)
            await connection.execute(
                "UPDATE model_evaluations SET shadow_metrics_json = $2::jsonb, "
                "critical_regressions = $3::jsonb, sample_size = $4, decision = $5, "
                "evaluator = 'offline+shadow-review' WHERE evaluation_id = $1",
                evaluation["evaluation_id"], json.dumps(report, ensure_ascii=False),
                json.dumps(report["critical_regressions"]), report["sample_count"], decision,
            )
            return {"status": report["status"], "decision": decision,
                    "sample_count": report["sample_count"],
                    "rejected_counts": export["rejected_counts"], "cycle_id": context["cycle_id"]}
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--review-links", type=Path,
                        help="approved text checksums required for real-origin rows")
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "FAILED", "error": "DATABASE_URL_REQUIRED"}), file=sys.stderr)
        return 2
    try:
        result = asyncio.run(record_report(database_url, args.cycle_id, args.policy,
                                           args.review_links))
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "SHADOW_REPORT_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
