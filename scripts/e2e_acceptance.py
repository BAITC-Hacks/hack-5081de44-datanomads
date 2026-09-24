#!/usr/bin/env python3
"""End-to-end acceptance flow for the PostgreSQL-backed Pulse 109 stack.

This intentionally exercises the public contract instead of importing service
internals.  It is safe to run repeatedly: every run uses a unique source and
the import is checked for idempotency.  The normal worker is expected to leave
the learning candidate at ``TRAINER_NOT_CONFIGURED``; when
``PULSE_TEST_FAKE_TRAINER=true`` is enabled for an integration run, the same
flow also verifies candidate promotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile


class AcceptanceError(RuntimeError):
    pass


def request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: Any | None = None,
    role: str = "ADMIN",
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    headers = {
        "Accept": "application/json",
        "X-Pulse-Role": role,
        "X-User-Id": "e2e-acceptance",
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()
    except URLError as error:
        raise AcceptanceError(f"{method} {path}: {error}") from error


def json_request(*args: Any, **kwargs: Any) -> tuple[int, dict[str, str], dict[str, Any]]:
    status, headers, raw = request(*args, **kwargs)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"expected JSON from {args[1]} {args[2]}, got {raw[:160]!r}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"expected object from {args[1]} {args[2]}")
    return status, headers, payload


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def read_sse_event(response: Any) -> bytes:
    lines = []
    while line := response.readline():
        if line in (b"\n", b"\r\n"):
            if lines:
                return b"".join(lines)
            continue
        if not line.startswith(b":"):
            lines.append(line)
    raise AcceptanceError("SSE stream closed before the next event")


def run(base_url: str, timeout: float, restart_core: bool = False) -> dict[str, Any]:
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    source = f"e2e_acceptance_{suffix}"
    dataset = f"e2e-dataset-{suffix}"
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    external_ids = [f"{source}-{index}" for index in range(3)]
    quarantine = [
        {
            "row_number": 99,
            "reason": "INVALID_VALUE",
            "field": "priority",
            "detail": "acceptance quarantine row",
            "row": {"external_ticket_id": f"{source}-quarantine"},
        }
    ]
    status, _, health = json_request(base_url, "GET", "/healthz", timeout=timeout)
    expect(status == 200 and health.get("status") == "ok", f"health failed: {status} {health}")
    status, _, ready = json_request(base_url, "GET", "/readyz", timeout=timeout)
    expect(status == 200 and ready.get("storage") == "postgres", f"ready failed: {status} {ready}")

    # Keep repeated acceptance runs isolated without weakening the production
    # incident key/cooldown semantics.  A closed incident for the same
    # region/topic/day must stay closed; choose a region whose water-series
    # key has not already been consumed by an earlier run.
    region_candidates = (
        "KZ-KYZYLORDA",
        "KZ-AKTOBE",
        "KZ-ALMATY-REGION",
        "KZ-ASTANA",
        "KZ-ZHAMBYL",
        "KZ-KOSTANAY",
        "KZ-TURKESTAN",
        "KZ-SHYMKENT",
        "KZ-ABAY",
        "KZ-ZHETISU",
    )
    status, _, alert_list = json_request(base_url, "GET", "/api/v1/alerts?limit=500", role="MANAGER", timeout=timeout)
    expect(status == 200, f"alert list failed while selecting test series: {alert_list}")
    today = datetime.now(timezone.utc).date().isoformat()
    consumed_regions = {
        str(item.get("incident_key", "")).split(":")[1]
        for item in alert_list.get("items", [])
        if str(item.get("incident_key", "")).startswith(f"topic_spike:")
        and str(item.get("incident_key", "")).endswith(f":{today}")
        and ":water_supply:" in str(item.get("incident_key", ""))
    }
    region = next((candidate for candidate in region_candidates if candidate not in consumed_regions), region_candidates[0])
    tickets = [
        {
            "external_ticket_id": external_id,
            "region_id": region,
            "created_at": created_at,
            "original_text": f"E2E {suffix}: порыв водопровода, нет воды в доме {index}",
            "language": "ru",
            "topic_raw": "water_supply",
            "status": "OPEN",
            "channel": "e2e",
        }
        for index, external_id in enumerate(external_ids)
    ]
    import_payload = {
        "source_system": source,
        "source_uri": f"memory://{source}.json",
        "dataset_version": dataset,
        "manifest_uri": f"memory://{dataset}/manifest.json",
        "manifest_sha256": "e2e-test-manifest",
        "is_synthetic": True,
        "tickets": tickets,
        "quarantine": quarantine,
    }

    status, _, imported = json_request(base_url, "POST", "/api/v1/import", body=import_payload, timeout=timeout)
    expect(status == 201, f"first import failed: {status} {imported}")
    expect(imported.get("imported_rows") == 3 and imported.get("indexed_rows") == 3, f"import counts: {imported}")
    expect(imported.get("quarantined_rows") == 1, f"quarantine count: {imported}")
    status, _, repeated = json_request(base_url, "POST", "/api/v1/import", body=import_payload, timeout=timeout)
    expect(status == 201, f"repeat import failed: {status} {repeated}")
    expect(repeated.get("imported_rows") == 0 and repeated.get("duplicate_rows") == 3 and repeated.get("indexed_rows") == 0, f"repeat import is not idempotent: {repeated}")
    changed_payload = json.loads(json.dumps(import_payload))
    changed_payload["tickets"][0]["original_text"] = "Изменённый текст с прежней версией набора"
    status, _, conflict = json_request(base_url, "POST", "/api/v1/import", body=changed_payload, timeout=timeout)
    expect(status == 409, f"dataset version accepted different content: {status} {conflict}")
    changed_payload["dataset_version"] = f"{dataset}-changed-ticket"
    changed_payload["manifest_uri"] = f"memory://{dataset}-changed-ticket/manifest.json"
    status, _, conflict = json_request(base_url, "POST", "/api/v1/import", body=changed_payload, timeout=timeout)
    expect(status == 409, f"new dataset silently linked changed source ticket: {status} {conflict}")
    next_version = json.loads(json.dumps(import_payload))
    next_version["dataset_version"] = f"{dataset}-same-ticket"
    next_version["manifest_uri"] = f"memory://{dataset}-same-ticket/manifest.json"
    status, _, linked = json_request(base_url, "POST", "/api/v1/import", body=next_version, timeout=timeout)
    expect(status == 201 and linked.get("duplicate_rows") == 3, f"new dataset could not link unchanged tickets: {status} {linked}")

    status, _, listed = json_request(base_url, "GET", "/api/v1/tickets?limit=100", timeout=timeout)
    expect(status == 200, f"ticket list failed: {status} {listed}")
    selected = [item for item in listed.get("items", []) if item.get("external_ref") in external_ids]
    expect(len(selected) == 3, f"imported tickets not visible: {external_ids}")
    ticket_ids = [str(item["id"]) for item in selected]

    status, _, preview = json_request(base_url, "POST", "/api/v1/assist/preview", body={"ticket_id": ticket_ids[0]}, timeout=timeout)
    expect(status == 200 and preview.get("source") == "postgres-ticket+ml+qdrant", f"preview failed: {status} {preview}")
    response_template = preview.get("response_template", {})
    expect(response_template.get("body"), "demo/manual response template is missing")
    expect(
        response_template.get("approved") is False
        and response_template.get("source") == "MANUAL_DEMO",
        f"invented response template must remain manual/demo: {response_template}",
    )
    status, _, decision = json_request(
        base_url,
        "POST",
        f"/api/v1/assist/{ticket_ids[0]}/correct",
        body={"topic_id": "water_supply", "service": "service_water", "priority": "high", "note": "e2e correction"},
        role="OPERATOR",
        timeout=timeout,
    )
    expect(status == 200 and decision.get("decision", {}).get("action") == "correct", f"correction failed: {status} {decision}")
    status, _, persisted = json_request(base_url, "GET", f"/api/v1/tickets/{ticket_ids[0]}", role="OPERATOR", timeout=timeout)
    expect(status == 200 and persisted.get("latest_decision", {}).get("action") == "correct", f"decision did not persist: {persisted}")
    if restart_core:
        repo_root = Path(__file__).resolve().parent.parent
        subprocess.run(["docker", "compose", "restart", "core-api"], cwd=repo_root, check=True)
        for _ in range(30):
            time.sleep(1)
            status, _, restarted_health = json_request(base_url, "GET", "/healthz", timeout=timeout)
            if status == 200 and restarted_health.get("status") == "ok":
                break
        else:
            raise AcceptanceError("core-api did not become healthy after restart")
        status, _, persisted_after_restart = json_request(
            base_url,
            "GET",
            f"/api/v1/tickets/{ticket_ids[0]}",
            role="OPERATOR",
            timeout=timeout,
        )
        expect(
            status == 200 and persisted_after_restart.get("latest_decision", {}).get("action") == "correct",
            f"decision was lost after core restart: {persisted_after_restart}",
        )

    status, _, relation = json_request(
        base_url,
        "POST",
        f"/api/v1/tickets/{ticket_ids[0]}/relation-feedback",
        body={"related_ticket_id": ticket_ids[1], "relation": "REPEAT", "decision": "CONFIRMED"},
        role="OPERATOR",
        timeout=timeout,
    )
    expect(status == 201 and "relation:REPEAT:CONFIRMED" == relation.get("feedback_type"), f"relation feedback failed: {relation}")

    query = urlencode({"dimension": "region", "value": region, "range": "30d", "limit": "2"})
    status, _, drilldown = json_request(base_url, "GET", f"/api/v1/analytics/drilldown?{query}", role="MANAGER", timeout=timeout)
    expect(status == 200 and drilldown.get("total", 0) >= 3, f"analytics drilldown failed: {drilldown}")
    for text in ("Сколько обращений за 30 дней?", "Какой тренд?", "Сравни регионы", "Топ тем", "Где всплески?", "Дай прогноз"):
        status, _, result = json_request(base_url, "POST", "/api/v1/analytics/query", body={"text": text, "filters": {"range": "30d"}}, role="MANAGER", timeout=timeout)
        expect(status == 200 and result.get("intent"), f"QueryIntent failed for {text!r}: {result}")

    for horizon in (30, 60, 90):
        status, _, forecast = json_request(base_url, "GET", f"/api/v1/forecast?horizon={horizon}", role="MANAGER", timeout=timeout)
        expect(status == 200 and forecast.get("horizon_days") == horizon, f"forecast {horizon} failed: {forecast}")
        expect(forecast.get("history"), f"forecast {horizon} has no PostgreSQL history")
        expect(
            str(forecast.get("model_version", "")).startswith("forecast-")
            and forecast.get("model_version") != "embedder-demo-2026-09-21-001",
            f"forecast must expose its own model_version: {forecast}",
        )

    status, _, detected = json_request(base_url, "POST", "/api/v1/alerts/detect", role="MANAGER", timeout=timeout)
    expect(status == 200 and detected.get("items"), f"spike detector found no alert: {detected}")
    alert_id = str(detected["items"][0]["id"])
    event_request = Request(
        base_url.rstrip("/") + "/api/v1/events",
        headers={"Accept": "text/event-stream", "X-Pulse-Role": "MANAGER", "X-User-Id": "e2e-acceptance"},
    )
    with urlopen(event_request, timeout=timeout) as event_response:
        expect(event_response.status == 200 and "text/event-stream" in event_response.headers.get("Content-Type", ""), "SSE endpoint did not return an event stream")
        snapshot = read_sse_event(event_response)
        expect(b"event: alerts.snapshot" in snapshot, f"SSE snapshot missing: {snapshot[:160]!r}")
        status, _, acknowledged = json_request(base_url, "POST", f"/api/v1/alerts/{alert_id}/ack", role="MANAGER", timeout=timeout)
        expect(status == 200 and acknowledged.get("status") == "ACKNOWLEDGED", f"alert ack failed: {acknowledged}")
        changed = read_sse_event(event_response)
        expect(b"event: alerts.changed" in changed, f"SSE alert update missing: {changed[:160]!r}")
    status, _, closed = json_request(base_url, "POST", f"/api/v1/alerts/{alert_id}/close", role="MANAGER", timeout=timeout)
    expect(status == 200 and closed.get("status") == "CLOSED", f"alert close failed: {closed}")

    for report_path, magic in (("/api/v1/analytics/export.pdf", b"%PDF-1.4"), ("/api/v1/analytics/export.xlsx", b"PK\x03\x04")):
        status, _, report = request(base_url, "GET", report_path, role="MANAGER", timeout=timeout)
        expect(status == 200 and report.startswith(magic), f"report failed: {report_path} HTTP {status}")
        if report_path.endswith("xlsx"):
            temp_path = Path(os.getenv("TMPDIR", "/tmp")) / f"pulse109-{suffix}.xlsx"
            temp_path.write_bytes(report)
            try:
                with zipfile.ZipFile(temp_path) as archive:
                    bad = archive.testzip()
                expect(bad is None, f"XLSX archive is corrupt: {bad}")
            finally:
                temp_path.unlink(missing_ok=True)

    status, _, unauthorized = json_request(base_url, "GET", "/api/v1/analytics", role="OPERATOR", timeout=timeout)
    expect(status == 403, f"RBAC expected 403, got {status} {unauthorized}")
    status, _, reindex = json_request(base_url, "POST", "/api/v1/retrieval/reindex", role="MANAGER", timeout=timeout)
    expect(status == 202 and reindex.get("job_type") == "REINDEX_QDRANT", f"reindex queue failed: {reindex}")

    status, _, cycle = json_request(
        base_url,
        "POST",
        "/api/v1/learning",
        body={"dataset_version": f"{dataset}-feedback", "candidate_model_version": f"classifier-{suffix}"},
        role="ML_REVIEWER",
        timeout=timeout,
    )
    expect(status == 201, f"learning cycle creation failed: {cycle}")
    cycle_id = str(cycle["id"])
    status, _, feedback = json_request(base_url, "POST", f"/api/v1/learning/{cycle_id}/feedback", body={"ticket_id": ticket_ids[0], "decision": "confirm"}, role="OPERATOR", timeout=timeout)
    expect(status == 201, f"learning feedback failed: {feedback}")
    status, _, closed_cycle = json_request(base_url, "POST", "/api/v1/learning/cycle/close", body={"cycle_id": cycle_id}, role="ML_REVIEWER", timeout=timeout)
    expect(status == 202 and closed_cycle.get("state") == "TRAINING", f"learning close failed: {closed_cycle}")
    evaluation: dict[str, Any] = {}
    for _ in range(20):
        time.sleep(0.5)
        status, _, evaluation = json_request(base_url, "GET", "/api/v1/learning/candidate/evaluation", role="ML_REVIEWER", timeout=timeout)
        if evaluation.get("state") != "TRAINING":
            break
    if evaluation.get("decision") == "READY_TO_REVIEW":
        status, _, promoted = json_request(base_url, "POST", "/api/v1/learning/candidate/promote", body={"note": "e2e fake trainer"}, role="ML_REVIEWER", timeout=timeout)
        expect(status == 200 and promoted.get("state") == "PROMOTED", f"candidate promotion failed: {promoted}")
        learning_result = "PROMOTED_TEST_CANDIDATE"
    else:
        expect(evaluation.get("offline_metrics", {}).get("status") == "TRAINER_NOT_CONFIGURED", f"unexpected normal-mode trainer result: {evaluation}")
        status, _, rejected = json_request(base_url, "POST", "/api/v1/learning/candidate/reject", body={"note": "e2e normal mode"}, role="ML_REVIEWER", timeout=timeout)
        expect(status == 200 and rejected.get("state") == "REJECTED", f"candidate rejection failed: {rejected}")
        learning_result = "TRAINER_NOT_CONFIGURED_AND_REJECTED"

    return {
        "source_system": source,
        "ticket_ids": ticket_ids,
        "alert_id": alert_id,
        "learning": learning_result,
        "checks": "postgres+ml+qdrant+operator+analytics+alerts+forecast+reports+rbac",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("PULSE_BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--restart-core",
        action="store_true",
        help="restart the local core-api container and verify persisted state",
    )
    args = parser.parse_args()
    try:
        result = run(args.base_url, args.timeout, restart_core=args.restart_core)
    except AcceptanceError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
