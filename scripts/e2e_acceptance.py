#!/usr/bin/env python3
"""End-to-end acceptance flow for the PostgreSQL-backed Pulse 109 stack.

This intentionally exercises the public contract instead of importing service
internals. It is safe to run repeatedly: every run uses a unique source and
the import is checked for idempotency. The normal worker trains and
shadow-serves a candidate during its evaluation window while production
remains the operator recommendation. When ``PULSE_TEST_FAKE_TRAINER=true`` is
enabled in demo/test mode, the same flow verifies that synthetic evidence
cannot be promoted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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


def postgres_ticket_counts(
    repo_root: Path,
    *,
    region_id: str,
    topic_id: str | None = None,
    status: str | None = None,
    channel: str | None = None,
) -> tuple[int, int]:
    query = """
        SELECT
          COUNT(*) FILTER (
            WHERE t.created_at >= now() - INTERVAL '30 days'
              AND t.created_at <= now()
          ),
          COUNT(*) FILTER (
            WHERE t.created_at >= now() - INTERVAL '60 days'
              AND t.created_at < now() - INTERVAL '30 days'
          )
        FROM tickets t
        WHERE t.region_id = :'region'
          AND (NULLIF(:'topic', '') IS NULL OR t.topic_id = NULLIF(:'topic', ''))
          AND (NULLIF(:'status', '') IS NULL OR t.status = UPPER(NULLIF(:'status', '')))
          AND (NULLIF(:'channel', '') IS NULL OR t.channel = NULLIF(:'channel', ''));
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "sh",
            "-lc",
            'exec psql -XAtq -F "|" -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"',
            "pulse109-e2e-db-check",
            "-v",
            f"region={region_id}",
            "-v",
            f"topic={topic_id or ''}",
            "-v",
            f"status={status or ''}",
            "-v",
            f"channel={channel or ''}",
        ],
        cwd=repo_root,
        input=query,
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        current, previous = result.stdout.strip().split("|")
        return int(current), int(previous)
    except ValueError as error:
        raise AcceptanceError(
            f"PostgreSQL returned invalid current/previous ticket counts: {result.stdout!r}"
        ) from error


def postgres_ticket_count_lookback(
    repo_root: Path,
    *,
    region_id: str,
    lookback_days: int,
    topic_id: str | None = None,
    status: str | None = None,
    channel: str | None = None,
) -> int:
    query = """
        SELECT COUNT(*)
        FROM tickets t
        WHERE t.created_at >= now() - (:'days'::int * INTERVAL '1 day')
          AND t.created_at <= now()
          AND t.region_id = :'region'
          AND (NULLIF(:'topic', '') IS NULL OR t.topic_id = NULLIF(:'topic', ''))
          AND (NULLIF(:'status', '') IS NULL OR t.status = UPPER(NULLIF(:'status', '')))
          AND (NULLIF(:'channel', '') IS NULL OR t.channel = NULLIF(:'channel', ''));
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "sh",
            "-lc",
            'exec psql -XAtq -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"',
            "pulse109-e2e-lookback-check",
            "-v",
            f"days={lookback_days}",
            "-v",
            f"region={region_id}",
            "-v",
            f"topic={topic_id or ''}",
            "-v",
            f"status={status or ''}",
            "-v",
            f"channel={channel or ''}",
        ],
        cwd=repo_root,
        input=query,
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError as error:
        raise AcceptanceError(
            f"PostgreSQL returned invalid lookback ticket count: {result.stdout!r}"
        ) from error


def postgres_detector_history_counts(
    repo_root: Path,
    *,
    region_ids: tuple[str, ...],
    topic_id: str,
) -> dict[str, list[int]]:
    query = """
        WITH bounds AS (SELECT now() AS period_end),
        candidate_regions AS (
          SELECT unnest(string_to_array(:'regions', ',')) AS region_id
        ), periods AS (
          SELECT r.region_id, bucket.period_index,
            b.period_end - ((bucket.period_index + 1) * 7) * INTERVAL '1 day' AS period_start,
            b.period_end - (bucket.period_index * 7) * INTERVAL '1 day' AS period_end
          FROM candidate_regions r CROSS JOIN bounds b
          CROSS JOIN LATERAL generate_series(1, 4) AS bucket(period_index)
        ), period_counts AS (
          SELECT p.region_id, p.period_index, count(t.id)::bigint AS ticket_count
          FROM periods p
          LEFT JOIN tickets t ON t.region_id = p.region_id AND t.topic_id = :'topic'
            AND t.created_at >= p.period_start AND t.created_at < p.period_end
          GROUP BY p.region_id, p.period_index
        )
        SELECT region_id, array_to_string(array_agg(ticket_count ORDER BY period_index), ',')
        FROM period_counts
        GROUP BY region_id
        ORDER BY region_id;
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "sh",
            "-lc",
            'exec psql -XAtq -F "|" -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"',
            "pulse109-detector-history-check",
            "-v",
            f"regions={','.join(region_ids)}",
            "-v",
            f"topic={topic_id}",
        ],
        cwd=repo_root,
        input=query,
        text=True,
        capture_output=True,
        check=True,
    )
    counts_by_region: dict[str, list[int]] = {}
    try:
        for line in result.stdout.splitlines():
            region, counts = line.split("|", 1)
            counts_by_region[region] = [int(count) for count in counts.split(",")]
    except ValueError as error:
        raise AcceptanceError(
            f"PostgreSQL returned invalid detector baseline counts: {result.stdout!r}"
        ) from error
    return counts_by_region


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
    current_time = datetime.now(timezone.utc).replace(microsecond=0)
    created_at = current_time.isoformat().replace("+00:00", "Z")
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

    # Keep repeated acceptance runs isolated from active or closed incidents
    # inside the detector's seven-day cooldown window.
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
    cooldown_cutoff = current_time - timedelta(days=7)
    consumed_regions = set()
    for item in alert_list.get("items", []):
        incident_key = str(item.get("incident_key", ""))
        if not incident_key.startswith("topic_spike:") or ":water_supply:" not in incident_key:
            continue
        try:
            created_at_value = datetime.fromisoformat(
                str(item.get("created_at", "")).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if created_at_value >= cooldown_cutoff:
            consumed_regions.add(incident_key.split(":")[1])
    available_regions = tuple(
        candidate for candidate in region_candidates if candidate not in consumed_regions
    ) or region_candidates
    repo_root = Path(__file__).resolve().parent.parent
    existing_history = postgres_detector_history_counts(
        repo_root,
        region_ids=available_regions,
        topic_id="water_supply",
    )

    def required_current_count(region_id: str) -> int:
        # The four added baseline tickets make the fixture's coverage complete.
        counts = sorted(value + 1 for value in existing_history.get(region_id, [0, 0, 0, 0]))
        baseline = (counts[1] + counts[2]) / 2
        return max(3, math.floor(max(baseline, 1.0) * 1.5) + 1)

    region = min(available_regions, key=lambda value: (required_current_count(value), value))
    current_ticket_count = required_current_count(region)
    external_ids = [f"{source}-{index}" for index in range(current_ticket_count)]
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
    historical_tickets = [
        {
            "external_ticket_id": f"{source}-baseline-{index}",
            "region_id": region,
            "created_at": (current_time - timedelta(days=days)).isoformat().replace("+00:00", "Z"),
            "original_text": f"E2E baseline fixture {suffix}-{index}",
            "language": "ru",
            "topic_raw": "water_supply",
            "status": "OPEN",
            "channel": "e2e",
        }
        for index, days in enumerate((8, 15, 22, 29, 36))
    ]
    import_tickets = tickets + historical_tickets
    import_payload = {
        "source_system": source,
        "source_uri": f"memory://{source}.json",
        "dataset_version": dataset,
        "manifest_uri": f"memory://{dataset}/manifest.json",
        "manifest_sha256": "e2e-test-manifest",
        "is_synthetic": True,
        "tickets": import_tickets,
        "quarantine": quarantine,
    }

    status, _, imported = json_request(base_url, "POST", "/api/v1/import", body=import_payload, timeout=timeout)
    expect(status == 201, f"first import failed: {status} {imported}")
    expect(
        imported.get("imported_rows") == len(import_tickets)
        and imported.get("indexed_rows") == len(import_tickets),
        f"import counts: {imported}",
    )
    expect(imported.get("quarantined_rows") == 1, f"quarantine count: {imported}")
    status, _, forbidden_import = json_request(
        base_url,
        "POST",
        "/api/v1/import",
        body=import_payload,
        role="OPERATOR",
        timeout=timeout,
    )
    expect(status == 403, f"operator import should be forbidden: {status} {forbidden_import}")
    status, _, repeated = json_request(base_url, "POST", "/api/v1/import", body=import_payload, timeout=timeout)
    expect(status == 201, f"repeat import failed: {status} {repeated}")
    expect(
        repeated.get("imported_rows") == 0
        and repeated.get("duplicate_rows") == len(import_tickets)
        and repeated.get("indexed_rows") == 0,
        f"repeat import is not idempotent: {repeated}",
    )
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
    expect(
        status == 201 and linked.get("duplicate_rows") == len(import_tickets),
        f"new dataset could not link unchanged tickets: {status} {linked}",
    )

    status, _, listed = json_request(base_url, "GET", "/api/v1/tickets?limit=100", timeout=timeout)
    expect(status == 200, f"ticket list failed: {status} {listed}")
    selected = [item for item in listed.get("items", []) if item.get("external_ref") in external_ids]
    expect(len(selected) == len(external_ids), f"imported tickets not visible: {external_ids}")
    ticket_ids = [str(item["id"]) for item in selected]

    status, _, preview = json_request(base_url, "POST", "/api/v1/assist/preview", body={"ticket_id": ticket_ids[0]}, timeout=timeout)
    expect(status == 200 and preview.get("source") == "postgres-ticket+ml+qdrant", f"preview failed: {status} {preview}")
    prediction = preview.get("prediction", {})
    orchestration = preview.get("orchestration", {})
    confidence_states = {
        "confident",
        "uncertain",
        "low_confidence",
        "unavailable",
        "high",
        "medium",
        "low",
    }
    expect(
        prediction.get("confidence_state") in confidence_states
        and orchestration.get("needs_review") is True,
        f"preview must expose classification confidence and the review decision: {preview}",
    )
    related_ids = {
        str(item.get("ticket_id"))
        for item in preview.get("similar_tickets", [])
        if item.get("ticket_id") is not None
    }
    expect(
        related_ids.intersection(ticket_ids[1:]),
        f"preview did not retrieve a related E2E ticket: {preview.get('similar_tickets')}",
    )
    duplicate_ids = {
        str(item.get("ticket_id")) for item in preview.get("duplicate_candidates", [])
    }
    repeat_ids = {
        str(item.get("ticket_id")) for item in preview.get("repeat_candidates", [])
    }
    expect(
        bool(duplicate_ids.union(repeat_ids).intersection(related_ids))
        and duplicate_ids.issubset(related_ids)
        and repeat_ids.issubset(related_ids),
        "duplicate/repeat suggestions must be subsets of retrieved similar tickets",
    )
    response_template = preview.get("response_template", {})
    expect(
        response_template.get("approved") is False
        and response_template.get("source") == "MANUAL_REQUIRED"
        and not response_template.get("body"),
        f"unapproved copy must not be returned as a response template: {response_template}",
    )
    status, _, uncertain_preview = json_request(
        base_url,
        "POST",
        "/api/v1/assist/preview",
        body={"text": "ticket 123", "language": "UNKNOWN", "region_id": region},
        role="OPERATOR",
        timeout=timeout,
    )
    expect(
        status == 200
        and uncertain_preview.get("orchestration", {}).get("language") == "UNKNOWN"
        and uncertain_preview.get("orchestration", {}).get("needs_review") is True
        and uncertain_preview.get("response_template", {}).get("approved") is False,
        f"unknown-language preview must require operator review: {uncertain_preview}",
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

    region_filters = {
        "range": "30d",
        "region_id": region,
        "status": "OPEN",
        "channel": "e2e",
    }
    analytics_query = urlencode(region_filters)
    status, _, analytics = json_request(base_url, "GET", f"/api/v1/analytics?{analytics_query}", role="MANAGER", timeout=timeout)
    expect(status == 200, f"regional analytics failed: {analytics}")
    region_total = int(analytics.get("overview", {}).get("total_tickets", -1))
    database_region_total, database_previous_region_total = postgres_ticket_counts(
        repo_root, region_id=region, status="OPEN", channel="e2e"
    )
    region_previous_total = int(analytics.get("overview", {}).get("previous_total_tickets", -1))
    expect(
        region_total == database_region_total
        and region_previous_total == database_previous_region_total,
        f"regional analytics differs from PostgreSQL: API={region_total}/{region_previous_total} "
        f"DB={database_region_total}/{database_previous_region_total}",
    )
    region_bucket = next((item for item in analytics.get("by_region", []) if item.get("id") == region), None)
    expect(region_bucket is not None and region_bucket.get("tickets") == region_total, f"regional aggregate differs from the overview slice: {region_bucket} vs {region_total}")
    time_series_total = sum(int(point.get("tickets", 0)) for point in analytics.get("time_series", []))
    expect(time_series_total == region_total, f"regional time series differs from the overview slice: {time_series_total} vs {region_total}")

    drilldown_filters = {**region_filters, "dimension": "overview", "value": "all", "limit": "100"}
    drilldown_query = urlencode(drilldown_filters)
    status, _, drilldown = json_request(base_url, "GET", f"/api/v1/analytics/drilldown?{drilldown_query}", role="MANAGER", timeout=timeout)
    expect(status == 200 and drilldown.get("total") == region_total, f"regional drilldown differs from the aggregate slice: {drilldown}")

    selected_topic = next((item.get("id") for item in analytics.get("by_topic", []) if item.get("tickets", 0) > 0), None)
    expect(isinstance(selected_topic, str), f"regional analytics has no topic with tickets: {analytics.get('by_topic')}")
    topic_analytics_query = urlencode({**region_filters, "topic_id": selected_topic})
    status, _, topic_analytics = json_request(base_url, "GET", f"/api/v1/analytics?{topic_analytics_query}", role="MANAGER", timeout=timeout)
    expect(status == 200, f"region and topic analytics failed: {topic_analytics}")
    topic_total = int(topic_analytics.get("overview", {}).get("total_tickets", -1))
    database_topic_total, database_previous_topic_total = postgres_ticket_counts(
        repo_root,
        region_id=region,
        topic_id=selected_topic,
        status="OPEN",
        channel="e2e",
    )
    topic_previous_total = int(topic_analytics.get("overview", {}).get("previous_total_tickets", -1))
    expect(
        topic_total == database_topic_total
        and topic_previous_total == database_previous_topic_total,
        f"region/topic analytics differs from PostgreSQL: API={topic_total}/{topic_previous_total} "
        f"DB={database_topic_total}/{database_previous_topic_total}",
    )
    topic_drilldown_query = urlencode({**region_filters, "topic_id": selected_topic, "dimension": "topic", "value": selected_topic, "limit": "100"})
    status, _, topic_drilldown = json_request(base_url, "GET", f"/api/v1/analytics/drilldown?{topic_drilldown_query}", role="MANAGER", timeout=timeout)
    expect(status == 200 and topic_drilldown.get("total") == topic_total, f"region/topic drilldown differs from its aggregate slice: {topic_drilldown}")

    status, _, report = json_request(base_url, "GET", f"/api/v1/reports?{topic_analytics_query}", role="MANAGER", timeout=timeout)
    expect(status == 200 and report.get("source") == "postgres", f"filtered report slice failed: {status} {report}")
    report_slice = report.get("slice", {})
    report_analytics = report_slice.get("analytics", {})
    report_overview = report_analytics.get("overview", {})
    report_forecast = report_slice.get("forecast", {})
    report_filters = report_slice.get("filters", {})
    expect(
        report_filters.get("range") == "30d"
        and report_filters.get("region_id") == region
        and report_filters.get("topic_id") == selected_topic
        and report_filters.get("status") == "OPEN"
        and report_filters.get("channel") == "e2e",
        f"report slice lost selected filters: {report_filters}",
    )
    expect(
        int(report_overview.get("total_tickets", -1)) == database_topic_total
        and int(report_overview.get("previous_total_tickets", -1)) == database_previous_topic_total,
        f"report analytics differs from PostgreSQL: {report_overview} vs "
        f"{database_topic_total}/{database_previous_topic_total}",
    )
    report_topic = next(
        (item for item in report_analytics.get("by_topic", []) if item.get("id") == selected_topic),
        None,
    )
    expect(
        report_topic is not None and int(report_topic.get("tickets", -1)) == topic_total,
        f"report topic summary differs from analytics: {report_topic} vs {topic_total}",
    )
    forecast_lookback_total = postgres_ticket_count_lookback(
        repo_root,
        region_id=region,
        topic_id=selected_topic,
        status="OPEN",
        channel="e2e",
        lookback_days=366,
    )
    report_forecast_total = sum(
        int(point.get("tickets", 0)) for point in report_forecast.get("history", [])
    )
    expect(
        report_forecast_total == forecast_lookback_total,
        f"report forecast history differs from PostgreSQL: API={report_forecast_total} "
        f"DB={forecast_lookback_total}",
    )
    query_cases = (
        ("Сколько обращений за 30 дней?", "count", "none", "bar"),
        ("Какой тренд?", "trend", "day", "line"),
        ("Сравни регионы", "compare_regions", "region", "bar"),
        ("Топ тем", "top_topics", "topic", "bar"),
        ("Где всплески?", "spikes", "day", "line"),
        ("Дай прогноз", "forecast", "day", "line"),
    )
    for text, expected_intent, expected_grouping, expected_chart in query_cases:
        query_body = {
            "text": text,
            "filters": {
                "range": "30d",
                "region_id": region,
                "topic_id": selected_topic,
                "status": "OPEN",
                "channel": "e2e",
            },
        }
        if expected_intent == "forecast":
            query_body["horizon_days"] = 30
        status, _, result = json_request(
            base_url,
            "POST",
            "/api/v1/analytics/query",
            body=query_body,
            role="MANAGER",
            timeout=timeout,
        )
        expect(
            status == 200 and result.get("intent") == expected_intent,
            f"QueryIntent mapping failed for {text!r}: {result}",
        )
        interpreted_filters = result.get("interpreted_filters", {})
        expect(
            interpreted_filters.get("region_id") == region
            and interpreted_filters.get("topic_id") == selected_topic
            and interpreted_filters.get("status") == "OPEN"
            and interpreted_filters.get("channel") == "e2e"
            and interpreted_filters.get("range") == "30d"
            and interpreted_filters.get("group_by") == expected_grouping
            and (
                expected_intent != "forecast"
                or interpreted_filters.get("horizon_days") == 30
            ),
            f"QueryIntent lost or misinterpreted filters for {text!r}: {interpreted_filters}",
        )
        chart = result.get("chart", {})
        expect(
            (isinstance(result.get("number"), (int, float)) or result.get("number") is None)
            and "number" in result
            and result.get("summary", {}).get("value") == result.get("number")
            and isinstance(result.get("table"), list)
            and result.get("table") == result.get("rows")
            and isinstance(result.get("table_columns"), list)
            and bool(result.get("table_columns"))
            and isinstance(result.get("series"), list)
            and bool(result.get("series"))
            and chart.get("type") == expected_chart
            and chart.get("x") in {"date", "label"}
            and chart.get("y") == "count"
            and bool(chart.get("title")),
            f"QueryIntent lacks consistent number/table/chart output for {text!r}: {result}",
        )

    for horizon in (30, 60, 90):
        status, _, forecast = json_request(base_url, "GET", f"/api/v1/forecast?horizon={horizon}", role="MANAGER", timeout=timeout)
        expect(status == 200 and forecast.get("horizon_days") == horizon, f"forecast {horizon} failed: {forecast}")
        expect(forecast.get("history"), f"forecast {horizon} has no PostgreSQL history")
        expect(
            str(forecast.get("model_version", "")).startswith("forecast-")
            and forecast.get("model_version") != "embedder-demo-2026-09-21-001",
            f"forecast must expose its own model_version: {forecast}",
        )
        forecast_points = forecast.get("points", [])
        expected_peaks = forecast.get("expected_peaks", [])
        backtest = forecast.get("backtest", {})
        expect(
            forecast.get("status") == "OK"
            and forecast.get("insufficient_history") is False
            and len(forecast_points) == horizon
            and isinstance(expected_peaks, list)
            and set(expected_peaks).issubset(
                {str(point.get("date")) for point in forecast_points}
            ),
            f"forecast {horizon} did not return a complete forecast and peak dates: {forecast}",
        )
        expect(
            all(
                isinstance(backtest.get(metric), (int, float))
                and math.isfinite(float(backtest[metric]))
                and backtest[metric] >= 0
                for metric in ("mae", "rmse", "wape", "smape")
            )
            and backtest.get("sample_count", 0) > 0
            and backtest.get("window_count", 0) > 0,
            f"forecast {horizon} is missing rolling backtest metrics: {backtest}",
        )

    reforecast_request = {"horizon": 30}
    status, _, reforecast = json_request(
        base_url,
        "POST",
        "/api/v1/forecast/reforecast",
        role="MANAGER",
        body=reforecast_request,
        timeout=timeout,
    )
    expect(
        status == 200
        and reforecast.get("run_id")
        and reforecast.get("issued_at")
        and isinstance(reforecast.get("manager_signals"), list),
        f"rolling forecast was not persisted with signal history: {reforecast}",
    )
    status, _, repeated_reforecast = json_request(
        base_url,
        "POST",
        "/api/v1/forecast/reforecast",
        role="MANAGER",
        body=reforecast_request,
        timeout=timeout,
    )
    expect(
        status == 200
        and repeated_reforecast.get("run_id") == reforecast.get("run_id")
        and repeated_reforecast.get("issued_at") == reforecast.get("issued_at"),
        f"same-day rolling forecast request was not idempotent: {repeated_reforecast}",
    )

    status, _, detected = json_request(base_url, "POST", "/api/v1/alerts/detect", role="MANAGER", timeout=timeout)
    expect(
        status == 200
        and detected.get("status") in ("OK", "INSUFFICIENT_HISTORY")
        and detected.get("items"),
        f"spike detector found no alert: {detected}",
    )
    e2e_ticket_ids = set(ticket_ids)
    alert = next(
        (
            item
            for item in detected.get("items", [])
            if e2e_ticket_ids.intersection(map(str, item.get("linked_ticket_ids", [])))
        ),
        None,
    )
    expect(alert is not None, f"detector did not link the E2E source tickets: {detected}")
    source_ticket_ids = list(map(str, alert.get("linked_ticket_ids", [])))
    evidence = alert.get("detail", {})
    evidence_source_ticket_ids = list(map(str, evidence.get("source_ticket_ids", [])))
    expect(
        alert.get("current_count") == len(source_ticket_ids)
        and alert.get("ticket_count") == len(source_ticket_ids)
        and len(evidence_source_ticket_ids) == len(source_ticket_ids)
        and set(evidence_source_ticket_ids) == set(source_ticket_ids)
        and e2e_ticket_ids.issubset(set(source_ticket_ids)),
        f"alert evidence does not reproduce its source tickets: {alert}",
    )
    expect(
        alert.get("detector_version") == detected.get("detector_version")
        and evidence.get("configuration") == detected.get("config")
        and len(evidence.get("historical_counts", [])) == detected.get("config", {}).get("baseline_periods"),
        f"alert did not preserve its detector configuration and baseline: {alert}",
    )
    expect(
        alert.get("description") == "Зафиксирован необычный рост обращений. Требуется проверка."
        and alert.get("incident_key"),
        f"alert copy or incident key is invalid: {alert}",
    )
    expect(
        alert.get("status") == "OPEN"
        and bool(alert.get("title"))
        and bool(alert.get("severity"))
        and alert.get("region_id") == region
        and alert.get("topic_id") == "water_supply"
        and alert.get("baseline") is not None
        and isinstance(alert.get("linked_ticket_ids"), list),
        f"signal card is missing its safe display fields or source evidence: {alert}",
    )
    alert_id = str(alert["id"])
    status, _, repeated_detection = json_request(
        base_url, "POST", "/api/v1/alerts/detect", role="MANAGER", timeout=timeout
    )
    repeated_alert = next(
        (
            item
            for item in repeated_detection.get("items", [])
            if str(item.get("id")) == alert_id
        ),
        None,
    )
    expect(
        status == 200
        and repeated_alert is not None
        and repeated_alert.get("detail") == evidence
        and repeated_alert.get("linked_ticket_ids") == source_ticket_ids,
        f"detector cooldown changed persisted evidence: {repeated_detection}",
    )
    event_request = Request(
        base_url.rstrip("/") + "/api/v1/events",
        headers={"Accept": "text/event-stream", "X-Pulse-Role": "MANAGER", "X-User-Id": "e2e-acceptance"},
    )
    with urlopen(event_request, timeout=timeout) as event_response:
        expect(event_response.status == 200 and "text/event-stream" in event_response.headers.get("Content-Type", ""), "SSE endpoint did not return an event stream")
        snapshot = read_sse_event(event_response)
        expect(b"event: alerts.snapshot" in snapshot, f"SSE snapshot missing: {snapshot[:160]!r}")
        status, _, forbidden_ack = json_request(
            base_url,
            "POST",
            f"/api/v1/alerts/{alert_id}/ack",
            role="OPERATOR",
            timeout=timeout,
        )
        expect(status == 403, f"operator alert acknowledgement should be forbidden: {forbidden_ack}")
        monitoring_period_days = 3 * int(evidence["configuration"]["period_days"])
        status, _, forbidden_monitoring = json_request(
            base_url,
            "POST",
            f"/api/v1/alerts/{alert_id}/monitor",
            role="OPERATOR",
            body={"monitoring_period_days": monitoring_period_days},
            timeout=timeout,
        )
        expect(status == 403, f"operator should not start alert monitoring: {forbidden_monitoring}")
        status, _, monitoring = json_request(
            base_url,
            "POST",
            f"/api/v1/alerts/{alert_id}/monitor",
            role="MANAGER",
            body={"monitoring_period_days": monitoring_period_days},
            timeout=timeout,
        )
        expect(
            status == 200
            and monitoring.get("status") == "OPEN"
            and monitoring.get("monitoring", {}).get("state") == "MONITORING"
            and monitoring.get("monitoring", {}).get("monitoring_period_days") == monitoring_period_days
            and monitoring.get("detail") == evidence,
            f"manager monitoring did not preserve the alert status and evidence: {monitoring}",
        )
        monitoring_changed = read_sse_event(event_response)
        expect(b"event: alerts.changed" in monitoring_changed, f"SSE monitoring update missing: {monitoring_changed[:160]!r}")
        status, _, persisted_monitoring = json_request(
            base_url,
            "GET",
            f"/api/v1/alerts/{alert_id}",
            role="MANAGER",
            timeout=timeout,
        )
        expect(
            status == 200
            and persisted_monitoring.get("monitoring", {}).get("started_by") == "e2e-acceptance"
            and persisted_monitoring.get("monitoring", {}).get("state") == "MONITORING",
            f"manager monitoring was not persisted: {persisted_monitoring}",
        )
        status, _, acknowledged = json_request(base_url, "POST", f"/api/v1/alerts/{alert_id}/ack", role="MANAGER", timeout=timeout)
        expect(status == 200 and acknowledged.get("status") == "ACKNOWLEDGED", f"alert ack failed: {acknowledged}")
        changed = read_sse_event(event_response)
        expect(b"event: alerts.changed" in changed, f"SSE alert update missing: {changed[:160]!r}")
    status, _, closed = json_request(base_url, "POST", f"/api/v1/alerts/{alert_id}/close", role="MANAGER", timeout=timeout)
    expect(status == 200 and closed.get("status") == "CLOSED", f"alert close failed: {closed}")
    status, _, after_close_detection = json_request(
        base_url, "POST", "/api/v1/alerts/detect", role="MANAGER", timeout=timeout
    )
    expect(
        status == 200
        and all(str(item.get("id")) != alert_id for item in after_close_detection.get("items", [])),
        f"detector reopened a closed incident during cooldown: {after_close_detection}",
    )

    short_forecast_source = f"{source}-short-forecast"
    short_forecast_dataset = f"{dataset}-short-forecast"
    short_forecast_channel = f"e2e-short-forecast-{suffix}"
    short_forecast_ticket = {
        "external_ticket_id": f"{short_forecast_source}-0",
        "region_id": region,
        "created_at": created_at,
        "original_text": f"E2E {suffix}: короткая история для прогноза",
        "language": "ru",
        "topic_raw": "electricity",
        "status": "OPEN",
        "channel": short_forecast_channel,
    }
    status, _, short_forecast_import = json_request(
        base_url,
        "POST",
        "/api/v1/import",
        body={
            "source_system": short_forecast_source,
            "source_uri": f"memory://{short_forecast_source}.json",
            "dataset_version": short_forecast_dataset,
            "manifest_uri": f"memory://{short_forecast_dataset}/manifest.json",
            "manifest_sha256": "e2e-short-forecast-manifest",
            "is_synthetic": True,
            "tickets": [short_forecast_ticket],
        },
        timeout=timeout,
    )
    expect(
        status == 201 and short_forecast_import.get("imported_rows") == 1,
        f"short-history forecast fixture import failed: {short_forecast_import}",
    )
    short_forecast_query = urlencode(
        {
            "horizon": "30",
            "region_id": region,
            "topic_id": "electricity",
            "channel": short_forecast_channel,
        }
    )
    status, _, short_forecast = json_request(
        base_url,
        "GET",
        f"/api/v1/forecast?{short_forecast_query}",
        role="MANAGER",
        timeout=timeout,
    )
    expect(
        status == 200
        and short_forecast.get("status") == "INSUFFICIENT_HISTORY"
        and short_forecast.get("insufficient_history") is True
        and short_forecast.get("points") == []
        and short_forecast.get("expected_peaks") == []
        and short_forecast.get("backtest", {}).get("status") == "INSUFFICIENT_HISTORY",
        f"short forecast history did not remain explicitly insufficient: {short_forecast}",
    )

    for report_path, magic in (("/api/v1/analytics/export.pdf", b"%PDF-"), ("/api/v1/analytics/export.xlsx", b"PK\x03\x04")):
        filtered_path = f"{report_path}?{topic_analytics_query}"
        status, _, report = request(base_url, "GET", filtered_path, role="MANAGER", timeout=timeout)
        expect(status == 200 and report.startswith(magic), f"report failed: {filtered_path} HTTP {status}")
        if report_path.endswith("pdf"):
            expect(
                len(report) > 1_000 and b"%%EOF" in report[-1_024:],
                "PDF export is truncated or missing its final marker",
            )
        if report_path.endswith("xlsx"):
            temp_path = Path(os.getenv("TMPDIR", "/tmp")) / f"pulse109-{suffix}.xlsx"
            temp_path.write_bytes(report)
            try:
                with zipfile.ZipFile(temp_path) as archive:
                    bad = archive.testzip()
                    sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
                expect(bad is None, f"XLSX archive is corrupt: {bad}")
                namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                rows = []
                for row in sheet.findall(".//main:sheetData/main:row", namespace):
                    values = {}
                    for cell in row.findall("main:c", namespace):
                        column = "".join(character for character in cell.attrib["r"] if character.isalpha())
                        text_value = cell.find("main:is/main:t", namespace)
                        numeric_value = cell.find("main:v", namespace)
                        if text_value is not None:
                            values[column] = text_value.text or ""
                        elif numeric_value is not None:
                            values[column] = numeric_value.text or ""
                        else:
                            values[column] = ""
                    rows.append(values)
                headers = rows[0]
                columns = {value: key for key, value in headers.items()}
                topic_export = next(
                    (
                        row
                        for row in rows[1:]
                        if row.get(columns.get("record_type", "")) == "topic"
                        and row.get(columns.get("topic_id", "")) == selected_topic
                    ),
                    None,
                )
                filters_export = next(
                    (
                        row
                        for row in rows[1:]
                        if row.get(columns.get("record_type", "")) == "metadata"
                        and row.get(columns.get("metric", "")) == "filters"
                    ),
                    None,
                )
                expected_change = report_topic.get("change_abs")
                required_columns = {
                    "period",
                    "region_id",
                    "topic_id",
                    "record_type",
                    "value",
                    "previous_value",
                    "change_abs",
                    "change_pct",
                    "share_pct",
                    "generated_at",
                    "alert_type",
                    "severity",
                    "ticket_count",
                }
                expect(
                    required_columns.issubset(columns)
                    and topic_export is not None
                    and int(topic_export.get(columns["value"], "-1")) == topic_total
                    and math.isclose(float(topic_export.get(columns["share_pct"], "nan")), 100.0)
                    and expected_change is not None
                    and int(topic_export.get(columns["change_abs"], "-1")) == int(expected_change)
                    and int(topic_export.get(columns["previous_value"], "-1"))
                    == max(topic_total - int(expected_change), 0)
                    and filters_export is not None
                    and filters_export.get(columns["region_id"]) == region
                    and filters_export.get(columns["topic_id"]) == selected_topic
                    and f"region={region}" in filters_export.get(columns["details"], "")
                    and f"topic={selected_topic}" in filters_export.get(columns["details"], "")
                    and any(row.get(columns["generated_at"]) for row in rows[1:]),
                    "XLSX export omitted the selected topic or its matching numeric analytics values",
                )
            finally:
                temp_path.unlink(missing_ok=True)

    status, _, unauthorized = json_request(base_url, "GET", "/api/v1/analytics", role="OPERATOR", timeout=timeout)
    expect(status == 403, f"RBAC expected 403, got {status} {unauthorized}")
    status, _, reindex = json_request(base_url, "POST", "/api/v1/retrieval/reindex", role="MANAGER", timeout=timeout)
    expect(status == 202 and reindex.get("job_type") == "REINDEX_QDRANT", f"reindex queue failed: {reindex}")

    evaluation_dataset = f"{dataset}-evaluation"
    evaluation_source = f"{source}-evaluation"
    evaluation_ticket = {
        "external_ticket_id": f"{evaluation_source}-0",
        "region_id": region,
        "created_at": created_at,
        "original_text": f"E2E {suffix}: holdout обращение по водоснабжению",
        "language": "ru",
        "topic_raw": "water_supply",
        "status": "OPEN",
        "channel": "e2e",
    }
    status, _, evaluation_import = json_request(
        base_url,
        "POST",
        "/api/v1/import",
        body={
            "source_system": evaluation_source,
            "source_uri": f"memory://{evaluation_source}.json",
            "dataset_version": evaluation_dataset,
            "manifest_uri": f"memory://{evaluation_dataset}/manifest.json",
            "manifest_sha256": "e2e-evaluation-manifest",
            "is_synthetic": True,
            "tickets": [evaluation_ticket],
        },
        timeout=timeout,
    )
    expect(status == 201, f"frozen evaluation dataset import failed: {status} {evaluation_import}")

    status, _, cycle = json_request(
        base_url,
        "POST",
        "/api/v1/learning",
        body={
            "evaluation_dataset_version": evaluation_dataset,
            "candidate_model_version": f"classifier-{suffix}",
        },
        role="ML_REVIEWER",
        timeout=timeout,
    )
    expect(status == 201, f"learning cycle creation failed: {cycle}")
    cycle_id = str(cycle["id"])
    status, _, forbidden_learning = json_request(
        base_url,
        "GET",
        "/api/v1/learning",
        role="OPERATOR",
        timeout=timeout,
    )
    expect(
        status == 403,
        f"operator learning-cycle access should be forbidden: {forbidden_learning}",
    )
    status, _, feedback = json_request(
        base_url,
        "POST",
        f"/api/v1/assist/{ticket_ids[0]}/correct",
        body={
            "topic_id": "water_supply",
            "service": "service_water",
            "priority": "high",
            "note": "e2e learning correction",
        },
        role="OPERATOR",
        timeout=timeout,
    )
    expect(status == 200, f"learning feedback correction failed: {feedback}")
    status, _, closed_cycle = json_request(base_url, "POST", "/api/v1/learning/cycle/close", body={"cycle_id": cycle_id}, role="ML_REVIEWER", timeout=timeout)
    expect(status == 202 and closed_cycle.get("state") == "TRAINING", f"learning close failed: {closed_cycle}")
    fake_trainer = os.getenv("PULSE_TEST_FAKE_TRAINER", "false").strip().lower() in {"1", "true", "yes"}
    if not fake_trainer:
        learning_snapshot: dict[str, Any] = {}
        cycle_record: dict[str, Any] = {}
        for _ in range(40):
            time.sleep(0.5)
            status, _, learning_snapshot = json_request(
                base_url,
                "GET",
                "/api/v1/learning",
                role="ML_REVIEWER",
                timeout=timeout,
            )
            expect(status == 200, f"learning state unavailable after training: {learning_snapshot}")
            cycle_record = next(
                (item for item in learning_snapshot.get("items", []) if str(item.get("id")) == cycle_id),
                {},
            )
            if cycle_record.get("state") != "TRAINING":
                break
        if cycle_record.get("state") == "DATASET_BUILD_FAILED":
            production_model_id = (learning_snapshot.get("production_model") or {}).get("id")
            expect(
                cycle_record.get("decision_note") == "PRODUCTION_BASELINE_MISMATCH"
                and production_model_id == cycle.get("production_model_version"),
                f"model handoff failure was not fail-closed: cycle={cycle_record}, learning={learning_snapshot}",
            )
            return {
                "source_system": source,
                "ticket_ids": ticket_ids,
                "alert_id": alert_id,
                "learning": "BLOCKED_POST_HANDOFF_PRODUCTION_BASELINE_MISMATCH",
                "checks": "postgres+ml+qdrant+operator+similarity+analytics+query-intent+alerts+sse+forecast+reports+learning-fail-closed+rbac",
            }
        expect(
            cycle_record.get("state") == "EVALUATE",
            f"normal worker reached neither EVALUATE nor the documented post-handoff blocker: {cycle_record}",
        )
    evaluation: dict[str, Any] = {}
    for _ in range(40):
        time.sleep(0.5)
        status, _, evaluation = json_request(base_url, "GET", "/api/v1/learning/candidate/evaluation", role="ML_REVIEWER", timeout=timeout)
        expect(status == 200, f"candidate evaluation read failed: HTTP {status}")
        if evaluation.get("state") != "TRAINING":
            break
    expect(evaluation.get("state") == "EVALUATE", f"normal worker did not start the evaluation window: {evaluation}")

    if not fake_trainer:
        status, _, shadow_ticket = json_request(
            base_url,
            "POST",
            "/api/v1/tickets",
            body={
                "text": f"E2E {suffix}: нет воды в доме",
                "language": "RU",
                "region_id": region,
                "source": "e2e-shadow",
            },
            role="OPERATOR",
            timeout=timeout,
        )
        expect(status == 201, f"shadow ticket creation failed: HTTP {status}")
        expect(
            shadow_ticket.get("prediction", {}).get("model_version") == cycle.get("production_model_version"),
            "ticket recommendation did not remain on the production model",
        )
        shadow_ticket_id = str(shadow_ticket.get("ticket", {}).get("id", ""))
        expect(bool(shadow_ticket_id), "shadow ticket response has no ticket id")
        status, _, operator_decision = json_request(
            base_url,
            "POST",
            f"/api/v1/assist/{shadow_ticket_id}/confirm",
            body={},
            role="OPERATOR",
            timeout=timeout,
        )
        expect(
            status == 200 and operator_decision.get("decision", {}).get("action") == "confirm",
            f"shadow ticket decision failed: HTTP {status}",
        )
        status, _, confirmed_ticket = json_request(
            base_url,
            "GET",
            f"/api/v1/tickets/{shadow_ticket_id}",
            role="OPERATOR",
            timeout=timeout,
        )
        expect(
            status == 200
            and confirmed_ticket.get("latest_decision", {}).get("action") == "confirm",
            f"confirmed operator decision was not persisted: {confirmed_ticket}",
        )

        status, _, learning = json_request(base_url, "GET", "/api/v1/learning", role="ML_REVIEWER", timeout=timeout)
        expect(status == 200, f"learning cycle read failed: HTTP {status}")
        active_cycle = learning.get("active_cycle") or {}
        expect(active_cycle.get("state") == "EVALUATE", f"evaluation window closed before requested: {active_cycle}")
        expect(
            active_cycle.get("shadow_prediction_count", 0) >= 1
            and active_cycle.get("shadow_inference_failures", 0) == 0
            and active_cycle.get("shadow_operator_decision_count", 0) >= 1,
            f"fresh ticket did not produce a successful shadow prediction: {active_cycle}",
        )
        expect(
            active_cycle.get("blind_ab_enabled") is False,
            "blind A/B preference collection must remain disabled",
        )

    status, _, closed_evaluation = json_request(
        base_url,
        "POST",
        "/api/v1/learning/cycle/close",
        body={"cycle_id": cycle_id},
        role="ML_REVIEWER",
        timeout=timeout,
    )
    expect(
        status == 202
        and closed_evaluation.get("state") == "DECISION"
        and closed_evaluation.get("production_model_unchanged") is True,
        f"evaluation window did not close without changing production: {closed_evaluation}",
    )
    evaluation: dict[str, Any] = {}
    status = 0
    for _ in range(40):
        status, _, evaluation = json_request(
            base_url,
            "GET",
            "/api/v1/learning/candidate/evaluation",
            role="ML_REVIEWER",
            timeout=timeout,
        )
        expect(status == 200, f"closed candidate evaluation is unavailable: {evaluation}")
        if evaluation.get("status") != "PENDING":
            break
        time.sleep(0.5)
    expect(
        evaluation.get("cycle_id") == cycle_id
        and evaluation.get("status") in {"COMPLETED", "FAILED"},
        f"candidate evaluation job did not produce a terminal result: {evaluation}",
    )
    if fake_trainer:
        expect(
            evaluation.get("synthetic") is True
            and evaluation.get("decision") == "INSUFFICIENT_EVIDENCE",
            f"fake trainer evidence must remain ineligible for promotion: {evaluation}",
        )
    else:
        expect(
            evaluation.get("decision") != "PENDING_HUMAN_DECISION",
            f"candidate without complete verified evidence must not be promotable: {evaluation}",
        )
    status, _, learning_before_rejection = json_request(
        base_url,
        "GET",
        "/api/v1/learning",
        role="ML_REVIEWER",
        timeout=timeout,
    )
    expect(status == 200, f"learning state before rejection is unavailable: {learning_before_rejection}")
    production_before_rejection = (learning_before_rejection.get("production_model") or {}).get("id")
    expect(bool(production_before_rejection), "production model is missing before rejection")

    rejection_note = "e2e evaluation reviewed"
    status, _, rejected = json_request(base_url, "POST", "/api/v1/learning/candidate/reject", body={"note": rejection_note}, role="ML_REVIEWER", timeout=timeout)
    expect(
        status == 200
        and rejected.get("state") == "REJECTED"
        and rejected.get("decision_note") == rejection_note
        and rejected.get("production_model_version") == cycle.get("production_model_version"),
        f"candidate rejection failed or did not persist its note: {rejected}",
    )
    candidate_model_version = str(rejected.get("candidate_model_version") or "")
    candidate_dataset_version = str(rejected.get("dataset_version") or "")
    expect(
        candidate_model_version
        and candidate_model_version == evaluation.get("candidate_model_version")
        and candidate_dataset_version
        and candidate_dataset_version == evaluation.get("candidate_dataset_version"),
        f"candidate evaluation lineage does not match the rejected cycle: {evaluation}",
    )
    status, _, rejected_model = json_request(
        base_url,
        "GET",
        f"/api/v1/models/{candidate_model_version}",
        role="ML_REVIEWER",
        timeout=timeout,
    )
    expect(
        status == 200 and str(rejected_model.get("status", "")).upper() == "REJECTED",
        f"candidate model registry did not persist REJECTED status: {rejected_model}",
    )
    expect(
        rejected_model.get("dataset_version") == candidate_dataset_version,
        f"candidate model registry lost dataset lineage: {rejected_model}",
    )
    expected_model_family = (
        "deterministic-demo-candidate" if fake_trainer else "multinomial-naive-bayes"
    )
    expect(
        rejected_model.get("model_family") == expected_model_family,
        f"candidate was not produced by the expected training adapter: {rejected_model}",
    )
    status, _, learning_after_rejection = json_request(
        base_url,
        "GET",
        "/api/v1/learning",
        role="ML_REVIEWER",
        timeout=timeout,
    )
    production_after_rejection = (learning_after_rejection.get("production_model") or {}).get("id")
    expect(
        status == 200 and production_after_rejection == production_before_rejection,
        f"reject changed the production model pointer: {learning_after_rejection}",
    )
    learning_result = "FAKE_CANDIDATE_REJECTED" if fake_trainer else "CANDIDATE_EVIDENCE_VERIFIED_AND_REJECTED"

    return {
        "source_system": source,
        "ticket_ids": ticket_ids,
        "alert_id": alert_id,
        "learning": learning_result,
        "checks": "postgres+ml+qdrant+operator+similarity+analytics+query-intent+alerts+sse+forecast+reports+learning+rbac",
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
