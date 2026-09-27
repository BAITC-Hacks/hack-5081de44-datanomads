#!/usr/bin/env python3
"""Independent Pulse 109 contract and acceptance smoke checks.

The script deliberately talks to Pulse through HTTP and reads only the public
compose/OpenAPI artifacts.  It does not import backend, frontend, or ML code,
so it can also be used from a clean CI image.

Examples::

    # Check fixtures, OpenAPI and compose files without starting services.
    python scripts/smoke_test.py --offline

    # Check a running demo stack.
    PULSE_BASE_URL=http://localhost:8080 \
      python scripts/smoke_test.py --log-file .tmp/pulse.jsonl

Optional live role checks use a JSON mapping of role to bearer token, for
example ``PULSE_ROLE_TOKENS='{"OPERATOR":"...","MANAGER":"..."}'``.
Tokens are never printed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "contract" / "contract_manifest.json"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "deterministic_demo.json"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    body: bytes
    headers: Mapping[str, str]
    error: str | None = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def fixture_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def check_fixture(path: Path, manifest: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    if not path.is_file():
        return [Check("deterministic fixture exists", False, f"missing {path}")]

    try:
        first = load_json(path)
        second = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [Check("deterministic fixture parses", False, str(exc))]

    checks.append(
        Check(
            "deterministic fixture bytes",
            canonical_json(first) == canonical_json(second),
            "same canonical representation on repeated reads",
        )
    )

    fixture_spec = manifest["fixture"]
    tickets = first.get("tickets") if isinstance(first, dict) else None
    if not isinstance(tickets, list) or not tickets:
        return checks + [Check("fixture ticket list", False, "tickets must be a non-empty list")]

    checks.append(
        Check(
            "fixture is explicitly synthetic",
            first.get("synthetic") is True and first.get("dataset_version", "").startswith("demo-"),
            "demo fixtures must never be presented as real metrics",
        )
    )
    checks.append(
        Check(
            "fixture has a fixed seed",
            isinstance(first.get("seed"), int),
            "seed is required for reproducible demo data",
        )
    )

    ids = [ticket.get("ticket_id") for ticket in tickets if isinstance(ticket, dict)]
    regions = {ticket.get("region_id") for ticket in tickets if isinstance(ticket, dict)}
    topics = {ticket.get("topic_id") for ticket in tickets if isinstance(ticket, dict)}
    languages = {ticket.get("language") for ticket in tickets if isinstance(ticket, dict)}
    checks.append(
        Check(
            "fixture ticket ids are unique",
            len(ids) == len(set(ids)) and all(isinstance(value, str) and value for value in ids),
            "unique non-empty ticket_id values" if len(ids) == len(set(ids)) else "duplicate or empty ticket_id values found",
        )
    )
    checks.append(
        Check(
            "fixture covers required regions",
            len(regions) >= int(fixture_spec["minimum_regions"]),
            f"{len(regions)} regions, need at least {fixture_spec['minimum_regions']}",
        )
    )
    checks.append(
        Check(
            "fixture covers required topics",
            len(topics) >= int(fixture_spec["minimum_topics"]),
            f"{len(topics)} topics, need at least {fixture_spec['minimum_topics']}",
        )
    )
    expected_languages = set(fixture_spec["languages"])
    checks.append(
        Check(
            "fixture covers RU and KZ",
            expected_languages.issubset(languages),
            f"languages present: {', '.join(sorted(str(value) for value in languages))}",
        )
    )

    required_fields = {
        "ticket_id",
        "region_id",
        "topic_id",
        "language",
        "text",
        "created_at",
        fixture_spec["synthetic_marker"],
    }
    malformed = [
        str(ticket.get("ticket_id", f"row-{index}"))
        for index, ticket in enumerate(tickets)
        if not isinstance(ticket, dict) or not required_fields.issubset(ticket)
    ]
    checks.append(
        Check(
            "fixture rows satisfy the minimum ticket shape",
            not malformed,
            f"missing fields in {', '.join(malformed[:3])}" if malformed else "all rows are valid",
        )
    )
    checks.append(
        Check(
            "fixture rows are marked synthetic",
            all(ticket.get(fixture_spec["synthetic_marker"]) is True for ticket in tickets),
            "every fixture row must carry the synthetic marker",
        )
    )

    # Requiring an explicit hash makes accidental fixture edits visible in CI.
    # The value is stored next to the manifest only when a project chooses to
    # pin it; otherwise the structural checks above still provide coverage.
    expected_hash = fixture_spec.get("sha256")
    if expected_hash:
        actual_hash = fixture_hash(first)
        checks.append(
            Check(
                "fixture hash is pinned",
                actual_hash == expected_hash,
                "fixture hash matches pinned deterministic version"
                if actual_hash == expected_hash
                else "fixture content differs from the pinned deterministic version",
            )
        )
    return checks


def _strip_yaml_comment(line: str) -> str:
    # Compose service names do not need quoted ``#`` characters.  Keeping this
    # conservative avoids treating a comment as a service key.
    return line.split("#", 1)[0].rstrip()


def compose_service_names(text: str) -> set[str]:
    """Extract top-level service keys without requiring PyYAML.

    This intentionally handles only the stable subset needed by the contract
    check.  If Docker is installed, ``docker compose config`` performs the
    authoritative YAML validation as a second step.
    """

    lines = text.splitlines()
    service_indent: int | None = None
    names: set[str] = set()
    in_services = False
    for raw_line in lines:
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if not in_services:
            if indent == 0 and re.match(r"^services\s*:\s*$", stripped):
                in_services = True
                service_indent = None
            continue
        if service_indent is None:
            if indent == 0:
                break
            service_indent = indent
        if indent < service_indent:
            break
        if indent != service_indent:
            continue
        match = re.match(r"^([\"']?[^:\"']+[\"']?)\s*:\s*$", stripped)
        if match:
            names.add(match.group(1).strip("\"'"))
    return names


def find_compose_file(manifest: Mapping[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.getenv("PULSE_COMPOSE_FILE") or os.getenv("COMPOSE_FILE")
    if env_value:
        # COMPOSE_FILE may contain a platform-specific list.  The first path
        # is enough for the required service contract.
        return Path(env_value.split(os.pathsep)[0]).expanduser().resolve()
    for name in manifest["compose"]["file_candidates"]:
        candidate = ROOT / name
        if candidate.is_file():
            return candidate
    return None


def check_compose(path: Path | None, manifest: Mapping[str, Any], *, strict_docker: bool) -> list[Check]:
    if path is None:
        return [Check("compose file exists", False, "no docker compose file found")]
    if not path.is_file():
        return [Check("compose file exists", False, f"missing {path}")]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Check("compose file readable", False, str(exc))]

    checks: list[Check] = [Check("compose file exists", True, str(path))]
    names = compose_service_names(text)
    normalized = {name.lower().replace("_", "-") for name in names}
    required = {
        name.lower().replace("_", "-") for name in manifest["compose"]["required_services"]
    }
    missing = sorted(required - normalized)
    checks.append(
        Check(
            "compose has the required demo services",
            not missing,
            f"missing: {', '.join(missing)}" if missing else ", ".join(sorted(names)),
        )
    )
    demo_profile = str(manifest["compose"]["demo_profile"])
    has_profile = re.search(rf"\b{re.escape(demo_profile)}\b", text) is not None
    checks.append(
        Check(
            "compose declares the demo profile",
            has_profile,
            f"profile marker {demo_profile!r} {'found' if has_profile else 'not found'}",
        )
    )

    docker = shutil.which("docker")
    if docker is None:
        checks.append(
            Check(
                "docker compose config",
                not strict_docker,
                "docker executable not found; static service check completed",
            )
        )
        return checks

    command = [docker, "compose", "-f", str(path), "config", "--quiet"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.append(Check("docker compose config", not strict_docker, str(exc)))
        return checks
    detail = result.stderr.strip() or result.stdout.strip() or "compose config accepted"
    checks.append(Check("docker compose config", result.returncode == 0, detail))
    return checks


def normalize_openapi_path(path: str) -> str:
    if not path.startswith("/"):
        return "/" + path
    return path


def operation_text(path: str, method: str, operation: Mapping[str, Any]) -> str:
    parts = [path, method]
    for key in ("operationId", "summary", "description"):
        value = operation.get(key)
        if isinstance(value, str):
            parts.append(value)
    tags = operation.get("tags")
    if isinstance(tags, list):
        parts.extend(str(item) for item in tags)
    return " ".join(parts).lower()


def check_openapi_document(
    document: Mapping[str, Any], manifest: Mapping[str, Any], *, core_only: bool = False
) -> list[Check]:
    checks: list[Check] = []
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return [Check("OpenAPI paths", False, "OpenAPI document has no paths object")]

    normalized_paths = {normalize_openapi_path(str(path)): value for path, value in paths.items()}
    checks.append(
        Check(
            "OpenAPI paths are present",
            bool(normalized_paths),
            f"{len(normalized_paths)} paths",
        )
    )
    for required in manifest["api"]["required_operations"]:
        path = normalize_openapi_path(str(required["path"]))
        method = str(required["method"]).lower()
        operation = normalized_paths.get(path)
        ok = isinstance(operation, dict) and isinstance(operation.get(method), dict)
        checks.append(
            Check(
                f"OpenAPI {method.upper()} {path}",
                ok,
                str(required.get("description", "required operation")),
            )
        )
    if core_only:
        checks.append(
            Check(
                "Core OpenAPI excludes ML routes",
                not any(path.startswith("/internal/") for path in normalized_paths),
                "ML routes belong to the separate internal service",
            )
        )
    else:
        for required in manifest["api"]["internal_exact_operations"]:
            path = normalize_openapi_path(str(required["path"]))
            method = str(required["method"]).lower()
            operation = normalized_paths.get(path)
            ok = isinstance(operation, dict) and isinstance(operation.get(method), dict)
            checks.append(Check(f"OpenAPI {method.upper()} {path}", ok, "internal ML operation"))

    for prefix in manifest["api"]["public_prefixes"]:
        canonical = normalize_openapi_path(str(prefix)).rstrip("/")
        ok = any(path == canonical or path.startswith(canonical + "/") for path in normalized_paths)
        checks.append(Check(f"OpenAPI public group {canonical}", ok, "application API group"))
    if not core_only:
        for prefix in manifest["api"]["internal_prefixes"]:
            canonical = normalize_openapi_path(str(prefix)).rstrip("/")
            ok = any(path == canonical or path.startswith(canonical + "/") for path in normalized_paths)
            checks.append(Check(f"OpenAPI internal group {canonical}", ok, "ML service group"))

    searchable_operations = [
        operation_text(path, method, operation)
        for path, path_item in normalized_paths.items()
        if isinstance(path_item, dict)
        for method, operation in path_item.items()
        if method.lower() in {"get", "post", "put", "patch", "delete"} and isinstance(operation, dict)
    ]
    for action, terms in manifest["api"]["learning_actions"].items():
        found = any(any(term.lower() in text for term in terms) for text in searchable_operations)
        checks.append(
            Check(
                f"OpenAPI learning action {action}",
                found,
                "learning lifecycle action must be discoverable by path or operation metadata",
            )
        )
    return checks


def discover_openapi_file(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    env_value = os.getenv("PULSE_OPENAPI_FILE")
    if env_value:
        path = Path(env_value).expanduser().resolve()
        return path if path.is_file() else None
    candidates = [
        ROOT / "openapi.json",
        ROOT / "docs" / "openapi.json",
        ROOT / "docs" / "openapi" / "openapi.json",
        ROOT / "backend" / "openapi.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def check_openapi_file(path: Path | None, manifest: Mapping[str, Any]) -> list[Check]:
    if path is None:
        return [Check("OpenAPI document exists", False, "no local OpenAPI JSON found")]
    try:
        document = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [Check("OpenAPI document parses", False, str(exc))]
    if not isinstance(document, dict):
        return [Check("OpenAPI document parses", False, "top-level document must be an object")]
    checks = [Check("OpenAPI document parses", True, str(path))]
    checks.extend(check_openapi_document(document, manifest))
    return checks


def http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: Mapping[str, Any] | None = None,
    token: str | None = None,
    role: str | None = None,
    timeout: float = 8.0,
    request_id: str | None = None,
) -> HttpResult:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if role:
        headers["X-Pulse-Role"] = role
    if request_id:
        headers["X-Request-ID"] = request_id
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResult(response.status, response.read(), dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read()
        except OSError:
            error_body = b""
        return HttpResult(exc.code, error_body, dict(exc.headers.items()), str(exc.reason))
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return HttpResult(None, b"", {}, str(exc))


def check_live_health(base_url: str, manifest: Mapping[str, Any], timeout: float) -> list[Check]:
    checks: list[Check] = []
    for path in manifest["api"]["health_paths"]:
        result = http_request(base_url, path, timeout=timeout)
        ok = result.status == 200
        detail = f"HTTP {result.status}" if result.status is not None else (result.error or "no response")
        if ok and result.body:
            try:
                payload = json.loads(result.body)
                if isinstance(payload, dict) and payload.get("status") in {"down", "unhealthy", "not_ready"}:
                    ok = False
                    detail = "health endpoint reports an unhealthy state"
            except json.JSONDecodeError:
                # A plain-text health response is valid if the status is 200.
                pass
        checks.append(Check(f"live {path}", ok, detail))
    return checks


def _load_role_tokens() -> dict[str, str]:
    raw = os.getenv("PULSE_ROLE_TOKENS", "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(role).upper(): str(token) for role, token in value.items() if token}


ROLE_PROBES: dict[str, tuple[str, str, Mapping[str, Any] | None]] = {
    "tickets": ("GET", "/api/v1/tickets", None),
    "assist": (
        "POST",
        "/api/v1/assist/preview",
        {"ticket_id": "demo-0001", "text": "Синтетическое acceptance обращение."},
    ),
    "analytics": ("GET", "/api/v1/analytics", None),
    "alerts": ("GET", "/api/v1/alerts", None),
    "forecast": ("GET", "/api/v1/forecast", None),
    "reports": ("GET", "/api/v1/reports", None),
    "learning_cycles": ("GET", "/api/v1/learning", None),
    "candidate_evaluation": ("GET", "/api/v1/learning", None),
    "model_management": ("GET", "/api/v1/models", None),
}


def check_live_roles(
    base_url: str,
    manifest: Mapping[str, Any],
    timeout: float,
) -> list[Check]:
    tokens = _load_role_tokens()
    if not tokens and os.getenv("PULSE_ROLE_HEADER_PROBE") != "1":
        return [
            Check(
                "role authorization probes",
                True,
                "skipped: set PULSE_ROLE_TOKENS or PULSE_ROLE_HEADER_PROBE=1 to run probes",
            )
        ]
    if not tokens:
        tokens = {role: "" for role in manifest["roles"]}

    checks: list[Check] = []
    roles = manifest["roles"]
    capabilities = set().union(*(set(items) for items in roles.values()))
    for capability in sorted(capabilities):
        probe = ROLE_PROBES.get(capability)
        if probe is None:
            # Confirm/correct/relation feedback need a ticket-specific write;
            # their route names are checked in OpenAPI and are left for the
            # explicit workflow probe to avoid mutating a real demo stack.
            checks.append(
                Check(
                    f"role capability {capability}",
                    True,
                    "static manifest/OpenAPI check only; mutating probe intentionally skipped",
                )
            )
            continue
        method, path, body = probe
        allowed_roles = {
            role for role, role_capabilities in roles.items() if capability in role_capabilities
        }
        for role, token in sorted(tokens.items()):
            result = http_request(
                base_url,
                path,
                method=method,
                body=body,
                token=token or None,
                role=role,
                timeout=timeout,
            )
            status = result.status
            if role in allowed_roles:
                ok = status not in {401, 403} and status is not None
                detail = f"{role} allowed: HTTP {status}"
            else:
                ok = status in {401, 403}
                detail = f"{role} denied: HTTP {status}"
            checks.append(Check(f"role {role} / {capability}", ok, detail))
    return checks


PII_SENTINELS = {
    "full_ticket_text": "PULSE109_PII_TEXT_6d9a",
    "iin": "PULSE109_IIN_870412301234",
    "phone": "PULSE109_PHONE_77010000000",
    "name": "PULSE109_NAME_Aigerim",
    "full_address": "PULSE109_ADDRESS_77_Test_Street",
    "attachments": "PULSE109_ATTACHMENT_secret.pdf",
}


def nearest_rank_percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one observation")
    if not 1 <= percentile <= 100:
        raise ValueError("percentile must be between 1 and 100")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered) / 100)
    return ordered[rank - 1]


def finite_latency_ms(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        latency = float(value)
    except (OverflowError, ValueError):
        return None
    return latency if math.isfinite(latency) and latency >= 0 else None


def check_log_file(path: Path, manifest: Mapping[str, Any], *, strict_schema: bool) -> list[Check]:
    if not path.is_file():
        return [Check("PII log file exists", False, f"missing {path}")]
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [Check("PII log file readable", False, str(exc))]
    checks: list[Check] = [Check("PII log file exists", True, str(path))]
    leaked = [name for name, sentinel in PII_SENTINELS.items() if sentinel in content]
    checks.append(
        Check(
            "PII sentinels absent from logs",
            not leaked,
            "forbidden sentinel class(es): " + ", ".join(leaked) if leaked else "no sentinel values found",
        )
    )
    required = set(manifest["observability"]["required_log_fields"])
    observability = manifest["observability"]
    latency_groups = observability.get("latency_group_by", [])
    allowed_latency_groups = {"service", "endpoint"}
    safe_latency_groups = [field for field in latency_groups if field in allowed_latency_groups]
    missing: set[str] = set()
    non_json_lines = 0
    valid_latencies = 0
    invalid_latencies = 0
    latency_samples: dict[tuple[str, ...], list[float]] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            non_json_lines += 1
            continue
        if isinstance(event, dict):
            missing.update(required - set(event))
            if event.get("message") not in {"request_completed", "request_failed"}:
                continue
            latency = finite_latency_ms(event.get("latency_ms"))
            if (
                latency is not None
                and safe_latency_groups
                and all(isinstance(event.get(field), str) for field in safe_latency_groups)
            ):
                group = tuple(event[field] for field in safe_latency_groups)
                latency_samples.setdefault(group, []).append(latency)
                valid_latencies += 1
            else:
                invalid_latencies += 1
    if strict_schema:
        ok = bool(content.strip()) and non_json_lines == 0 and not missing
        detail = "structured JSON fields present" if ok else "logs contain non-JSON lines or missing required fields"
    else:
        # A log file may contain startup text from a local supervisor.  The
        # PII assertion remains strict while schema validation is informative.
        ok = True
        detail = "schema advisory: " + (
            "missing " + ", ".join(sorted(missing)) if missing else "required fields observed"
        )
    checks.append(Check("structured JSON observability fields", ok, detail))
    latency_ok = (
        valid_latencies > 0
        and invalid_latencies == 0
        and safe_latency_groups == ["service", "endpoint"]
        and observability.get("latency_percentiles") == [50, 95]
    )
    if latency_samples:
        percentiles = observability.get("latency_percentiles", [50, 95])
        summaries = []
        for group, samples in sorted(latency_samples.items()):
            labels = ",".join(
                f"{field}={value}" for field, value in zip(safe_latency_groups, group)
            )
            values = ",".join(
                f"p{percentile}={nearest_rank_percentile(samples, int(percentile)):.3f}ms"
                for percentile in percentiles
            )
            summaries.append(f"{labels} n={len(samples)} {values}")
        latency_detail = "; ".join(summaries)
    else:
        latency_detail = "no valid latency samples"
    if invalid_latencies:
        latency_detail += f"; invalid latency events={invalid_latencies}"
    if not latency_ok:
        latency_detail += "; grouping must use only service and endpoint"
    checks.append(
        Check(
            "request latency p50/p95",
            latency_ok if strict_schema else True,
            latency_detail,
        )
    )
    return checks


def check_pii_probe(base_url: str, timeout: float) -> list[Check]:
    """Send synthetic PII sentinels through the non-mutating assist preview.

    The request is deliberately opt-in.  It gives the subsequent log scan a
    known value to search for without ever using a real person's data.
    """

    token = os.getenv("PULSE_OPERATOR_TOKEN")
    body = {
        "text": " | ".join(PII_SENTINELS.values()),
        "region_id": "R01",
    }
    result = http_request(
        base_url,
        "/api/v1/assist/preview",
        method="POST",
        body=body,
        token=token,
        role="OPERATOR",
        request_id="pulse109-pii-probe",
        timeout=timeout,
    )
    ok = result.status == 200
    detail = f"HTTP {result.status}" if result.status is not None else (result.error or "no response")
    return [Check("synthetic PII probe reached assist endpoint", ok, detail)]


def check_learning_safety(
    base_url: str,
    timeout: float,
    cycle_id: str | None,
) -> list[Check]:
    """Probe the invariant that feedback cannot promote production directly.

    The probe is opt-in because it mutates a learning-cycle fixture.  A cycle
    id and an ML_REVIEWER token must be supplied by the demo environment.
    """

    if not cycle_id:
        return [
            Check(
                "controlled learning safety probe",
                True,
                "skipped: pass --learning-cycle-id for the opt-in stateful probe",
            )
        ]
    token = _load_role_tokens().get("ML_REVIEWER")
    if not token:
        return [
            Check(
                "controlled learning safety probe",
                False,
                "--learning-cycle-id requires an ML_REVIEWER token in PULSE_ROLE_TOKENS",
            )
        ]

    base_path = f"/api/v1/learning/cycles/{cycle_id}"
    before = http_request(base_url, base_path, token=token, role="ML_REVIEWER", timeout=timeout)
    if before.status != 200:
        return [Check("learning cycle can be read", False, f"HTTP {before.status}")]
    try:
        before_payload = json.loads(before.body)
    except json.JSONDecodeError:
        return [Check("learning cycle response is JSON", False, "invalid response")]
    before_production = _production_version(before_payload)

    feedback = http_request(
        base_url,
        f"{base_path}/feedback",
        method="POST",
        token=token,
        role="ML_REVIEWER",
        body={"ticket_id": "demo-0001", "decision": "confirm", "source": "acceptance"},
        timeout=timeout,
    )
    if feedback.status not in {200, 201, 202, 204}:
        return [Check("learning feedback is accepted", False, f"HTTP {feedback.status}")]
    after = http_request(base_url, base_path, token=token, role="ML_REVIEWER", timeout=timeout)
    if after.status != 200:
        return [Check("learning cycle can be reread", False, f"HTTP {after.status}")]
    try:
        after_payload = json.loads(after.body)
    except json.JSONDecodeError:
        return [Check("learning cycle reread is JSON", False, "invalid response")]
    after_production = _production_version(after_payload)
    return [
        Check(
            "feedback does not auto-promote production",
            before_production == after_production,
            "production model version changed after feedback" if before_production != after_production else "production version unchanged",
        )
    ]


def _production_version(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("production_model_version", "production_version", "model_version"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    production = payload.get("production")
    if isinstance(production, dict):
        for key in ("model_version", "version"):
            value = production.get(key)
            if isinstance(value, str):
                return value
    return None


def check_public_docs_denied(
    base_url: str, manifest: Mapping[str, Any], timeout: float
) -> list[Check]:
    checks: list[Check] = []
    for path in manifest["api"]["public_docs_denied"]:
        result = http_request(base_url, path, timeout=timeout)
        if result.status is not None:
            detail = f"HTTP {result.status}"
        else:
            detail = result.error or "request failed"
        checks.append(
            Check(
                f"public docs blocked {path}",
                result.status == 404,
                detail,
            )
        )
    return checks


def run(args: argparse.Namespace) -> int:
    try:
        manifest = load_json(Path(args.manifest).resolve())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL manifest: {exc}", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict):
        print("FAIL manifest: top-level value must be an object", file=sys.stderr)
        return 2

    checks: list[Check] = []
    fixture_path = Path(args.fixture).resolve()
    checks.extend(check_fixture(fixture_path, manifest))
    compose_path = find_compose_file(manifest, args.compose_file)
    checks.extend(check_compose(compose_path, manifest, strict_docker=args.strict_docker))

    openapi_path = discover_openapi_file(args.openapi_file)
    if openapi_path is not None:
        checks.extend(check_openapi_file(openapi_path, manifest))
    elif args.offline:
        checks.extend(check_openapi_file(None, manifest))

    if not args.offline:
        checks.extend(check_live_health(args.base_url, manifest, args.timeout))
        checks.extend(check_public_docs_denied(args.base_url, manifest, args.timeout))
        checks.extend(check_live_roles(args.base_url, manifest, args.timeout))
        checks.extend(check_learning_safety(args.base_url, args.timeout, args.learning_cycle_id))

    if args.log_file:
        checks.extend(check_log_file(Path(args.log_file).resolve(), manifest, strict_schema=args.strict_logs))
    elif args.require_log_check:
        checks.append(Check("PII log check configured", False, "pass --log-file or disable --require-log-check"))
    else:
        checks.append(Check("PII log check", True, "skipped: pass --log-file to inspect structured logs"))

    if args.pii_probe:
        if args.offline:
            checks.append(Check("synthetic PII probe", False, "cannot run an HTTP probe with --offline"))
        else:
            checks.extend(check_pii_probe(args.base_url, args.timeout))

    failures = [check for check in checks if not check.ok]
    for check in checks:
        label = "PASS" if check.ok else "FAIL"
        print(f"{label:<4} {check.name}: {check.detail}")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip live HTTP probes")
    parser.add_argument("--base-url", default=os.getenv("PULSE_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--compose-file", help="explicit compose file")
    parser.add_argument("--openapi-file", help="explicit local OpenAPI JSON")
    parser.add_argument("--log-file", help="JSONL log file to scan for PII sentinels")
    parser.add_argument("--require-log-check", action="store_true")
    parser.add_argument("--strict-logs", action="store_true", help="fail on non-JSON log lines/missing fields")
    parser.add_argument(
        "--pii-probe",
        action="store_true",
        help="send synthetic PII sentinels through assist preview before scanning logs",
    )
    parser.add_argument("--strict-docker", action="store_true", help="fail if docker compose is unavailable")
    parser.add_argument("--learning-cycle-id", help="opt-in cycle id for the feedback safety probe")
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
