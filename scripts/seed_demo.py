#!/usr/bin/env python3
"""Load the checked-in synthetic fixture into a fresh demo stack."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATA_ROOT = Path("/data")
MANIFEST_PATH = DATA_ROOT / "manifests/demo-2026-09-29.json"
CORE_URL = os.environ.get("CORE_URL", "http://core-api:8080").rstrip("/")


def wait_for_core() -> None:
    for _ in range(60):
        try:
            with urlopen(f"{CORE_URL}/readyz", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise RuntimeError("Core API did not become ready for demo seed")


def main() -> None:
    if os.environ.get("PULSE_ENV", "").lower() != "demo":
        raise RuntimeError("demo seed requires PULSE_ENV=demo")

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    fixture = next(item for item in manifest["files"] if item["kind"] == "normalized_tickets")
    tickets_path = DATA_ROOT / Path(fixture["path"]).relative_to("data")
    ticket_bytes = tickets_path.read_bytes()
    checksum = hashlib.sha256(ticket_bytes).hexdigest()
    if checksum != fixture["sha256"]:
        raise RuntimeError("demo tickets checksum differs from manifest")
    tickets = [json.loads(line) for line in ticket_bytes.splitlines() if line]
    if len(tickets) != fixture["records"] or len(tickets) != manifest["record_count"]:
        raise RuntimeError("demo tickets count differs from manifest")

    wait_for_core()
    payload = {
        "source_system": "pulse109_demo",
        "source_uri": "synthetic://pulse109/demo",
        "dataset_version": manifest["dataset_version"],
        "manifest_uri": "data/manifests/demo-2026-09-29.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "is_synthetic": True,
        "tickets": tickets,
        "quarantine": [],
    }
    request = Request(
        f"{CORE_URL}/api/v1/import",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Pulse-Role": "ADMIN",
            "X-User-Id": "demo-seed",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"demo seed import failed with HTTP {error.code}") from error
    if result["imported_rows"] + result["duplicate_rows"] != len(tickets):
        raise RuntimeError("demo seed import did not account for every ticket")
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "imported_rows": result["imported_rows"],
        "duplicate_rows": result["duplicate_rows"],
        "indexed_rows": result["indexed_rows"],
        "status": "ready",
    }))


if __name__ == "__main__":
    main()
