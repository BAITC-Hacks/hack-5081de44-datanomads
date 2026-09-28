"""Fast, dependency-free checks for the externally visible Pulse contract."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "contract" / "contract_manifest.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "deterministic_demo.json"
SMOKE_PATH = ROOT / "scripts" / "smoke_test.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("pulse109_smoke_test", SMOKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SMOKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContractArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = load_smoke_module()
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            cls.manifest = json.load(handle)

    def assert_checks_pass(self, checks):
        failed = [f"{check.name}: {check.detail}" for check in checks if not check.ok]
        self.assertFalse(failed, "\n".join(failed))

    def test_manifest_declares_required_roles(self):
        self.assertEqual(
            set(self.manifest["roles"]),
            {"OPERATOR", "MANAGER", "ML_REVIEWER", "ADMIN"},
        )
        self.assertIn("confirm", self.manifest["roles"]["OPERATOR"])
        self.assertIn("promote", self.manifest["roles"]["ML_REVIEWER"])
        self.assertIn("promote", self.manifest["roles"]["ADMIN"])
        self.assertNotIn("promote", self.manifest["roles"]["OPERATOR"])

    def test_manifest_declares_privacy_and_observability_contract(self):
        observability = self.manifest["observability"]
        required = set(observability["required_log_fields"])
        self.assertTrue({"request_id", "trace_id", "latency_ms", "status"}.issubset(required))
        forbidden = set(observability["forbidden_payload_fields"])
        self.assertTrue({"iin", "phone", "name", "full_address", "attachments"}.issubset(forbidden))
        self.assertEqual(observability["latency_percentiles"], [50, 95])
        self.assertEqual(observability["latency_group_by"], ["service", "endpoint"])
        self.assertTrue(
            {"ticket_id", "user_id", "request_id", "trace_id"}.issubset(
                set(observability["forbidden_metric_labels"])
            )
        )

    def test_latency_percentile_uses_nearest_rank(self):
        self.assertEqual(self.smoke.nearest_rank_percentile([4.0, 1.0, 3.0, 2.0], 50), 2.0)
        self.assertEqual(self.smoke.nearest_rank_percentile([4.0, 1.0, 3.0, 2.0], 95), 4.0)
        self.assertIsNone(self.smoke.finite_latency_ms(True))
        self.assertIsNone(self.smoke.finite_latency_ms(-0.1))
        self.assertEqual(self.smoke.finite_latency_ms(0), 0.0)

    def test_log_check_reports_latency_percentiles(self):
        fields = {
            "timestamp": "2026-09-27T00:00:00Z",
            "level": "INFO",
            "request_id": "internal-1",
            "trace_id": "trace-safe",
            "service": "pulse109-core",
            "endpoint": "/api/v1/tickets",
            "latency_ms": 12.5,
            "model_version": "n/a",
            "status": 200,
            "error_code": "none",
            "message": "request_completed",
        }
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            events = [dict(fields, latency_ms=latency) for latency in (12.5, 20.0)]
            log_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)
        latency_check = next(check for check in checks if check.name == "request latency p50/p95")
        self.assertTrue(latency_check.ok, latency_check.detail)
        self.assertIn("p50=12.500ms", latency_check.detail)
        self.assertIn("p95=20.000ms", latency_check.detail)

    def test_log_schema_checks_request_events_not_startup_records(self):
        fields = {
            "timestamp": "2026-09-27T00:00:00Z",
            "level": "INFO",
            "request_id": "internal-1",
            "trace_id": "trace-safe",
            "service": "pulse109-core",
            "endpoint": "/api/v1/tickets",
            "latency_ms": 12.5,
            "model_version": "n/a",
            "status": 200,
            "error_code": "none",
            "message": "request_completed",
        }
        startup = {"message": "service_ready", "service": "pulse109-core"}
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(event) for event in (startup, fields)),
                encoding="utf-8",
            )
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)

        schema_check = next(check for check in checks if check.name == "structured JSON observability fields")
        self.assertTrue(schema_check.ok, schema_check.detail)
        self.assertIn("1 request events", schema_check.detail)

    def test_log_schema_rejects_request_events_missing_required_fields(self):
        event = {
            "message": "request_completed",
            "latency_ms": 10.0,
            "service": "pulse109-core",
            "endpoint": "/api/v1/tickets",
        }
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            log_path.write_text(json.dumps(event), encoding="utf-8")
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)

        schema_check = next(check for check in checks if check.name == "structured JSON observability fields")
        self.assertFalse(schema_check.ok)
        self.assertIn("missing required fields", schema_check.detail)

    def test_strict_log_schema_allows_only_known_uvicorn_startup_lines(self):
        fields = {
            "timestamp": "2026-09-27T00:00:00Z",
            "level": "INFO",
            "request_id": "internal-1",
            "trace_id": "trace-safe",
            "service": "pulse109-ml",
            "endpoint": "/internal/v1/classify",
            "latency_ms": 12.5,
            "model_version": "classifier-demo-v1",
            "status": 200,
            "error_code": "none",
            "message": "request_completed",
        }
        startup_lines = (
            "INFO:     Started server process [1]",
            "INFO:     Waiting for application startup.",
            "INFO:     Application startup complete.",
            "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            log_path.write_text(
                "\n".join((*startup_lines, json.dumps(fields))),
                encoding="utf-8",
            )
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)

        schema_check = next(check for check in checks if check.name == "structured JSON observability fields")
        self.assertTrue(schema_check.ok, schema_check.detail)
        self.assertIn("allowed Uvicorn startup lines=4", schema_check.detail)

    def test_strict_log_schema_rejects_unrecognized_plain_text(self):
        fields = {
            "timestamp": "2026-09-27T00:00:00Z",
            "level": "INFO",
            "request_id": "internal-1",
            "trace_id": "trace-safe",
            "service": "pulse109-core",
            "endpoint": "/api/v1/tickets",
            "latency_ms": 12.5,
            "model_version": "n/a",
            "status": 200,
            "error_code": "none",
            "message": "request_completed",
        }
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            log_path.write_text("unexpected plain text\n" + json.dumps(fields), encoding="utf-8")
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)

        schema_check = next(check for check in checks if check.name == "structured JSON observability fields")
        self.assertFalse(schema_check.ok)
        self.assertIn("non-JSON lines", schema_check.detail)

    def test_log_check_rejects_nonfinite_latency(self):
        event = {
            "timestamp": "2026-09-27T00:00:00Z",
            "level": "INFO",
            "request_id": "internal-1",
            "trace_id": "trace-safe",
            "service": "pulse109-core",
            "endpoint": "/api/v1/tickets",
            "latency_ms": float("inf"),
            "model_version": "n/a",
            "status": 200,
            "error_code": "none",
            "message": "request_completed",
        }
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"
            log_path.write_text(json.dumps(event), encoding="utf-8")
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=True)
        latency_check = next(check for check in checks if check.name == "request latency p50/p95")
        self.assertFalse(latency_check.ok)
        self.assertIn("invalid latency events=1", latency_check.detail)

    def test_compose_log_capture_saves_output_for_pii_scan(self):
        sentinel = self.smoke.PII_SENTINELS["full_ticket_text"]
        log_output = json.dumps({"message": "request_completed", "text": sentinel}) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "compose.log"
            with patch.object(self.smoke.shutil, "which", return_value="/usr/bin/docker"):
                with patch.object(
                    self.smoke.subprocess,
                    "run",
                    return_value=self.smoke.subprocess.CompletedProcess(
                        ["docker"], 0, log_output, ""
                    ),
                ) as run_compose:
                    captured = self.smoke.capture_compose_logs(log_path, "demo")
            self.assertTrue(captured.ok, captured.detail)
            self.assertEqual(log_path.read_text(encoding="utf-8"), log_output)
            run_compose.assert_called_once()
            checks = self.smoke.check_log_file(log_path, self.manifest, strict_schema=False)
        pii_check = next(check for check in checks if check.name == "PII sentinels absent from logs")
        self.assertFalse(pii_check.ok)
        self.assertIn("full_ticket_text", pii_check.detail)

    def test_live_pii_probe_runs_before_compose_log_scan(self):
        args = SimpleNamespace(
            manifest=str(MANIFEST_PATH),
            fixture=str(FIXTURE_PATH),
            compose_file=None,
            strict_docker=False,
            openapi_file=None,
            offline=False,
            base_url="http://127.0.0.1:8080",
            timeout=1.0,
            learning_cycle_id=None,
            pii_probe=True,
            capture_compose_logs=True,
            log_file="/tmp/pulse109-acceptance.log",
            strict_logs=False,
            require_log_check=True,
        )
        calls = []
        passing_check = self.smoke.Check("test", True, "ok")

        def record(name, result):
            def check(*_args, **_kwargs):
                calls.append(name)
                return result

            return check

        with patch.object(self.smoke, "check_fixture", return_value=[]):
            with patch.object(self.smoke, "find_compose_file", return_value=None):
                with patch.object(self.smoke, "check_compose", return_value=[]):
                    with patch.object(self.smoke, "discover_openapi_file", return_value=None):
                        with patch.object(self.smoke, "check_live_health", return_value=[]):
                            with patch.object(self.smoke, "check_public_docs_denied", return_value=[]):
                                with patch.object(self.smoke, "check_live_roles", return_value=[]):
                                    with patch.object(self.smoke, "check_learning_safety", return_value=[]):
                                        with patch.object(
                                            self.smoke,
                                            "check_pii_probe",
                                            side_effect=record("probe", [passing_check]),
                                        ):
                                            with patch.object(
                                                self.smoke,
                                                "capture_compose_logs",
                                                side_effect=record("capture", passing_check),
                                            ):
                                                with patch.object(
                                                    self.smoke,
                                                    "check_log_file",
                                                    side_effect=record("scan", [passing_check]),
                                                ):
                                                    with redirect_stdout(io.StringIO()):
                                                        status = self.smoke.run(args)

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["probe", "capture", "scan"])

    def test_deterministic_fixture(self):
        self.assert_checks_pass(self.smoke.check_fixture(FIXTURE_PATH, self.manifest))

    def test_compose_contract_when_compose_exists(self):
        compose_path = self.smoke.find_compose_file(self.manifest, None)
        if compose_path is None:
            self.skipTest("compose is supplied by the integration build")
        self.assert_checks_pass(
            self.smoke.check_compose(compose_path, self.manifest, strict_docker=False)
        )

    def test_openapi_contract_when_document_exists(self):
        openapi_path = self.smoke.discover_openapi_file(None)
        if openapi_path is None:
            self.skipTest("OpenAPI artifact is supplied by the integration build")
        self.assert_checks_pass(self.smoke.check_openapi_file(openapi_path, self.manifest))

    def test_core_openapi_does_not_claim_internal_ml_routes(self):
        umbrella = json.loads((ROOT / "docs/openapi/openapi.json").read_text())
        self.assertIn("503", umbrella["paths"]["/readyz"]["get"]["responses"])
        core = dict(umbrella)
        core["paths"] = {
            path: operations
            for path, operations in umbrella["paths"].items()
            if not path.startswith("/internal/")
        }
        self.assert_checks_pass(self.smoke.check_openapi_document(core, self.manifest, core_only=True))


if __name__ == "__main__":
    unittest.main()
