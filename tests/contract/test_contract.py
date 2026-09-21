"""Fast, dependency-free checks for the externally visible Pulse contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
