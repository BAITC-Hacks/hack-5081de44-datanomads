from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx
import yaml

from app import main as ml_main
from app.services import ModelRegistry
from app.schemas import ModelMetadata
from contracts import ContractValidationError, validate_document
from contracts.validate import validate_demo_artifacts


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml-service/artifacts/manifest.json"
ML_OPENAPI = ROOT / "docs/openapi/ml.openapi.yaml"
HTTP_METHODS = {
    "get",
    "put",
    "post",
    "delete",
    "options",
    "head",
    "patch",
    "trace",
}


def openapi_operations(document: dict) -> set[tuple[str, str]]:
    return {
        (path, method)
        for path, path_item in document["paths"].items()
        for method in path_item
        if method.lower() in HTTP_METHODS
    }


class SharedContractTests(unittest.TestCase):
    def test_uvicorn_access_logs_are_disabled_by_default(self) -> None:
        dockerfile = (ROOT / "ml-service/Dockerfile").read_text(encoding="utf-8")
        readme = (ROOT / "ml-service/README.md").read_text(encoding="utf-8")
        self.assertIn("--no-access-log", dockerfile)
        self.assertIn("--no-access-log", readme)

    def test_demo_data_and_model_artifacts_validate(self) -> None:
        checked = validate_demo_artifacts()
        self.assertEqual(
            set(checked),
            {
                "DatasetManifest",
                "UnifiedTicket records",
                "ModelManifest",
                "ClassifierManifest",
                "EmbedderManifest",
                "ModelEvaluation",
                "LearningFeedbackExport",
                "CandidateDatasetBuildRequest",
                "CandidateDatasetManifest",
                "CandidateTrainingJob",
                "CandidateTrainingResult",
                "CandidateEvaluation",
            },
        )

    def test_contract_package_cli_validates_without_frontend_or_backend(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "contracts.validate", "--check-demo"],
            cwd=ROOT / "ml-service",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated:", result.stdout)

    def test_training_artifact_metadata_accepts_real_versions_and_checksums(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]["classifier"]
        payload.update(
            {
                "schema_version": "classifier-manifest.v1",
                "model_version": "classifier-xlm-roberta-2026-09-27-001",
                "model_family": "xlm-roberta-classifier",
                "base_model": "FacebookAI/xlm-roberta-base",
                "status": "CANDIDATE",
                "artifact_kind": "TRAINED_ARTIFACT",
                "artifact_uri": "file:///models/classifier/model",
                "artifact_checksum": "sha256:" + "a" * 64,
                "demo_artifact_id": None,
                "synthetic": False,
                "evaluation_version": "eval-2026-09-27.v1",
            }
        )
        payload.pop("implementation", None)
        validate_document(payload, "ClassifierManifest")
        model = ModelMetadata.model_validate(payload)
        self.assertEqual(model.model_version, "classifier-xlm-roberta-2026-09-27-001")
        self.assertEqual(model.base_model, "FacebookAI/xlm-roberta-base")

    def test_invalid_trained_artifact_checksum_is_rejected(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]["classifier"]
        payload.update(
            {
                "artifact_kind": "TRAINED_ARTIFACT",
                "status": "CANDIDATE",
                "artifact_uri": "file:///models/classifier/model",
                "artifact_checksum": "demo-baseline:must-not-pass-as-trained",
            }
        )
        with self.assertRaises(ContractValidationError):
            validate_document(payload, "ClassifierManifest")

    def test_model_status_is_required_for_artifact_lifecycle(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]["classifier"]
        payload.pop("status")
        with self.assertRaises(ContractValidationError):
            validate_document(payload, "ClassifierManifest")

    def test_model_metrics_reject_free_form_text(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]["classifier"]
        payload["metrics"] = {"operator_note": "ФИО: Иван Иванов"}
        with self.assertRaises(ContractValidationError):
            validate_document(payload, "ClassifierManifest")

    def test_ml_openapi_matches_runtime_manifest_schemas(self) -> None:
        published = yaml.safe_load(ML_OPENAPI.read_text(encoding="utf-8"))
        runtime_openapi = ml_main.app.openapi()
        runtime = runtime_openapi["components"]["schemas"]
        documented = published["components"]["schemas"]
        for schema_name in ("ModelManifestResponse", "ModelMetadata"):
            self.assertEqual(documented[schema_name], runtime[schema_name])
        for path in ("/internal/v1/classify", "/internal/v1/embed"):
            runtime_headers = {
                parameter["name"]
                for parameter in runtime_openapi["paths"][path]["post"]["parameters"]
                if parameter["in"] == "header"
            }
            documented_headers = {
                parameter["name"]
                for parameter in published["paths"][path]["post"]["parameters"]
                if parameter["in"] == "header"
            }
            self.assertEqual(documented_headers, runtime_headers)

    def test_ml_openapi_documents_every_runtime_route(self) -> None:
        published = yaml.safe_load(ML_OPENAPI.read_text(encoding="utf-8"))
        runtime = ml_main.app.openapi()
        self.assertEqual(
            openapi_operations(runtime),
            openapi_operations(published),
        )

    def test_documentation_is_available_only_on_loopback_service_ports(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        self.assertEqual(
            compose["services"]["core-api"]["ports"],
            ["127.0.0.1:${PULSE_CORE_HTTP_PORT:-8081}:8080"],
        )
        self.assertEqual(
            compose["services"]["ml-service"]["ports"],
            ["127.0.0.1:${ML_HTTP_PORT:-8000}:8000"],
        )
        self.assertEqual(ml_main.app.docs_url, "/docs")
        self.assertEqual(ml_main.app.redoc_url, "/redoc")
        self.assertEqual(ml_main.app.openapi_url, "/openapi.json")

    def test_demo_bootstrap_waits_for_readiness_and_seed_completion(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]
        healthcheck = services["core-api"]["healthcheck"]
        healthcheck_command = " ".join(str(part) for part in healthcheck["test"])
        self.assertIn("/readyz", healthcheck_command)
        self.assertEqual(
            services["demo-seed"]["depends_on"]["core-api"]["condition"],
            "service_healthy",
        )
        self.assertIn("demo", services["nginx"]["profiles"])
        for service_name in ("ml-worker", "nginx"):
            with self.subTest(service=service_name):
                dependencies = services[service_name]["depends_on"]
                self.assertEqual(
                    dependencies["demo-seed"]["condition"],
                    "service_completed_successfully",
                )
                self.assertEqual(
                    dependencies["core-api"]["condition"],
                    "service_healthy",
                )

    def test_model_manifest_path_is_shared_and_defaults_to_bundled_baseline(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        baseline_path = "${PULSE_MODEL_MANIFEST_PATH:-/app/artifacts/manifest.json}"
        services = compose["services"]
        for service_name in ("ml-service", "ml-worker"):
            with self.subTest(service=service_name):
                environment = services[service_name]["environment"]
                self.assertEqual(environment["PULSE_ENV"], "${PULSE_ENV:-demo}")
                self.assertEqual(environment["PULSE_MODEL_MANIFEST_PATH"], baseline_path)
        manifest = json.loads((ROOT / "ml-service/artifacts/manifest.json").read_text())
        self.assertTrue(
            all(model["status"] == "DEMO_BASELINE" for model in manifest["models"].values())
        )

    def test_model_registry_uses_an_explicit_manifest_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_bytes(MANIFEST.read_bytes())
            with patch.dict(
                os.environ,
                {"PULSE_MODEL_MANIFEST_PATH": str(manifest_path)},
            ):
                registry = ModelRegistry(runtime_mode="demo")
        self.assertEqual(registry.path, manifest_path)
        self.assertTrue(registry.ready, registry.load_error)

    def test_schema_errors_do_not_include_rejected_values(self) -> None:
        with self.assertRaises(ContractValidationError) as raised:
            validate_document(
                {"schema_version": "sensitive-free-form-value"},
                "CandidateEvaluation",
            )
        self.assertNotIn("sensitive-free-form-value", str(raised.exception))

    def test_corrupt_or_missing_manifest_fails_closed_without_baseline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            registry = ModelRegistry(path=missing, runtime_mode="demo")
            self.assertFalse(registry.ready)
            self.assertIsNone(registry.manifest)
            self.assertEqual(registry.model_versions(), {})

            corrupt = Path(directory) / "corrupt.json"
            corrupt.write_text("{not-json", encoding="utf-8")
            registry = ModelRegistry(path=corrupt, runtime_mode="demo")
            self.assertFalse(registry.ready)
            self.assertIsNone(registry.manifest)
            self.assertEqual(registry.model_versions(), {})

    def test_unready_registry_returns_readiness_error_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(path=Path(directory) / "missing.json", runtime_mode="demo")

            async def requests() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(app=ml_main.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    readiness = await client.get("/readyz")
                    inference = await client.post(
                        "/internal/v1/classify",
                        json={"text": "Нет воды"},
                    )
                return readiness, inference

            with patch.object(ml_main, "registry", registry):
                readiness, inference = asyncio.run(requests())

        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(inference.status_code, 503)
        self.assertEqual(readiness.json()["status"], "not_ready")
        self.assertEqual(readiness.json()["detail"], "model runtime is not ready")
        self.assertEqual(
            readiness.json()["checks"]["model_manifest"]["error_code"],
            "MODEL_MANIFEST_UNAVAILABLE_OR_INVALID",
        )

    def test_production_mode_is_unready_until_a_real_runtime_adapter_exists(self) -> None:
        registry = ModelRegistry(path=MANIFEST, runtime_mode="production")
        self.assertFalse(registry.ready)
        self.assertEqual(registry.load_error, "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED")
        self.assertEqual(
            registry.get("classifier").artifact_kind,
            "DETERMINISTIC_BASELINE",
        )

        async def readiness_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=ml_main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/readyz")

        with patch.object(ml_main, "registry", registry):
            readiness = asyncio.run(readiness_request())
        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(readiness.json()["status"], "not_ready")
        self.assertEqual(
            readiness.json()["checks"]["model_artifact"]["error_code"],
            "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED",
        )

    def test_demo_mode_does_not_report_a_trained_manifest_as_baseline_serving(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        classifier = payload["models"]["classifier"]
        classifier.update(
            {
                "schema_version": "classifier-manifest.v1",
                "model_version": "classifier-trained-not-loaded",
                "status": "CANDIDATE",
                "artifact_kind": "TRAINED_ARTIFACT",
                "artifact_uri": "file:///models/classifier/model",
                "artifact_checksum": "sha256:" + "a" * 64,
                "demo_artifact_id": None,
                "synthetic": False,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ModelRegistry(path=path, runtime_mode="demo")
        self.assertFalse(registry.ready)
        self.assertEqual(registry.load_error, "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED")

    def test_registry_loads_model_manifest_without_data_or_training_imports(self) -> None:
        registry = ModelRegistry(path=MANIFEST, runtime_mode="demo")
        self.assertTrue(registry.ready)
        self.assertEqual(
            registry.model_versions()["classifier"],
            "classifier-demo-2026-09-21-001",
        )
        self.assertFalse(any(name == "data" or name.startswith("data.") for name in sys.modules))
        self.assertFalse(
            any(name == "training" or name.startswith("training.") for name in sys.modules)
        )


if __name__ == "__main__":
    unittest.main()
