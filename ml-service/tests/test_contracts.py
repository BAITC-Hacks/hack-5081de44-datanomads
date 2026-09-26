from __future__ import annotations

import asyncio
import json
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


class SharedContractTests(unittest.TestCase):
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
        runtime = ml_main.app.openapi()["components"]["schemas"]
        documented = published["components"]["schemas"]
        for schema_name in ("ModelManifestResponse", "ModelMetadata"):
            self.assertEqual(documented[schema_name], runtime[schema_name])

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
        self.assertEqual(readiness.json()["detail"], "MODEL_MANIFEST_UNAVAILABLE_OR_INVALID")

    def test_production_mode_is_unready_until_a_real_runtime_adapter_exists(self) -> None:
        registry = ModelRegistry(path=MANIFEST, runtime_mode="production")
        self.assertFalse(registry.ready)
        self.assertEqual(registry.load_error, "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED")
        self.assertEqual(
            registry.get("classifier").artifact_kind,
            "DETERMINISTIC_BASELINE",
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
