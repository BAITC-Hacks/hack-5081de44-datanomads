from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from training.contracts import ClassifierManifest, DatasetManifest, EmbedderManifest, EvaluationReport


CHECKSUM = "sha256:" + "a" * 64


class ManifestTests(unittest.TestCase):
    def test_dataset_manifest_round_trip_and_strict_counts(self) -> None:
        values = {
            "dataset_version": "synthetic-v1",
            "schema_version": "unified-ticket.v1",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "synthetic": True,
            "seed": 109,
            "sources": ["ikomek109"],
            "source_checksums": {"ikomek109": CHECKSUM},
            "record_count": 2,
            "quarantine_count": 1,
            "languages": {"RU": 1, "KZ": 1},
            "topics": {"roads": 2},
            "regions": {"KZ-ASTANA": 2},
            "split_policy": "scenario-group",
            "split_group_key": "scenario_id",
            "frozen_evaluation_version": "eval-v1",
            "pii_policy_version": "pii-v1",
            "content_sha256": CHECKSUM,
        }
        manifest = DatasetManifest(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write(path)
            self.assertEqual(DatasetManifest.read(path), manifest)
        with self.assertRaises(ValidationError):
            DatasetManifest(**{**values, "languages": {"RU": 1}})
        with self.assertRaises(ValidationError):
            DatasetManifest(**{**values, "content_sha256": "sha256:bad"})
        with self.assertRaises(ValidationError):
            DatasetManifest(**{**values, "unexpected": "ignored"})

    def test_classifier_and_embedder_manifests_require_artifact_evidence(self) -> None:
        common = {
            "model_version": "model-v1",
            "model_family": "transformer",
            "base_model": "local-base-v1",
            "dataset_version": "dataset-v1",
            "synthetic": True,
            "seed": 109,
            "training_config": {"epochs": 1},
            "artifact_checksum": CHECKSUM,
            "runtime_requirements": {"python": "3.11.15"},
        }
        classifier = ClassifierManifest(
            **common, labels=["roads", "water_supply"], languages=["RU", "KZ"],
            input_length_strategy="truncate-96", calibration={"method": "temperature"},
            confidence_thresholds={"review": 0.7}, held_out_metrics={"macro_f1": 0.8},
            dataset_content_sha256=CHECKSUM, evaluation_version="eval-v1",
            evaluation_report_sha256=CHECKSUM, artifact_uri="model",
            created_at=datetime(2026, 9, 27, tzinfo=timezone.utc), artifact_files={"model.safetensors": CHECKSUM},
            bundle_files={"metrics.json": CHECKSUM},
        )
        self.assertEqual(classifier.model_type, "classifier")
        with self.assertRaises(ValidationError):
            ClassifierManifest.model_validate({**classifier.model_dump(), "labels": ["roads", "roads"]})
        with self.assertRaises(ValidationError):
            ClassifierManifest.model_validate({**classifier.model_dump(), "confidence_thresholds": {"review": 1.2}})
        embedder = EmbedderManifest(
            **common, embedding_dimension=768, pooling="mean", normalization="l2",
            retrieval_metrics={"recall_at_1": 0.5},
            base_model_artifact_sha256=CHECKSUM, dataset_content_sha256=CHECKSUM,
            frozen_evaluation_version="eval-v1", frozen_evaluation_sha256=CHECKSUM,
            evaluation_report_sha256=CHECKSUM, artifact_uri="model",
            created_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
            artifact_files={"model.safetensors": CHECKSUM}, bundle_files={"metrics.json": CHECKSUM},
        )
        self.assertEqual(embedder.model_type, "embedder")
        with self.assertRaises(ValidationError):
            EmbedderManifest.model_validate({**embedder.model_dump(), "embedding_dimension": 0})

    def test_evaluation_report_requires_explicit_origin_and_checksum(self) -> None:
        report = EvaluationReport(
            evaluated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            task_type="classifier", dataset_version="eval-v1", model_version="candidate-v1",
            synthetic=True, sample_count=10, metrics={"macro_f1": 0.7},
            evaluation_data_checksum=CHECKSUM,
        )
        self.assertTrue(report.synthetic)
        with self.assertRaises(ValidationError):
            EvaluationReport.model_validate({**report.model_dump(), "evaluated_at": datetime(2026, 1, 2)})
        with self.assertRaises(ValidationError):
            EvaluationReport.model_validate({key: value for key, value in report.model_dump().items() if key != "synthetic"})


if __name__ == "__main__":
    unittest.main()
