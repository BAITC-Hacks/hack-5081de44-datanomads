from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.candidate_runtime import CandidateArtifactError, classify_candidate
from app.schemas import CandidateTrainingJob
from app.training.classifier import train_candidate_classifier
from contracts import validate_document
from worker import process_job


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _training_job(directory: Path) -> CandidateTrainingJob:
    dataset_version = "candidate-dataset-cycle-14-aaaaaaaaaaaaaaaa"
    candidate_model_version = "classifier-candidate-cycle-14"
    production_model_version = "classifier-production-v1"
    dataset_directory = directory / "datasets"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_directory / "dataset.jsonl"
    dataset_bytes = b"".join(
        [
            _canonical_json(
                {
                    "text": "Не работает насосная станция",
                    "label": "water_supply",
                    "topic_id": "water_supply",
                    "language": "RU",
                }
            )
            + b"\n",
            _canonical_json(
                {
                    "text": "Не горит уличный фонарь",
                    "label": "street_lighting",
                    "topic_id": "street_lighting",
                    "language": "RU",
                }
            )
            + b"\n",
        ]
    )
    dataset_path.write_bytes(dataset_bytes)
    dataset_uri = dataset_path.resolve().as_uri()
    manifest = {
        "schema_version": "candidate-dataset-manifest.v1",
        "dataset_version": dataset_version,
        "artifact_uri": dataset_uri,
        "content_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "record_count": 2,
        "feedback_ids": ["feedback-1", "feedback-2"],
        "feedback_export_uri": "file:///artifacts/feedback-export.json",
        "feedback_export_sha256": "b" * 64,
        "candidate_model_version": candidate_model_version,
        "production_model_version": production_model_version,
        "frozen_evaluation_dataset_version": "evaluation-v1",
        "frozen_evaluation_ticket_ids": ["ticket-evaluation-1"],
        "source_dataset_versions": ["source-v1"],
        "training_ticket_ids": ["ticket-train-1", "ticket-train-2"],
        "synthetic": False,
        "created_at": "2026-09-27T00:00:00Z",
    }
    manifest_path = dataset_directory / "dataset.manifest.json"
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    output_path = directory / "candidates" / "candidate-014" / "classifier.json"
    payload = {
        "schema_version": "candidate-training-job.v1",
        "cycle_id": "cycle-14",
        "candidate_model_version": candidate_model_version,
        "candidate_dataset_version": dataset_version,
        "production_model_version": production_model_version,
        "training_config_version": "classifier-training.v1",
        "dataset_uri": dataset_uri,
        "dataset_checksum": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_manifest_uri": manifest_path.resolve().as_uri(),
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "output_artifact_uri": output_path.resolve().as_uri(),
        "min_samples": 2,
    }
    return CandidateTrainingJob.model_validate(payload)


class _RecordingPool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "OK"

    def acquire(self) -> "_PoolAcquire":
        return _PoolAcquire(self)


class _PoolAcquire:
    def __init__(self, pool: _RecordingPool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _RecordingPool:
        return self.pool

    async def __aexit__(self, *_: object) -> None:
        return None


class CandidateClassifierTrainerTests(unittest.TestCase):
    def test_trains_candidate_from_verified_artifact_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            job = _training_job(directory)

            with patch.dict(os.environ, {"MODEL_DIR": str(directory)}):
                result = train_candidate_classifier(job)

            artifact_path = Path(result.artifact_uri.removeprefix("file://"))
            manifest_path = Path(result.manifest_uri.removeprefix("file://"))
            artifact_bytes = artifact_path.read_bytes()
            artifact = json.loads(artifact_bytes)
            validate_document(artifact, "TrainedClassifierArtifact")
            self.assertEqual(result.status, "COMPLETED")
            self.assertEqual(result.candidate_model_version, job.candidate_model_version)
            self.assertEqual(result.candidate_dataset_version, job.candidate_dataset_version)
            self.assertEqual(result.sample_count, 2)
            self.assertEqual(result.metrics["class_count"], 2)
            self.assertEqual(result.manifest["status"], "CANDIDATE")
            self.assertFalse(result.synthetic)
            self.assertEqual(
                result.artifact_checksum,
                f"sha256:{hashlib.sha256(artifact_bytes).hexdigest()}",
            )
            self.assertEqual(
                result.manifest_checksum,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                set(artifact["labels"]), {"water_supply", "street_lighting"}
            )
            self.assertEqual(artifact["feature_encoding"], "sha256")
            self.assertTrue(
                all(len(feature) == 64 for feature in artifact["vocabulary"])
            )
            self.assertNotIn("Не работает насосная станция", artifact_bytes.decode())

    def test_shadow_runtime_serves_only_the_checksum_pinned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            job = _training_job(directory)

            with patch.dict(os.environ, {"MODEL_DIR": str(directory), "PULSE_ENV": "test"}):
                result = train_candidate_classifier(job)
                prediction = classify_candidate(
                    result.candidate_model_version,
                    result.artifact_checksum,
                    "Не работает насосная станция",
                    "RU",
                    2,
                )
                self.assertEqual(prediction.model_version, result.candidate_model_version)
                self.assertEqual(prediction.topic_id, "water_supply")
                self.assertEqual(len(prediction.alternatives), 2)
                with self.assertRaisesRegex(
                    CandidateArtifactError, "CANDIDATE_ARTIFACT_CHECKSUM_MISMATCH"
                ):
                    classify_candidate(
                        result.candidate_model_version,
                        f"sha256:{'0' * 64}",
                        "Не работает насосная станция",
                        "RU",
                        2,
                    )

    def test_production_runtime_rejects_synthetic_candidate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            original_job = _training_job(directory)
            manifest_path = Path(original_job.dataset_manifest_uri.removeprefix("file://"))
            manifest = json.loads(manifest_path.read_bytes())
            manifest["synthetic"] = True
            manifest_bytes = _canonical_json(manifest)
            manifest_path.write_bytes(manifest_bytes)
            job_payload = original_job.model_dump()
            job_payload["dataset_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            job = CandidateTrainingJob.model_validate(job_payload)

            with patch.dict(os.environ, {"MODEL_DIR": str(directory), "PULSE_ENV": "production"}):
                result = train_candidate_classifier(job)
                self.assertTrue(result.synthetic)
                with self.assertRaisesRegex(
                    CandidateArtifactError, "CANDIDATE_SYNTHETIC_ARTIFACT_FORBIDDEN"
                ):
                    classify_candidate(
                        result.candidate_model_version,
                        result.artifact_checksum,
                        "Не работает насосная станция",
                        "RU",
                        2,
                    )

    def test_rejects_a_dataset_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            valid_job = _training_job(Path(temporary_directory))
            payload = valid_job.model_dump()
            payload["dataset_checksum"] = "0" * 64
            invalid_job = CandidateTrainingJob.model_validate(payload)

            with patch.dict(os.environ, {"MODEL_DIR": str(Path(temporary_directory))}):
                with self.assertRaisesRegex(
                    RuntimeError, "CANDIDATE_DATASET_CHECKSUM_MISMATCH"
                ):
                    train_candidate_classifier(invalid_job)

    def test_production_model_version_cannot_be_reused_for_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            valid_job = _training_job(Path(temporary_directory))
            payload = valid_job.model_dump()
            payload["candidate_model_version"] = payload["production_model_version"]
            invalid_job = CandidateTrainingJob.model_validate(payload)

            with self.assertRaisesRegex(RuntimeError, "CANDIDATE_MODEL_VERSION_CONFLICT"):
                train_candidate_classifier(invalid_job)

    def test_normal_worker_starts_shadow_evaluation_without_promoting_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            job_payload = _training_job(directory).model_dump(mode="json")
            pool = _RecordingPool()

            with patch.dict(
                os.environ,
                {
                    "MODEL_DIR": str(directory),
                    "PULSE_ENV": "production",
                    "PULSE_TEST_FAKE_TRAINER": "false",
                },
            ):
                asyncio.run(
                    process_job(
                        pool,
                        {
                            "id": 18,
                            "job_type": "TRAIN_CLASSIFIER",
                            "payload": job_payload,
                        },
                    )
                )

            artifact_path = Path(
                job_payload["output_artifact_uri"].removeprefix("file://")
            )
            self.assertTrue(artifact_path.is_file())
            model_registration = next(
                (query, args)
                for query, args in pool.executed
                if "INSERT INTO model_versions" in query
            )
            self.assertIn("'SHADOW'", model_registration[0])
            self.assertNotIn("'PRODUCTION'", model_registration[0])
            self.assertEqual(
                model_registration[1][3],
                artifact_path.with_name("manifest.json").resolve().as_uri(),
            )
            self.assertTrue(str(model_registration[1][4]).startswith("sha256:"))
            self.assertFalse(
                any("UPDATE model_versions SET status = 'PRODUCTION'" in query for query, _ in pool.executed)
            )

            training_result = next(
                json.loads(args[1])
                for query, args in pool.executed
                if "payload = payload || jsonb_build_object('result'" in query
            )
            self.assertEqual(training_result["candidate_model_version"], job_payload["candidate_model_version"])
            self.assertEqual(training_result["status"], "COMPLETED")
            self.assertNotIn("samples", training_result)
            cycle_update = next(
                query
                for query, _ in pool.executed
                if "UPDATE learning_cycles SET state = 'EVALUATE'" in query
            )
            self.assertIn("evaluation_started_at = now()", cycle_update)
            self.assertIn("evaluation_ends_at = now() + (collect_ends_at - collect_started_at)", cycle_update)
            self.assertIn("blind_ab_enabled = false", cycle_update)


if __name__ == "__main__":
    unittest.main()
