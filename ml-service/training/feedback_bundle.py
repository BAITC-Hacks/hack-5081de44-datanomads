"""Finalize an evaluated feedback classifier inside an unpublished cycle stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil

import torch
import transformers

from app.confidence import ConfidencePolicy
from app.schemas import ModelMetadata
from training.classifier_baselines import load_verified_classifier_package
from training.classifier_pair_eval import CriticalRegressionPolicy
from training.dataset_builder import checksum
from training.feedback_dataset import load_verified_candidate


MODEL_FILES = {"model.safetensors", "config.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json"}
OPTIONAL_MODEL_FILES = {"added_tokens.json"}
BUNDLE_FILES = {"metrics.json", "training_config.json", "label_map.json",
                "thresholds.json", "artifact_checksum.txt", "MODEL_CARD.md"}
BUNDLE_VERSION = "feedback-classifier-bundle.v1"


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _lineage(model: ModelMetadata, package: Path, frozen: Path, metrics: dict, policy: Path) -> tuple:
    if not isinstance(metrics, dict):
        raise ValueError("feedback classifier evaluation report is invalid")
    candidate, samples = load_verified_candidate(package, frozen)
    evaluation, splits = load_verified_classifier_package(frozen)
    policy_data = CriticalRegressionPolicy.model_validate_json(policy.read_text(encoding="utf-8"))
    extra = model.model_extra or {}
    candidate_metrics = metrics.get("candidate")
    production_metrics = metrics.get("production")
    required_metrics = {"macro_f1", "weighted_f1", "per_class", "confusion_matrix", "accuracy"}
    sample_ids = sorted(row["variant_id"] for row in splits["test"])
    sample_ids_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(sample_ids, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if (extra.get("status") != "CANDIDATE" or
            extra.get("confident_enabled") is not False or
            model.dataset_version != candidate.candidate_dataset_version or
            model.training_config.get("train_samples") != len(samples) or
            set(model.labels) != set(evaluation.topics) or
            not {"RU", "KZ"} <= set(model.languages) or
            extra.get("dataset_content_sha256") != candidate.content_sha256 or
            model.base_model != candidate.production_model_version or
            extra.get("frozen_evaluation_version") != evaluation.frozen_evaluation_version or
            extra.get("frozen_evaluation_sha256") != evaluation.frozen_evaluation_sha256 or
            metrics.get("report_version") != "classifier-pair-evaluation.v1" or
            metrics.get("dataset_version") != evaluation.dataset_version or
            metrics.get("dataset_content_sha256") != evaluation.content_sha256 or
            metrics.get("frozen_evaluation_version") != evaluation.frozen_evaluation_version or
            metrics.get("frozen_evaluation_sha256") != evaluation.frozen_evaluation_sha256 or
            metrics.get("synthetic") != evaluation.synthetic or
            metrics.get("sample_count") != len(splits["test"]) or
            metrics.get("sample_ids_sha256") != sample_ids_sha256 or
            metrics.get("labels") != sorted({row["topic_id"] for row in splits["test"]}) or
            metrics.get("policy_sha256") != checksum(policy) or
            metrics.get("policy_version") != policy_data.policy_version or
            metrics.get("policy") != policy_data.model_dump() or
            not isinstance(candidate_metrics, dict) or
            candidate_metrics.get("model_version") != model.model_version or
            candidate_metrics.get("dataset_version") != candidate.candidate_dataset_version or
            candidate_metrics.get("artifact_checksum") != model.artifact_checksum or
            not isinstance(candidate_metrics.get("metrics"), dict) or
            not required_metrics <= set(candidate_metrics["metrics"]) or
            not isinstance(production_metrics, dict) or
            production_metrics.get("model_version") != model.base_model or
            production_metrics.get("artifact_checksum") != extra.get("base_model_artifact_checksum") or
            not isinstance(production_metrics.get("metrics"), dict) or
            not required_metrics <= set(production_metrics["metrics"]) or
            not isinstance(metrics.get("critical_topics"), dict) or
            not isinstance(metrics.get("regressed_critical_topics"), list) or
            not isinstance(metrics.get("insufficient_critical_topics"), list) or
            metrics.get("decision") not in {"PENDING_HUMAN_REVIEW", "CRITICAL_REGRESSION",
                                            "INSUFFICIENT_EVIDENCE"}):
        raise ValueError("feedback classifier bundle lineage does not match evaluation")
    return candidate, evaluation


def verify_feedback_bundle(model_dir: Path, package: Path, frozen: Path,
                           offline_report: Path, policy: Path) -> ModelMetadata:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError("feedback classifier bundle directory is invalid")
    entries = list(model_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("feedback classifier bundle contains linked or non-file entries")
    model = ModelMetadata.model_validate_json((model_dir / "manifest.json").read_text(encoding="utf-8"))
    extra = model.model_extra or {}
    bundle = extra.get("handoff_bundle")
    if (not isinstance(bundle, dict) or set(bundle) != {
            "version", "artifact_uri", "artifact_files", "bundle_files", "evaluation_version",
            "evaluation_report_sha256", "candidate_manifest_sha256", "origin_counts",
            "runtime_requirements",
        } or bundle["version"] != BUNDLE_VERSION or bundle["artifact_uri"] != "." or
            extra.get("model_type") != "classifier" or extra.get("artifact_uri") != "."):
        raise ValueError("feedback classifier handoff metadata is invalid")
    artifact_files = bundle["artifact_files"]
    bundle_files = bundle["bundle_files"]
    if (not isinstance(artifact_files, dict) or not isinstance(bundle_files, dict) or
            not MODEL_FILES <= set(artifact_files) or
            set(artifact_files) - MODEL_FILES - OPTIONAL_MODEL_FILES or
            set(bundle_files) != BUNDLE_FILES or
            {path.name for path in entries} != {"manifest.json", *artifact_files, *bundle_files}):
        raise ValueError("feedback classifier bundle layout is invalid")
    for name, expected in {**artifact_files, **bundle_files}.items():
        if checksum(model_dir / name) != expected:
            raise ValueError("feedback classifier bundle checksum mismatch")
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
    labels = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    candidate, evaluation = _lineage(model, package, frozen, metrics, policy)
    if (model.artifact_checksum != artifact_files["model.safetensors"] or
            (model_dir / "artifact_checksum.txt").read_text(encoding="utf-8") != model.artifact_checksum + "\n" or
            bundle["evaluation_version"] != evaluation.frozen_evaluation_version or
            bundle["evaluation_report_sha256"] != checksum(metrics_path) or
            checksum(offline_report) != checksum(metrics_path) or
            bundle["candidate_manifest_sha256"] != checksum(package / "manifest.json") or
            bundle["origin_counts"] != candidate.origin_counts or
            not isinstance(bundle["runtime_requirements"], dict) or
            set(bundle["runtime_requirements"]) != {"python", "torch", "transformers"} or
            config != model.training_config or
            labels != {"labels": model.labels,
                       "id2label": {str(index): label for index, label in enumerate(model.labels)}} or
            thresholds != {"confidence_policy_version": (model.model_extra or {}).get("confidence_policy_version"),
                           "confidence_thresholds": (model.model_extra or {}).get("confidence_thresholds"),
                           "confident_enabled": (model.model_extra or {}).get("confident_enabled")} or
            model.metrics.get("status") != "offline_evaluated" or
            model.metrics.get("offline_report_sha256") != checksum(metrics_path) or
            model.metrics.get("frozen_test_evaluated") is not True):
        raise ValueError("feedback classifier bundle manifest does not match its evidence")
    ConfidencePolicy.from_metadata(model)
    return model


def finalize_staged_feedback_bundle(model_dir: Path, package: Path, frozen: Path,
                                    offline_report: Path, policy: Path) -> ModelMetadata:
    """Call only before publishing the enclosing private learning-cycle stage."""
    if model_dir.is_symlink() or not model_dir.is_dir() or any(
        path.is_symlink() or not path.is_file() for path in model_dir.iterdir()
    ):
        raise ValueError("staged feedback classifier artifact is invalid")
    if any((model_dir / name).exists() for name in BUNDLE_FILES):
        raise FileExistsError("feedback classifier bundle files already exist")
    files = {path.name for path in model_dir.iterdir()}
    if not MODEL_FILES <= files or files - MODEL_FILES - OPTIONAL_MODEL_FILES != {"manifest.json"}:
        raise ValueError("staged feedback classifier artifact layout is invalid")
    manifest_path = model_dir / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = ModelMetadata.model_validate(metadata)
    if (model.artifact_checksum != checksum(model_dir / "model.safetensors") or
            (model.model_extra or {}).get("handoff_bundle") is not None or
            model.metrics.get("status") != "feedback_candidate_unverified" or
            model.metrics.get("frozen_test_evaluated") is not False):
        raise ValueError("staged feedback classifier is not an unevaluated candidate")
    metrics = json.loads(offline_report.read_text(encoding="utf-8"))
    candidate, evaluation = _lineage(model, package, frozen, metrics, policy)
    shutil.copyfile(offline_report, model_dir / "metrics.json")
    _write_json(model_dir / "training_config.json", model.training_config)
    _write_json(model_dir / "label_map.json", {
        "labels": model.labels,
        "id2label": {str(index): label for index, label in enumerate(model.labels)},
    })
    extra = model.model_extra or {}
    _write_json(model_dir / "thresholds.json", {
        "confidence_policy_version": extra.get("confidence_policy_version"),
        "confidence_thresholds": extra.get("confidence_thresholds"),
        "confident_enabled": extra.get("confident_enabled"),
    })
    (model_dir / "artifact_checksum.txt").write_text(model.artifact_checksum + "\n", encoding="utf-8")
    (model_dir / "MODEL_CARD.md").write_text(
        f"# {model.model_version}\n\n"
        "Status: CANDIDATE; human promotion required.\n\n"
        f"Feedback dataset: {candidate.candidate_dataset_version} ({candidate.content_sha256}).\n"
        f"Composition: {candidate.origin_counts.get('synthetic', 0)} synthetic and "
        f"{candidate.origin_counts.get('real', 0)} reviewed real feedback rows.\n"
        f"Frozen evaluation: {evaluation.frozen_evaluation_version}.\n"
        f"Offline decision: {metrics['decision']}. Confidence remains disabled.\n\n"
        "Limits: results apply only to the stated frozen evaluation set. "
        "Real RU/KZ quality, shadow agreement and production suitability require "
        "separate evidence; operator and human model review remain required.\n",
        encoding="utf-8",
    )
    metadata["model_type"] = "classifier"
    metadata["artifact_uri"] = "."
    metadata["metrics"] = {**model.metrics, "status": "offline_evaluated",
                           "frozen_test_evaluated": True,
                           "offline_report_sha256": checksum(model_dir / "metrics.json")}
    metadata["handoff_bundle"] = {
        "version": BUNDLE_VERSION, "artifact_uri": ".",
        "artifact_files": {name: checksum(model_dir / name)
                           for name in sorted(files - {"manifest.json"})},
        "bundle_files": {name: checksum(model_dir / name) for name in sorted(BUNDLE_FILES)},
        "evaluation_version": evaluation.frozen_evaluation_version,
        "evaluation_report_sha256": checksum(model_dir / "metrics.json"),
        "candidate_manifest_sha256": checksum(package / "manifest.json"),
        "origin_counts": candidate.origin_counts,
        "runtime_requirements": {"python": platform.python_version(), "torch": str(torch.__version__),
                                 "transformers": transformers.__version__},
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    return verify_feedback_bundle(model_dir, package, frozen, offline_report, policy)
