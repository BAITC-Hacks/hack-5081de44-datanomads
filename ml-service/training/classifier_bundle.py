"""Build and verify immutable classifier handoff bundles."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

import torch
import transformers

from app.confidence import ConfidencePolicy
from app.schemas import ModelMetadata
from training.classifier_baselines import load_verified_classifier_package
from training.classifier_candidate_eval import evaluate_candidate
from training.contracts import ClassifierManifest
from training.dataset_builder import checksum


MODEL_FILES = (
    "model.safetensors", "config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "manifest.json",
)
BUNDLE_FILES = (
    "metrics.json", "training_config.json", "label_map.json",
    "thresholds.json", "artifact_checksum.txt", "MODEL_CARD.md",
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_classifier_bundle(bundle: Path) -> ClassifierManifest:
    if bundle.is_symlink() or (bundle / "manifest.json").is_symlink() or (bundle / "model").is_symlink():
        raise ValueError("classifier bundle must be self-contained")
    manifest = ClassifierManifest.read(bundle / "manifest.json")
    if (manifest.artifact_uri != "model" or manifest.status != "CANDIDATE" or
            set(manifest.artifact_files) != {f"model/{name}" for name in MODEL_FILES} or
            set(manifest.bundle_files) != set(BUNDLE_FILES)):
        raise ValueError("classifier bundle layout or status is invalid")
    if ({path.name for path in bundle.iterdir()} != {"manifest.json", "model", *BUNDLE_FILES} or
            {path.name for path in (bundle / "model").iterdir()} != set(MODEL_FILES)):
        raise ValueError("classifier bundle contains unexpected or missing files")
    for files in (manifest.artifact_files, manifest.bundle_files):
        for name, expected in files.items():
            if (bundle / name).is_symlink():
                raise ValueError(f"classifier bundle contains symlink: {name}")
            if checksum(bundle / name) != expected:
                raise ValueError(f"classifier bundle checksum mismatch: {name}")
    runtime = ModelMetadata.model_validate_json((bundle / "model/manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((bundle / "training_config.json").read_text(encoding="utf-8"))
    labels = json.loads((bundle / "label_map.json").read_text(encoding="utf-8"))
    thresholds = json.loads((bundle / "thresholds.json").read_text(encoding="utf-8"))
    if any(not isinstance(value, dict) for value in (metrics, config, labels, thresholds)):
        raise ValueError("classifier bundle metadata must contain JSON objects")
    validation = runtime.metrics.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("classifier artifact lacks validation evidence")
    if (runtime.model_version != manifest.model_version or
            runtime.model_family != manifest.model_family or
            runtime.base_model != manifest.base_model or
            runtime.dataset_version != manifest.dataset_version or
            (runtime.model_extra or {}).get("dataset_content_sha256") != manifest.dataset_content_sha256 or
            (runtime.model_extra or {}).get("frozen_evaluation_version") != manifest.evaluation_version or
            runtime.artifact_checksum != manifest.artifact_checksum or
            datetime.fromisoformat(runtime.created_at) != manifest.created_at or
            runtime.labels != manifest.labels or runtime.languages != manifest.languages or
            runtime.training_config != config or manifest.training_config != config or
            manifest.seed != config.get("seed") or
            manifest.input_length_strategy != config.get("input_length_strategy") or
            manifest.calibration != validation.get("calibration") or
            manifest.artifact_files["model/model.safetensors"] != manifest.artifact_checksum or
            (bundle / "artifact_checksum.txt").read_text(encoding="utf-8") != manifest.artifact_checksum + "\n" or
            metrics.get("report_version") != "classifier-candidate-evaluation.v1" or
            metrics.get("model_version") != manifest.model_version or
            metrics.get("dataset_version") != manifest.dataset_version or
            metrics.get("dataset_content_sha256") != manifest.dataset_content_sha256 or
            metrics.get("frozen_evaluation_version") != manifest.evaluation_version or
            metrics.get("synthetic") != manifest.synthetic or
            metrics.get("model_artifact_checksum") != manifest.artifact_checksum or
            metrics.get("labels") != sorted(manifest.labels) or
            checksum(bundle / "metrics.json") != manifest.evaluation_report_sha256 or
            metrics.get("candidate_metrics") != manifest.held_out_metrics or
            labels.get("labels") != manifest.labels or
            labels.get("id2label") != {str(index): label for index, label in enumerate(manifest.labels)} or
            thresholds.get("confidence_thresholds") != manifest.confidence_thresholds or
            manifest.confidence_thresholds != (runtime.model_extra or {}).get("confidence_thresholds") or
            thresholds.get("confidence_policy_version") != (runtime.model_extra or {}).get("confidence_policy_version") or
            thresholds.get("confident_enabled") != (runtime.model_extra or {}).get("confident_enabled")):
        raise ValueError("classifier bundle manifest does not match its evidence")
    ConfidencePolicy.from_metadata(runtime)
    return manifest


def build_classifier_bundle(package: Path, model_dir: Path, baseline_path: Path,
                            output: Path) -> ClassifierManifest:
    if output.exists() or output.is_symlink():
        raise FileExistsError("classifier bundle already exists")
    if (output.resolve().is_relative_to(package.resolve()) or
            output.resolve().is_relative_to(model_dir.resolve())):
        raise ValueError("classifier bundle must be outside immutable inputs")
    dataset, _ = load_verified_classifier_package(package)
    metrics = evaluate_candidate(package, model_dir, baseline_path)
    runtime = ModelMetadata.model_validate_json((model_dir / "manifest.json").read_text(encoding="utf-8"))
    ConfidencePolicy.from_metadata(runtime)
    extra = runtime.model_extra or {}
    if not {"confidence_policy_version", "confidence_thresholds", "confident_enabled"}.issubset(extra):
        raise ValueError("classifier artifact lacks versioned confidence policy")
    if metrics["dataset_content_sha256"] != dataset.content_sha256:
        raise ValueError("classifier evaluation does not match dataset")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as stage_root:
        bundle = Path(stage_root) / "bundle"
        artifact = bundle / "model"
        artifact.mkdir(parents=True)
        for name in MODEL_FILES:
            shutil.copyfile(model_dir / name, artifact / name)
        _write_json(bundle / "metrics.json", metrics)
        _write_json(bundle / "training_config.json", runtime.training_config)
        _write_json(bundle / "label_map.json", {
            "labels": runtime.labels,
            "id2label": {str(index): label for index, label in enumerate(runtime.labels)},
        })
        _write_json(bundle / "thresholds.json", {
            "confidence_policy_version": extra["confidence_policy_version"],
            "confidence_thresholds": extra["confidence_thresholds"],
            "confident_enabled": extra["confident_enabled"],
        })
        (bundle / "artifact_checksum.txt").write_text(runtime.artifact_checksum + "\n", encoding="utf-8")
        (bundle / "MODEL_CARD.md").write_text(
            f"# {runtime.model_version}\n\n"
            "Status: CANDIDATE. Human promotion is required.\n\n"
            f"Dataset: {dataset.dataset_version} (`{dataset.content_sha256}`).\n"
            "Composition: 100% human-reviewed synthetic scenarios; 0% real citizen texts.\n"
            f"Frozen evaluation: {dataset.frozen_evaluation_version}.\n"
            f"Candidate macro-F1: {metrics['candidate_metrics']['macro_f1']}; "
            f"TF-IDF macro-F1: {metrics['tfidf_baseline_metrics']['macro_f1']}.\n"
            f"Evaluation decision: {metrics['decision']}.\n\n"
            "Limits: no verified quality on customer appeal text; no approved real RU/KZ "
            "holdout, unknown/ambiguous challenge result, shadow result or critical-regression "
            "decision. Every prediction still requires operator review.\n",
            encoding="utf-8",
        )
        manifest = ClassifierManifest(
            model_version=runtime.model_version,
            model_family=runtime.model_family,
            base_model=runtime.base_model,
            dataset_version=dataset.dataset_version,
            dataset_content_sha256=dataset.content_sha256,
            evaluation_version=dataset.frozen_evaluation_version,
            evaluation_report_sha256=checksum(bundle / "metrics.json"),
            artifact_uri="model",
            created_at=datetime.fromisoformat(runtime.created_at),
            synthetic=dataset.synthetic,
            seed=int(runtime.training_config["seed"]),
            labels=runtime.labels,
            languages=runtime.languages,
            training_config=runtime.training_config,
            input_length_strategy=str(runtime.training_config["input_length_strategy"]),
            calibration=runtime.metrics["validation"]["calibration"],
            confidence_thresholds=extra["confidence_thresholds"],
            held_out_metrics=metrics["candidate_metrics"],
            artifact_checksum=runtime.artifact_checksum,
            artifact_files={f"model/{name}": checksum(artifact / name) for name in MODEL_FILES},
            bundle_files={name: checksum(bundle / name) for name in BUNDLE_FILES},
            runtime_requirements={
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": transformers.__version__,
            },
        )
        manifest.write(bundle / "manifest.json")
        verify_classifier_bundle(bundle)
        bundle.rename(output)
    return manifest
