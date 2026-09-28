"""Fine-tune an offline classifier candidate from verified operator feedback."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.confidence import POLICY_VERSION
from app.classifier_input import encode_classifier_texts
from app.trained_classifier import TrainedClassifierService
from training.atomic_publish import publish_directory
from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE, load_verified_candidate


class CandidateTrainingError(RuntimeError):
    """Machine-readable failure stage without feedback text or model paths."""


def train_feedback_candidate(
    package: Path, frozen_package: Path, production_dir: Path, output: Path, *,
    candidate_model_version: str, epochs: int = 1, batch_size: int = 8,
    learning_rate: float = 2e-5, seed: int = 109,
) -> dict:
    if (not ID_RE.fullmatch(candidate_model_version) or epochs < 1 or batch_size < 1 or
            not 0 < learning_rate <= 0.001 or seed < 0):
        raise ValueError("invalid candidate version or training configuration")
    if output.exists() or output.is_symlink():
        raise FileExistsError("candidate model artifact already exists")
    if any(output.resolve().is_relative_to(path.resolve()) for path in (package, frozen_package, production_dir)):
        raise ValueError("candidate model artifact must be outside immutable inputs")
    try:
        manifest, samples = load_verified_candidate(package, frozen_package)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CandidateTrainingError("INVALID_CANDIDATE_DATASET") from error
    try:
        production = TrainedClassifierService(production_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CandidateTrainingError("INVALID_PRODUCTION_ARTIFACT") from error
    base = production.metadata
    if (base.model_version != manifest.production_model_version or
            base.model_version == candidate_model_version or
            any(sample.topic_id not in base.labels for sample in samples)):
        raise ValueError("production artifact or feedback labels do not match candidate dataset")
    max_length = int(base.training_config["max_length"])
    if max_length < 16 or max_length > 512:
        raise ValueError("production classifier input length is unsupported")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = production.tokenizer
    model = production.model
    device = production.device
    encoded = encode_classifier_texts(tokenizer, [sample.text for sample in samples],
                                      max_length=max_length, strategy=production.input_length_strategy,
                                      pad_to_max_length=True)
    labels = torch.tensor([base.labels.index(sample.topic_id) for sample in samples], dtype=torch.long)
    loader = DataLoader(TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        total_loss = 0.0
        for input_ids, attention_mask, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                         labels=targets.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
        final_loss = round(total_loss / len(loader), 6)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as stage_root:
        artifact = Path(stage_root) / "artifact"
        artifact.mkdir()
        model.save_pretrained(artifact, safe_serialization=True)
        tokenizer.save_pretrained(artifact)
        artifact_checksum = checksum(artifact / "model.safetensors")
        metadata = {
            "model_version": candidate_model_version,
            "model_family": base.model_family,
            "base_model": base.model_version,
            "base_model_artifact_checksum": base.artifact_checksum,
            "base_model_manifest_sha256": checksum(production_dir / "manifest.json"),
            "dataset_version": manifest.candidate_dataset_version,
            "dataset_content_sha256": manifest.content_sha256,
            "frozen_evaluation_version": manifest.frozen_package["evaluation_version"],
            "frozen_evaluation_sha256": manifest.frozen_package["evaluation_sha256"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "CANDIDATE",
            "metrics": {"status": "feedback_candidate_unverified", "train_loss": final_loss,
                        "train_samples": len(samples), "frozen_test_evaluated": False},
            "languages": base.languages,
            "labels": base.labels,
            "confidence_policy_version": POLICY_VERSION,
            "confidence_thresholds": {"low_confidence_below": 0.55, "confident_at_or_above": 1.0},
            "confident_enabled": False,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "max_length": max_length,
                "input_length_strategy": production.input_length_strategy,
                "temperature": 1.0,
                "feedback_candidate": True,
                "train_samples": len(samples),
                "class_counts": dict(sorted(Counter(sample.topic_id for sample in samples).items())),
                "online_retraining": False,
            },
            "artifact_checksum": artifact_checksum,
        }
        (artifact / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
        try:
            checked = TrainedClassifierService(artifact)
            checked.classify("Проверка модели после обучения.")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise CandidateTrainingError("CANDIDATE_SANITY_FAILED") from error
        if output.exists() or output.is_symlink():
            raise FileExistsError("candidate model artifact already exists")
        publish_directory(artifact, output)
    return {"status": "COMPLETED", "candidate_model_version": candidate_model_version,
            "dataset_version": manifest.candidate_dataset_version,
            "dataset_content_sha256": manifest.content_sha256,
            "base_model_version": base.model_version, "artifact_checksum": artifact_checksum,
            "artifact_uri": str(output.resolve()), "sample_count": len(samples),
            "frozen_test_evaluated": False}
