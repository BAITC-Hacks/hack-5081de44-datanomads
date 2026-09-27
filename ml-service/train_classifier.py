"""Offline fine-tuning from a reviewed package or an explicit synthetic demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.constants import TOPICS
from app.confidence import ConfidencePolicy, POLICY_VERSION
from app.classifier_input import encode_classifier_texts
from training.atomic_publish import publish_directory
from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum as file_checksum, normalized_text

LABELS = tuple(topic.topic_id for topic in TOPICS if topic.topic_id != "other")


def read_split(path: Path, split: str) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty split: {path}")
    for row in rows:
        if row.get("split") != split or row.get("topic_id") not in LABELS:
            raise ValueError(f"invalid split or label in {path}")
        if row.get("language") not in {"RU", "KZ", "MIXED"} or row.get("synthetic") is not True:
            raise ValueError(f"expected synthetic RU/KZ/MIXED rows in {path}")
        if not row.get("text") or not row.get("scenario_id"):
            raise ValueError(f"missing text or scenario_id in {path}")
    return rows


def validate_splits(splits: dict[str, list[dict[str, str]]]) -> None:
    seen_scenarios: set[str] = set()
    seen_texts: set[str] = set()
    expected_languages: set[str] | None = None
    for name, rows in splits.items():
        scenarios = {row["scenario_id"] for row in rows}
        texts = {normalized_text(row["text"]) for row in rows}
        if seen_scenarios & scenarios or seen_texts & texts:
            raise ValueError(f"scenario or text leakage into {name}")
        if {row["topic_id"] for row in rows} != set(LABELS):
            raise ValueError(f"missing class in {name}")
        languages = {row["language"] for row in rows}
        if not {"RU", "KZ"}.issubset(languages) or not languages <= {"RU", "KZ", "MIXED"}:
            raise ValueError(f"missing or unsupported language in {name}")
        if expected_languages is not None and languages != expected_languages:
            raise ValueError(f"inconsistent languages in {name}")
        expected_languages = languages
        seen_scenarios.update(scenarios)
        seen_texts.update(texts)


def load_reviewed_splits(package: Path, audit_path: Path, base_model: Path, max_length: int,
                         input_length_strategy: str = "head"):
    if not base_model.is_dir():
        raise ValueError("reviewed training requires a local base-model directory")
    weights = base_model / "model.safetensors"
    if not weights.is_file():
        raise ValueError("reviewed training requires local safetensors base weights")
    if max_length not in (384, 512):
        raise ValueError("reviewed training requires max-length 384 or 512")
    if input_length_strategy not in ("head", "head-tail"):
        raise ValueError("reviewed training requires head or head-tail input strategy")
    manifest, splits = load_verified_classifier_package(package)
    if any({row["topic_id"] for row in rows} != set(LABELS) for rows in splits.values()):
        raise ValueError("reviewed classifier requires all canonical runtime labels in every split")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    expected_checksums = {name: file_checksum(base_model / name) for name in expected_files}
    if (audit.get("report_version") != "classifier-token-length-audit.v1" or
            audit.get("dataset_version") != manifest.dataset_version or
            audit.get("dataset_content_sha256") != manifest.content_sha256 or
            audit.get("frozen_evaluation_version") != manifest.frozen_evaluation_version or
            audit.get("tokenizer_file_checksums") != expected_checksums or
            audit.get("test_token_lengths_computed") is not False or
            f"{input_length_strategy}-{max_length}" not in audit.get("strategies_to_evaluate", [])):
        raise ValueError("token length audit does not match reviewed dataset and local tokenizer")
    return manifest, splits, file_checksum(weights)


def make_loader(rows: list[dict[str, str]], tokenizer, batch_size: int, max_length: int,
                shuffle: bool, input_length_strategy: str = "head") -> DataLoader:
    encoded = encode_classifier_texts(tokenizer, [row["text"] for row in rows],
                                      max_length=max_length,
                                      strategy=f"{input_length_strategy}-{max_length}",
                                      pad_to_max_length=True)
    labels = torch.tensor([LABELS.index(row["topic_id"]) for row in rows], dtype=torch.long)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def score(actual: list[str], predicted: list[str]) -> dict:
    if not actual or len(actual) != len(predicted):
        raise ValueError("classifier score requires equal non-empty actual and predicted lists")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, zero_division=0,
    )
    return {
        "accuracy": round(sum(a == p for a, p in zip(actual, predicted)) / len(actual), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "weighted_f1": round(float((f1 * support).sum() / len(actual)), 6),
        "per_class_f1": {label: round(float(f1[index]), 6) for index, label in enumerate(LABELS)},
        "per_class": {
            label: {
                "support": int(support[index]),
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader: DataLoader, rows: list[dict[str, str]], device: torch.device) -> dict:
    model.eval()
    predicted: list[str] = []
    for input_ids, attention_mask, _ in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        predicted.extend(LABELS[index] for index in logits.argmax(dim=-1).tolist())
    actual = [row["topic_id"] for row in rows]
    metrics = score(actual, predicted)
    metrics["by_language"] = {
        language: score(
            [a for a, row in zip(actual, rows) if row["language"] == language],
            [p for p, row in zip(predicted, rows) if row["language"] == language],
        )
        for language in ("RU", "KZ", "MIXED") if any(row["language"] == language for row in rows)
    }
    metrics["sample_count"] = len(rows)
    metrics["scenario_count"] = len({row["scenario_id"] for row in rows})
    return metrics


@torch.inference_mode()
def collect_logits(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    labels = []
    for input_ids, attention_mask, batch_labels in loader:
        logits.append(model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits.float())
        labels.append(batch_labels.to(device))
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimize validation NLL without using the held-out test split."""
    best_temperature = 1.0
    best_loss = float("inf")
    for index in range(10, 201):
        temperature = index / 10
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels).item()
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    correct = (predicted == labels).float()
    bins = torch.clamp((confidence * 10).long(), max=9)
    expected_calibration_error = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            expected_calibration_error += (
                (confidence[selected].mean() - correct[selected].mean()).abs()
                * selected.float().mean()
            ).item()
    return {
        "nll": round(torch.nn.functional.cross_entropy(logits / temperature, labels).item(), 6),
        "mean_confidence": round(confidence.mean().item(), 6),
        "accuracy": round(correct.mean().item(), 6),
        "ece_10_bins": round(expected_calibration_error, 6),
    }


def calibration_report(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict], temperature: float) -> dict:
    result = calibration_metrics(logits, labels, temperature)
    result["by_language"] = {}
    for language in ("RU", "KZ", "MIXED"):
        indices = [index for index, row in enumerate(rows) if row["language"] == language]
        if indices:
            result["by_language"][language] = calibration_metrics(logits[indices], labels[indices], temperature)
    return result


@torch.inference_mode()
def select_confidence_policy(
    logits: torch.Tensor, labels: torch.Tensor, rows: list[dict], temperature: float,
) -> tuple[ConfidencePolicy, dict]:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    correct = (predicted == labels).tolist()
    scores = confidence.tolist()
    selected_evidence = None
    for percent in range(55, 101):
        threshold = percent / 100
        selected = [index for index, value in enumerate(scores) if value >= threshold]
        if len(selected) < 30 or sum(correct[index] for index in selected) / len(selected) < 0.9:
            continue
        slices = {}
        for language in ("RU", "KZ"):
            indices = [index for index in selected if rows[index]["language"] == language]
            if len(indices) < 10:
                break
            precision = sum(correct[index] for index in indices) / len(indices)
            if precision < 0.9:
                break
            slices[language] = {"support": len(indices), "precision": round(precision, 6)}
        if len(slices) == 2:
            selected_evidence = {
                "threshold": threshold,
                "support": len(selected),
                "precision": round(sum(correct[index] for index in selected) / len(selected), 6),
                "by_language": slices,
            }
            break
    policy = ConfidencePolicy(
        low_confidence_below=0.55,
        confident_at_or_above=selected_evidence["threshold"] if selected_evidence else 1.0,
        confident_enabled=False,
    )
    return policy, {
        "policy_version": POLICY_VERSION,
        "candidate_selection": "validation precision >= 0.90 with >= 30 total and >= 10 RU/KZ each",
        "candidate": selected_evidence,
        "confident_disabled_reason": "SYNTHETIC_HOLDOUT_ONLY",
    }


@torch.inference_mode()
def confidence_state_report(logits: torch.Tensor, rows: list[dict], temperature: float,
                            policy: ConfidencePolicy) -> dict:
    confidence = torch.softmax(logits / temperature, dim=-1).max(dim=-1).values.tolist()
    states = [policy.state(value) for value in confidence]

    def summarize(indices: list[int]) -> dict:
        counts = {state: sum(states[index] == state for index in indices)
                  for state in ("CONFIDENT", "UNCERTAIN", "LOW_CONFIDENCE")}
        return {
            "sample_count": len(indices),
            "state_counts": counts,
            "confident_coverage": round(counts["CONFIDENT"] / len(indices), 6),
            "needs_review_share": 1.0,
            "low_confidence_share": round(counts["LOW_CONFIDENCE"] / len(indices), 6),
        }

    report = summarize(list(range(len(rows))))
    report["by_language"] = {
        language: summarize([index for index, row in enumerate(rows) if row["language"] == language])
        for language in ("RU", "KZ", "MIXED") if any(row["language"] == language for row in rows)
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reviewed-dataset", type=Path, help="verified, human-reviewed dataset package")
    source.add_argument("--demo-data-dir", type=Path, help="unreviewed generated synthetic splits")
    parser.add_argument("--token-audit", type=Path, help="matching aggregate token-length audit for reviewed training")
    parser.add_argument("--output-dir", type=Path, required=True, help="new immutable model artifact directory")
    parser.add_argument("--base-model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--input-length-strategy", choices=("head", "head-tail"), default="head")
    parser.add_argument("--validation-only", action="store_true",
                        help="select a training configuration without evaluating the frozen test")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 16:
        parser.error("epochs and batch-size must be positive; max-length must be at least 16")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("output-dir already exists; model artifacts are immutable")
    if args.reviewed_dataset:
        if args.token_audit is None:
            parser.error("reviewed training requires --token-audit")
        if args.output_dir.resolve().is_relative_to(args.reviewed_dataset.resolve()):
            parser.error("output-dir must be outside the immutable dataset package")
        if args.output_dir.resolve().is_relative_to(Path(args.base_model).resolve()):
            parser.error("output-dir must be outside the local base model")
        try:
            reviewed_manifest, splits, base_model_checksum = load_reviewed_splits(
                args.reviewed_dataset, args.token_audit, Path(args.base_model), args.max_length,
                args.input_length_strategy,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"reviewed dataset preflight failed: {error}")
        dataset_version = reviewed_manifest.dataset_version
        dataset_checksum = reviewed_manifest.content_sha256
        token_audit_checksum = file_checksum(args.token_audit)
        seed = reviewed_manifest.seed
    else:
        if args.token_audit is not None:
            parser.error("--token-audit applies only to --reviewed-dataset")
        splits = {name: read_split(args.demo_data_dir / f"{name}.jsonl", name) for name in ("train", "validation", "test")}
        validate_splits(splits)
        dataset_hash = hashlib.sha256()
        for name in splits:
            dataset_hash.update((args.demo_data_dir / f"{name}.jsonl").read_bytes())
        dataset_version = f"classifier-synthetic-v1-sha256:{dataset_hash.hexdigest()[:12]}"
        dataset_checksum = f"sha256:{dataset_hash.hexdigest()}"
        seed = 109

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"stage": "start", "device": str(device), "base_model": args.base_model}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=bool(args.reviewed_dataset))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        local_files_only=bool(args.reviewed_dataset),
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
    ).to(device)
    train_loader = make_loader(splits["train"], tokenizer, args.batch_size, args.max_length,
                               True, args.input_length_strategy)
    validation_loader = make_loader(splits["validation"], tokenizer, args.batch_size * 2,
                                    args.max_length, False, args.input_length_strategy)
    test_loader = (None if args.validation_only else
                   make_loader(splits["test"], tokenizer, args.batch_size * 2, args.max_length,
                               False, args.input_length_strategy))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent) as stage_root:
        artifact_dir = Path(stage_root) / "artifact"
        artifact_dir.mkdir()
        best_f1 = -1.0
        best_validation: dict = {}
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for step, (input_ids, attention_mask, labels) in enumerate(train_loader, 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = model(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        labels=labels.to(device),
                    ).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                if step % 250 == 0:
                    print(json.dumps({"stage": "train", "epoch": epoch + 1, "step": step, "steps": len(train_loader), "loss": round(sum(losses[-250:]) / min(len(losses), 250), 4)}), flush=True)
            validation = evaluate(model, validation_loader, splits["validation"], device)
            print(json.dumps({"stage": "validation", "epoch": epoch + 1, "macro_f1": validation["macro_f1"], "accuracy": validation["accuracy"]}), flush=True)
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                best_validation = validation
                model.save_pretrained(artifact_dir, safe_serialization=True)
                tokenizer.save_pretrained(artifact_dir)

        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device)
        validation_logits, validation_labels = collect_logits(model, validation_loader, device)
        temperature = fit_temperature(validation_logits, validation_labels)
        best_validation["calibration"] = calibration_report(
            validation_logits, validation_labels, splits["validation"], temperature,
        )
        confidence_policy, policy_evidence = select_confidence_policy(
            validation_logits, validation_labels, splits["validation"], temperature,
        )
        best_validation["confidence_states"] = confidence_state_report(
            validation_logits, splits["validation"], temperature, confidence_policy,
        )
        test = None
        if test_loader is not None:
            test = evaluate(model, test_loader, splits["test"], device)
            test_logits, test_labels = collect_logits(model, test_loader, device)
            test["calibration"] = calibration_report(test_logits, test_labels, splits["test"], temperature)
            test["confidence_states"] = confidence_state_report(
                test_logits, splits["test"], temperature, confidence_policy,
            )
        model_file = artifact_dir / "model.safetensors"
        with model_file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        version = (f"classifier-xlm-r-{'reviewed' if args.reviewed_dataset else 'synthetic-v1'}-"
                   f"{'selection-' if args.validation_only else ''}"
                   f"{args.input_length_strategy}-{args.max_length}-{checksum[:12]}-t{round(temperature * 10):03d}")
        metadata = {
            "model_version": version,
            "model_family": "xlm-roberta-sequence-classification",
            "base_model": args.base_model,
            "dataset_version": dataset_version,
            "dataset_content_sha256": dataset_checksum,
            "base_model_checksum": base_model_checksum if args.reviewed_dataset else None,
            "frozen_evaluation_version": reviewed_manifest.frozen_evaluation_version if args.reviewed_dataset else None,
            "token_audit_sha256": token_audit_checksum if args.reviewed_dataset else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"status": ("validation_only" if args.validation_only else
                                   "reviewed_synthetic_holdout_only" if args.reviewed_dataset else
                                   "synthetic_holdout_only"),
                        "validation": best_validation, "test": test},
            "confidence_policy_version": POLICY_VERSION,
            "confidence_thresholds": {
                "low_confidence_below": confidence_policy.low_confidence_below,
                "confident_at_or_above": confidence_policy.confident_at_or_above,
            },
            "confident_enabled": confidence_policy.confident_enabled,
            "confidence_policy_evidence": policy_evidence,
            "languages": (["RU", "KZ", "MIXED"] if args.reviewed_dataset else
                          [language for language in ("RU", "KZ", "MIXED")
                           if language in {row["language"] for row in splits["train"]}]),
            "labels": list(LABELS),
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "temperature": temperature,
                "seed": seed,
                "input_length_strategy": f"{args.input_length_strategy}-{args.max_length}",
                "validation_only": args.validation_only,
                "reviewed_dataset": bool(args.reviewed_dataset),
                "train_samples": len(splits["train"]),
                "train_scenarios": len({row["scenario_id"] for row in splits["train"]}),
                "class_counts": dict(Counter(row["topic_id"] for row in splits["train"])),
                "online_retraining": False,
            },
            "artifact_checksum": f"sha256:{checksum}",
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        publish_directory(artifact_dir, args.output_dir)
        print(json.dumps({"stage": "complete", "model_version": version,
                          "validation_macro_f1": best_f1,
                          "test_macro_f1": test["macro_f1"] if test is not None else None,
                          "test_accuracy": test["accuracy"] if test is not None else None}), flush=True)


if __name__ == "__main__":
    main()
