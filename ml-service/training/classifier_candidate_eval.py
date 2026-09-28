"""Compare one local classifier artifact with TF-IDF on the same frozen test."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import torch

from app.schemas import ModelMetadata
from app.trained_classifier import TrainedClassifierService
from training.classifier_baselines import evaluate_baselines, evaluate_predictions, load_verified_classifier_package
from training.dataset_builder import checksum


def _percentile(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, (percent * len(ordered) + 99) // 100 - 1)], 3)


def evaluate_candidate(package: Path, model_dir: Path, baseline_path: Path) -> dict:
    dataset, splits = load_verified_classifier_package(package)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError("TF-IDF baseline report must be a JSON object")
    model = ModelMetadata.model_validate_json((model_dir / "manifest.json").read_text(encoding="utf-8"))
    model_extra = model.model_extra or {}
    labels = sorted({row["topic_id"] for row in splits["train"]})
    if (baseline.get("report_version") != "classifier-baselines.v1" or
            baseline.get("dataset_version") != dataset.dataset_version or
            baseline.get("dataset_content_sha256") != dataset.content_sha256 or
            baseline.get("frozen_evaluation_version") != dataset.frozen_evaluation_version or
            baseline.get("frozen_evaluation_sha256") != dataset.frozen_evaluation_sha256 or
            baseline.get("synthetic") != dataset.synthetic or
            baseline.get("seed") != dataset.seed or
            baseline.get("labels") != labels or
            baseline.get("models", {}).get("tfidf_linear_svc", {}).get("test", {}).get("sample_count") != len(splits["test"])):
        raise ValueError("TF-IDF baseline does not match the frozen classifier package")
    if baseline != evaluate_baselines(package):
        raise ValueError("TF-IDF baseline metrics do not match the fixed evaluator")
    if (model.dataset_version != dataset.dataset_version or
            model_extra.get("dataset_content_sha256") != dataset.content_sha256 or
            model_extra.get("frozen_evaluation_version") != dataset.frozen_evaluation_version or
            set(model.labels) != set(labels) or
            model.metrics.get("status") != "reviewed_synthetic_holdout_only" or
            model.training_config.get("reviewed_dataset") is not True):
        raise ValueError("classifier artifact does not match the reviewed dataset package")

    classifier = TrainedClassifierService(model_dir)
    rows = splits["test"]
    for row in rows[:3]:
        classifier.classify(row["text"], language=row["language"] if row["language"] != "MIXED" else None)
    predicted = []
    state_counts = Counter({"CONFIDENT": 0, "UNCERTAIN": 0, "LOW_CONFIDENCE": 0})
    needs_review_count = 0
    latencies_ms = []
    for row in rows:
        if classifier.device.type == "cuda":
            torch.cuda.synchronize(classifier.device)
        started = perf_counter_ns()
        result = classifier.classify(row["text"], language=row["language"] if row["language"] != "MIXED" else None)
        if classifier.device.type == "cuda":
            torch.cuda.synchronize(classifier.device)
        latencies_ms.append((perf_counter_ns() - started) / 1_000_000)
        predicted.append(result.topic_id)
        state_counts[result.confidence_state] += 1
        needs_review_count += result.needs_review

    candidate = evaluate_predictions(rows, predicted, labels)
    tfidf = baseline["models"]["tfidf_linear_svc"]["test"]
    macro_f1_delta = round(candidate["macro_f1"] - tfidf["macro_f1"], 6)
    return {
        "report_version": "classifier-candidate-evaluation.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_content_sha256": dataset.content_sha256,
        "frozen_evaluation_version": dataset.frozen_evaluation_version,
        "frozen_evaluation_sha256": dataset.frozen_evaluation_sha256,
        "synthetic": dataset.synthetic,
        "model_version": model.model_version,
        "model_artifact_checksum": model.artifact_checksum,
        "baseline_report_sha256": checksum(baseline_path),
        "labels": labels,
        "sample_count": len(rows),
        "candidate_metrics": candidate,
        "tfidf_baseline_metrics": tfidf,
        "macro_f1_delta_vs_tfidf": macro_f1_delta,
        "decision": "NO_GO_BASELINE_OUTPERFORMS" if macro_f1_delta < 0 else "PENDING_HUMAN_REVIEW",
        "state_counts": dict(sorted(state_counts.items())),
        "needs_review_share": round(needs_review_count / len(rows), 6),
        "latency": {
            "scope": "single_text_end_to_end_classify",
            "device": classifier.device.type,
            "device_name": (torch.cuda.get_device_name(classifier.device) if classifier.device.type == "cuda"
                            else platform.processor() or platform.machine()),
            "machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
            "batch_size": 1,
            "warmup_samples": min(3, len(rows)),
            "timed_samples": len(latencies_ms),
            "p50_ms": _percentile(latencies_ms, 50),
            "p95_ms": _percentile(latencies_ms, 95),
        },
    }
