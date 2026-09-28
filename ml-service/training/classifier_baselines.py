"""Evaluate fixed classifier baselines against a reviewed, frozen package."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from training.contracts import DatasetManifest
from training.dataset_builder import SPLITS, checksum, normalized_text


def load_verified_classifier_package(package: Path) -> tuple[DatasetManifest, dict[str, list[dict]]]:
    manifest = DatasetManifest.read(package / "manifest.json")
    required_files = [f"classifier/{split}.jsonl" for split in SPLITS]
    required_files += [f"retrieval/{split}_pairs.jsonl" for split in SPLITS]
    if not manifest.synthetic or set(manifest.split_file_checksums) != set(required_files):
        raise ValueError("baseline requires a complete synthetic dataset package")
    content = hashlib.sha256()
    for name in sorted(required_files):
        digest = checksum(package / name)
        if digest != manifest.split_file_checksums[name]:
            raise ValueError("dataset split checksum mismatch")
        content.update(f"{name}\0{digest}\n".encode("utf-8"))
    for name, expected in (
        ("membership.json", manifest.membership_sha256),
        ("frozen_evaluation.json", manifest.frozen_evaluation_sha256),
        ("audit.json", manifest.audit_sha256),
    ):
        digest = checksum(package / name)
        if digest != expected:
            raise ValueError("dataset metadata checksum mismatch")
        content.update(f"{name}\0{digest}\n".encode("utf-8"))
    if "sha256:" + content.hexdigest() != manifest.content_sha256:
        raise ValueError("dataset content checksum mismatch")
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if (frozen.get("version") != manifest.frozen_evaluation_version or
            any(frozen.get("file_checksums", {}).get(name) != manifest.split_file_checksums[name]
                for name in ("classifier/test.jsonl", "retrieval/test_pairs.jsonl"))):
        raise ValueError("frozen evaluation index mismatch")
    membership = json.loads((package / "membership.json").read_text(encoding="utf-8"))
    result = {}
    seen_groups = set()
    seen_texts = set()
    for split in SPLITS:
        rows = [json.loads(line) for line in (package / f"classifier/{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if not rows or any(row.get("split") != split or row.get("review_status") != "APPROVED" for row in rows):
            raise ValueError("classifier split is empty or unapproved")
        groups = {row["scenario_id"] for row in rows}
        texts = {normalized_text(row["text"]) for row in rows}
        if seen_groups & groups or seen_texts & texts:
            raise ValueError("classifier scenario or text crosses splits")
        if (sorted(row["variant_id"] for row in rows) != membership["classifier"][split] or
                sorted(groups) != membership["classifier_groups"][split]):
            raise ValueError("classifier split membership mismatch")
        seen_groups.update(groups)
        seen_texts.update(texts)
        result[split] = rows
    if (sorted(row["variant_id"] for row in result["test"]) != frozen["classifier_ids"] or
            sorted({row["scenario_id"] for row in result["test"]}) != frozen["classifier_groups"]):
        raise ValueError("frozen classifier membership mismatch")
    return manifest, result


def _score(rows: list[dict], predictions: list[str], labels: list[str]) -> dict:
    actual = [row["topic_id"] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(actual, predictions, labels=labels, zero_division=0)
    count = len(rows)
    return {
        "sample_count": count,
        "accuracy": round(sum(a == p for a, p in zip(actual, predictions)) / count, 6),
        "macro_f1": round(float(f1.mean()), 6),
        "weighted_f1": round(float((f1 * support).sum() / count), 6),
        "per_class": {
            label: {"support": int(support[index]), "precision": round(float(precision[index]), 6),
                    "recall": round(float(recall[index]), 6), "f1": round(float(f1[index]), 6)}
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(actual, predictions, labels=labels).tolist(),
    }


def evaluate_predictions(rows: list[dict], predictions: list[str], labels: list[str]) -> dict:
    result = _score(rows, predictions, labels)
    result["by_language"] = {}
    for language in ("RU", "KZ", "MIXED"):
        indices = [index for index, row in enumerate(rows) if row["language"] == language]
        if indices:
            result["by_language"][language] = _score([rows[index] for index in indices], [predictions[index] for index in indices], labels)
    result["by_region"] = {}
    for region, count in Counter(row.get("region_id") for row in rows).items():
        if region is not None and count >= 30:
            indices = [index for index, row in enumerate(rows) if row.get("region_id") == region]
            result["by_region"][region] = _score([rows[index] for index in indices], [predictions[index] for index in indices], labels)
    return result


def evaluate_baselines(package: Path) -> dict:
    manifest, splits = load_verified_classifier_package(package)
    train = splits["train"]
    labels = sorted({row["topic_id"] for row in train})
    if len(labels) < 10 or any({row["topic_id"] for row in splits[name]} != set(labels) for name in SPLITS[1:]):
        raise ValueError("baseline requires the same 10 or more classes in every split")
    class_counts = dict(sorted(Counter(row["topic_id"] for row in train).items()))
    majority = sorted(labels, key=lambda label: (-class_counts[label], label))[0]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    train_features = vectorizer.fit_transform([row["text"] for row in train])
    classifier = LinearSVC(C=1.0, random_state=manifest.seed)
    classifier.fit(train_features, [row["topic_id"] for row in train])
    models = {"majority": {}, "tfidf_linear_svc": {}}
    for split in ("validation", "test"):
        rows = splits[split]
        models["majority"][split] = evaluate_predictions(rows, [majority] * len(rows), labels)
        features = vectorizer.transform([row["text"] for row in rows])
        models["tfidf_linear_svc"][split] = evaluate_predictions(rows, classifier.predict(features).tolist(), labels)
    return {
        "report_version": "classifier-baselines.v1",
        "dataset_version": manifest.dataset_version,
        "dataset_content_sha256": manifest.content_sha256,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "frozen_evaluation_sha256": manifest.frozen_evaluation_sha256,
        "synthetic": manifest.synthetic,
        "seed": manifest.seed,
        "labels": labels,
        "train_class_counts": class_counts,
        "class_imbalance_ratio": round(max(class_counts.values()) / min(class_counts.values()), 6),
        "tfidf_linear_svc_config": {"analyzer": "char", "ngram_range": [2, 5], "min_df": 2, "sublinear_tf": True, "C": 1.0},
        "models": models,
    }
