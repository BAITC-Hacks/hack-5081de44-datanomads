"""Fine-tune a local E5-compatible embedder from reviewed retrieval relations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import re
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from data.normalization.pii import scan_pii
from training.classifier_baselines import load_verified_classifier_package
from training.contracts import EmbedderManifest
from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE
from training.retrieval_baselines import GRADES, _load_retrieval, model_directory_checksum, rank_metrics


MAX_LENGTH = 512
MARGIN_PER_GRADE = 0.05
LOSS_SCALE = 10.0


def _triplets(rows: list[dict]) -> list[tuple[str, str, str, int]]:
    queries = defaultdict(list)
    for row in rows:
        queries[row["query_id"]].append(row)
    triplets = []
    for query_id in sorted(queries):
        candidates = sorted(queries[query_id], key=lambda row: row["candidate_id"])
        if len({row["query_text"] for row in candidates}) != 1:
            raise ValueError("retrieval query has conflicting text")
        for positive in candidates:
            for negative in candidates:
                difference = GRADES[positive["relation_label"]] - GRADES[negative["relation_label"]]
                if difference > 0:
                    triplets.append((positive["query_text"], positive["candidate_text"],
                                     negative["candidate_text"], difference))
    if not triplets:
        raise ValueError("reviewed retrieval train split has no graded positive and hard-negative pairs")
    return triplets


def _pooled(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return functional.normalize(pooled, p=2, dim=1)


def _scores(rows: list[dict], model, tokenizer, device: torch.device, batch_size: int) -> list[float]:
    model.eval()
    queries = {row["query_id"]: row["query_text"] for row in rows}
    candidates = {row["candidate_id"]: row["candidate_text"] for row in rows}
    vectors = {}
    for prefix, values in (("query: ", queries), ("passage: ", candidates)):
        ids = sorted(values)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            inputs = tokenizer([prefix + values[item] for item in batch_ids], max_length=MAX_LENGTH,
                               padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                embeddings = _pooled(model, inputs).cpu()
            vectors.update({(prefix, item): vector for item, vector in zip(batch_ids, embeddings)})
    scores = [float(torch.dot(vectors[("query: ", row["query_id"])],
                              vectors[("passage: ", row["candidate_id"])])) for row in rows]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("embedder produced non-finite similarity scores")
    return scores


def _safe_rank_report(rows: list[dict], scores: list[float]) -> dict:
    report = rank_metrics(rows, scores)
    report["top_3_review_queue"] = [
        {"query_sha256": "sha256:" + hashlib.sha256(item["query_id"].encode()).hexdigest(),
         "review_status": "PENDING",
         "top_3": [{"candidate_sha256": "sha256:" + hashlib.sha256(candidate["candidate_id"].encode()).hexdigest(),
                    "score": candidate["score"]} for candidate in item["top_3"]]}
        for item in report["top_3_review_queue"]
    ]
    return report


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def verify_embedder_candidate(artifact: Path, *, load_model: bool = True) -> EmbedderManifest:
    if artifact.is_symlink():
        raise ValueError("embedder artifact directory must not be a symlink")
    manifest = EmbedderManifest.read(artifact / "manifest.json")
    expected = {"manifest.json", *manifest.artifact_files, *manifest.bundle_files}
    actual = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
    if actual != expected or any(path.is_symlink() for path in artifact.rglob("*")):
        raise ValueError("embedder artifact has missing, extra or linked files")
    for name, digest in {**manifest.artifact_files, **manifest.bundle_files}.items():
        if checksum(artifact / name) != digest:
            raise ValueError("embedder artifact file checksum mismatch")
    if (manifest.artifact_checksum != manifest.artifact_files.get("model.safetensors") or
            manifest.evaluation_report_sha256 != manifest.bundle_files.get("metrics.json") or
            (artifact / "artifact_checksum.txt").read_text(encoding="utf-8").strip() != manifest.artifact_checksum):
        raise ValueError("embedder artifact checksum lineage mismatch")
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "training_config.json").read_text(encoding="utf-8"))
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    if (not isinstance(metrics, dict) or metrics.get("dataset_version") != manifest.dataset_version or
            metrics.get("frozen_evaluation_sha256") != manifest.frozen_evaluation_sha256 or
            metrics.get("model_version") != manifest.model_version or
            not isinstance(test_metrics, dict) or test_metrics.get("metrics") != manifest.retrieval_metrics or
            config != manifest.training_config):
        raise ValueError("embedder artifact manifest and evaluation disagree")
    if load_model:
        tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
        model = AutoModel.from_pretrained(artifact, local_files_only=True, trust_remote_code=False).eval()
        inputs = tokenizer(["query: проверка"], max_length=MAX_LENGTH, padding=True, truncation=True,
                           return_tensors="pt")
        with torch.inference_mode():
            vector = _pooled(model, inputs)
        if (tuple(vector.shape) != (1, manifest.embedding_dimension) or
                not bool(torch.isfinite(vector).all())):
            raise ValueError("embedder artifact cannot produce a finite vector of its declared dimension")
    return manifest


def train_embedder_candidate(package: Path, base_model: Path, baseline_report: Path, output: Path, *,
                             model_version: str, base_model_id: str, epochs: int = 3,
                             batch_size: int = 2, learning_rate: float = 2e-5, seed: int = 109) -> dict:
    if (not ID_RE.fullmatch(model_version) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", base_model_id) or
            scan_pii(base_model_id).detected or epochs < 1 or batch_size < 1 or
            not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.001 or seed < 0):
        raise ValueError("invalid embedder model version or training configuration")
    if output.exists() or output.is_symlink():
        raise FileExistsError("embedder candidate artifact already exists")
    if any(output.resolve().is_relative_to(path.resolve()) for path in (package, base_model)):
        raise ValueError("embedder output must be outside immutable inputs")
    manifest, _ = load_verified_classifier_package(package)
    splits = _load_retrieval(package, manifest.membership_sha256)
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if (sorted(row["pair_id"] for row in splits["test"]) != frozen["retrieval_pair_ids"] or
            sorted({row["relation_group"] for row in splits["test"]}) != frozen["retrieval_groups"]):
        raise ValueError("frozen retrieval membership mismatch")
    triplets = _triplets(splits["train"])
    labels = {row["relation_label"] for row in splits["train"]}
    if not {"DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"}.issubset(labels):
        raise ValueError("reviewed retrieval train split lacks duplicate, related and unrelated examples")
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    base_checksum = model_directory_checksum(base_model)
    if (baseline.get("report_version") != "retrieval-baselines.v1" or
            baseline.get("dataset_version") != manifest.dataset_version or
            baseline.get("dataset_content_sha256") != manifest.content_sha256 or
            baseline.get("frozen_evaluation_version") != manifest.frozen_evaluation_version or
            baseline.get("frozen_evaluation_sha256") != manifest.frozen_evaluation_sha256 or
            baseline.get("synthetic") != manifest.synthetic or
            baseline.get("e5_model_status") != "EVALUATED" or
            baseline.get("e5_artifact_sha256") != base_checksum or
            not all(name in baseline.get("models", {}) for name in ("tfidf_cosine", "multilingual_e5_base"))):
        raise ValueError("pretrained E5 baseline report does not match reviewed dataset and local artifact")
    for model_name in ("tfidf_cosine", "multilingual_e5_base"):
        for split in ("validation", "test"):
            result = baseline["models"][model_name].get(split)
            if (not isinstance(result, dict) or result.get("candidate_count") != len(splits[split]) or
                    not isinstance(result.get("metrics"), dict)):
                raise ValueError("pretrained E5 baseline report has incomplete split metrics")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModel.from_pretrained(base_model, local_files_only=True, trust_remote_code=False)
    dimension = int(model.config.hidden_size)
    if dimension < 1:
        raise ValueError("base embedder has no valid embedding dimension")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as stage_root:
        artifact = Path(stage_root) / "artifact"
        artifact.mkdir()
        best_key = (-1.0, -1.0, -1.0)
        best_epoch = 0
        best_loss = 0.0
        for epoch in range(1, epochs + 1):
            ordered = triplets.copy()
            random.Random(f"{seed}:{epoch}").shuffle(ordered)
            model.train()
            losses = []
            for start in range(0, len(ordered), batch_size):
                batch = ordered[start:start + batch_size]
                texts = (["query: " + item[0] for item in batch] +
                         ["passage: " + item[1] for item in batch] +
                         ["passage: " + item[2] for item in batch])
                inputs = tokenizer(texts, max_length=MAX_LENGTH, padding=True, truncation=True,
                                   return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                embeddings = _pooled(model, inputs)
                query, positive, negative = embeddings.split(len(batch))
                positive_scores = (query * positive).sum(dim=1)
                negative_scores = (query * negative).sum(dim=1)
                margins = torch.tensor([MARGIN_PER_GRADE * item[3] for item in batch], device=device)
                loss = functional.softplus((negative_scores - positive_scores + margins) * LOSS_SCALE).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("embedder training produced a non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().item()))
            validation_scores = _scores(splits["validation"], model, tokenizer, device, batch_size)
            validation = _safe_rank_report(splits["validation"], validation_scores)
            metrics = validation["metrics"]
            selection_key = (metrics["ndcg_at_5"], metrics["mrr"], metrics["recall_at_1"])
            if selection_key > best_key:
                model.save_pretrained(artifact, safe_serialization=True)
                best_key = selection_key
                best_epoch = epoch
                best_loss = round(sum(losses) / len(losses), 6)

        del optimizer, model, embeddings, query, positive, negative, positive_scores, negative_scores, margins, loss, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        tokenizer.save_pretrained(artifact)
        selected = AutoModel.from_pretrained(artifact, local_files_only=True, trust_remote_code=False).to(device)
        validation = _safe_rank_report(splits["validation"],
                                       _scores(splits["validation"], selected, tokenizer, device, batch_size))
        test = _safe_rank_report(splits["test"], _scores(splits["test"], selected, tokenizer, device, batch_size))
        training_config = {
            "epochs": epochs, "selected_epoch": best_epoch, "batch_size": batch_size,
            "learning_rate": learning_rate, "seed": seed, "max_length": MAX_LENGTH,
            "loss": "graded_pairwise_softplus", "margin_per_grade": MARGIN_PER_GRADE,
            "loss_scale": LOSS_SCALE, "pooling": "attention_mask_mean", "normalization": "l2",
            "query_prefix": "query: ", "passage_prefix": "passage: ",
            "selection_metric": "validation_ndcg_at_5_then_mrr_then_recall_at_1",
            "training_triplet_count": len(triplets), "selected_epoch_train_loss": best_loss,
        }
        metrics = {
            "report_version": "embedder-candidate-evaluation.v1", "model_version": model_version,
            "dataset_version": manifest.dataset_version, "dataset_content_sha256": manifest.content_sha256,
            "frozen_evaluation_version": manifest.frozen_evaluation_version,
            "frozen_evaluation_sha256": manifest.frozen_evaluation_sha256,
            "synthetic": manifest.synthetic, "selection_split": "validation",
            "validation": validation, "test": test,
            "pretrained_e5_test": baseline["models"]["multilingual_e5_base"]["test"]["metrics"],
            "lexical_test": baseline["models"]["tfidf_cosine"]["test"]["metrics"],
            "baseline_report_sha256": checksum(baseline_report),
            "top_3_manual_review_status": "PENDING",
            "promotion_status": "PENDING_HUMAN_REVIEW",
        }
        _write_json(artifact / "metrics.json", metrics)
        _write_json(artifact / "training_config.json", training_config)
        weight_checksum = checksum(artifact / "model.safetensors")
        (artifact / "artifact_checksum.txt").write_text(weight_checksum + "\n", encoding="utf-8")
        (artifact / "MODEL_CARD.md").write_text(
            f"# {model_version}\n\nCandidate multilingual retrieval embedder. Dataset: {manifest.dataset_version}.\n"
            f"Synthetic: {str(manifest.synthetic).lower()}. Base artifact: {base_checksum}.\n"
            "Selected on validation; frozen test evaluated once. Top-3 expert review is pending.\n"
            "No quality claim on real citizen appeals or approval for production/Qdrant reindex.\n",
            encoding="utf-8",
        )
        bundle_names = {"metrics.json", "training_config.json", "artifact_checksum.txt", "MODEL_CARD.md"}
        artifact_files = {path.name: checksum(path) for path in artifact.iterdir() if path.is_file() and path.name not in bundle_names}
        bundle_files = {name: checksum(artifact / name) for name in sorted(bundle_names)}
        embedder_manifest = EmbedderManifest(
            model_version=model_version, model_family="multilingual-e5-graded-pairwise",
            base_model=base_model_id, base_model_artifact_sha256=base_checksum,
            dataset_version=manifest.dataset_version, dataset_content_sha256=manifest.content_sha256,
            frozen_evaluation_version=manifest.frozen_evaluation_version,
            frozen_evaluation_sha256=manifest.frozen_evaluation_sha256,
            evaluation_report_sha256=checksum(artifact / "metrics.json"), artifact_uri=".",
            created_at=datetime.now(timezone.utc), synthetic=manifest.synthetic, seed=seed,
            embedding_dimension=dimension, pooling="attention_mask_mean", normalization="l2",
            training_config=training_config, retrieval_metrics=test["metrics"],
            artifact_checksum=weight_checksum, artifact_files=artifact_files, bundle_files=bundle_files,
            runtime_requirements={"python": platform.python_version(), "torch": version("torch"),
                                  "transformers": version("transformers")},
        )
        embedder_manifest.write(artifact / "manifest.json")
        verify_embedder_candidate(artifact, load_model=False)
        if output.exists() or output.is_symlink():
            raise FileExistsError("embedder candidate artifact already exists")
        artifact.rename(output)
    return {"status": "COMPLETED", "model_version": model_version, "dataset_version": manifest.dataset_version,
            "artifact_checksum": weight_checksum, "artifact_uri": str(output.resolve()),
            "embedding_dimension": dimension, "promotion_status": "PENDING_HUMAN_REVIEW"}
