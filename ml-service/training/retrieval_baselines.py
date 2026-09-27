"""Rank reviewed retrieval candidates with lexical and local E5 baselines."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import SPLITS, checksum


GRADES = {"DUPLICATE": 2, "SIMILAR_BUT_NOT_DUPLICATE": 1, "REPEAT": 1, "UNRELATED": 0}
RANKS = (1, 3, 5)


def _load_retrieval(package: Path, membership_sha256: str) -> dict[str, list[dict]]:
    membership_path = package / "membership.json"
    if checksum(membership_path) != membership_sha256:
        raise ValueError("retrieval membership checksum mismatch")
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    result = {}
    seen_groups = set()
    seen_pair_ids = set()
    for split in SPLITS:
        rows = [json.loads(line) for line in (package / f"retrieval/{split}_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if not rows or any(row.get("split") != split or row.get("review_status") != "APPROVED" or
                           row.get("relation_label") not in GRADES for row in rows):
            raise ValueError("retrieval split is empty or unapproved")
        groups = {row["relation_group"] for row in rows}
        pair_ids = {row["pair_id"] for row in rows}
        if seen_groups & groups or seen_pair_ids & pair_ids or len(pair_ids) != len(rows):
            raise ValueError("retrieval relation group or pair crosses splits")
        if (sorted(pair_ids) != membership["retrieval"][split] or
                sorted(groups) != membership["retrieval_groups"][split]):
            raise ValueError("retrieval split membership mismatch")
        seen_groups.update(groups)
        seen_pair_ids.update(pair_ids)
        result[split] = rows
    return result


def _metric_at(ranked: list[dict], rank: int) -> tuple[float, float]:
    relevant = sum(GRADES[row["relation_label"]] > 0 for row in ranked)
    selected = ranked[:rank]
    hits = sum(GRADES[row["relation_label"]] > 0 for row in selected)
    return hits / relevant if relevant else 0.0, hits / min(rank, len(ranked))


def _dcg(rows: list[dict]) -> float:
    return sum((2 ** GRADES[row["relation_label"]] - 1) / math.log2(index + 2)
               for index, row in enumerate(rows))


def rank_metrics(rows: list[dict], scores: list[float]) -> dict:
    if len(rows) != len(scores):
        raise ValueError("retrieval scores do not match pair count")
    queries = defaultdict(list)
    for row, score in zip(rows, scores):
        queries[row["query_id"]].append((row, score))
    accumulators = {f"recall_at_{rank}": [] for rank in RANKS}
    accumulators.update({f"precision_at_{rank}": [] for rank in RANKS})
    accumulators.update({"mrr": [], "ndcg_at_5": []})
    review_queue = []
    negative_only = 0
    for query_id in sorted(queries):
        pairs = queries[query_id]
        if len(pairs) < 2 or len({row["candidate_id"] for row, _ in pairs}) != len(pairs):
            raise ValueError("retrieval query needs at least two distinct candidates")
        if len({row["query_text"] for row, _ in pairs}) != 1:
            raise ValueError("retrieval query has conflicting text")
        ranked = sorted(pairs, key=lambda item: (-item[1], item[0]["candidate_id"]))
        ranked_rows = [row for row, _ in ranked]
        positive_ranks = [index + 1 for index, row in enumerate(ranked_rows) if GRADES[row["relation_label"]] > 0]
        for rank in RANKS:
            recall, precision = _metric_at(ranked_rows, rank)
            accumulators[f"precision_at_{rank}"].append(precision)
            if positive_ranks:
                accumulators[f"recall_at_{rank}"].append(recall)
        if positive_ranks:
            accumulators["mrr"].append(1 / positive_ranks[0])
            ideal = sorted(ranked_rows, key=lambda row: -GRADES[row["relation_label"]])
            accumulators["ndcg_at_5"].append(_dcg(ranked_rows[:5]) / _dcg(ideal[:5]))
        else:
            negative_only += 1
        review_queue.append({
            "query_id": query_id,
            "review_status": "PENDING",
            "top_3": [{"candidate_id": row["candidate_id"], "score": round(float(score), 6)}
                      for row, score in ranked[:3]],
        })
    if not accumulators["mrr"]:
        raise ValueError("retrieval evaluation has no queries with relevant candidates")
    return {
        "query_count": len(queries),
        "positive_query_count": len(accumulators["mrr"]),
        "negative_only_query_count": negative_only,
        "candidate_count": len(rows),
        "metrics": {name: round(sum(values) / len(values), 6) for name, values in accumulators.items()},
        "top_3_review_queue": review_queue,
    }


def _e5_scores(rows: list[dict], model_path: Path, batch_size: int) -> tuple[list[float], str]:
    if not model_path.is_dir() or batch_size < 1:
        raise ValueError("E5 requires a local model directory and positive batch size")
    model_file = next((model_path / name for name in ("model.safetensors", "pytorch_model.bin")
                       if (model_path / name).is_file()), None)
    if model_file is None:
        raise ValueError("E5 local model weights are missing")
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def encode(values: dict[str, str], prefix: str) -> dict[str, torch.Tensor]:
        ids = sorted(values)
        embeddings = {}
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            inputs = tokenizer([prefix + values[item] for item in batch_ids], max_length=512,
                               padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                vectors = functional.normalize(pooled, p=2, dim=1).cpu()
            embeddings.update(zip(batch_ids, vectors))
        return embeddings

    queries = {row["query_id"]: row["query_text"] for row in rows}
    candidates = {row["candidate_id"]: row["candidate_text"] for row in rows}
    query_vectors = encode(queries, "query: ")
    candidate_vectors = encode(candidates, "passage: ")
    scores = [float(torch.dot(query_vectors[row["query_id"]], candidate_vectors[row["candidate_id"]])) for row in rows]
    artifact = hashlib.sha256()
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        artifact.update(f"{path.relative_to(model_path).as_posix()}\0{checksum(path)}\n".encode("utf-8"))
    return scores, "sha256:" + artifact.hexdigest()


def evaluate_retrieval_baselines(package: Path, e5_model: Path | None = None, batch_size: int = 16) -> dict:
    manifest, _ = load_verified_classifier_package(package)
    splits = _load_retrieval(package, manifest.membership_sha256)
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if (sorted(row["pair_id"] for row in splits["test"]) != frozen["retrieval_pair_ids"] or
            sorted({row["relation_group"] for row in splits["test"]}) != frozen["retrieval_groups"]):
        raise ValueError("frozen retrieval membership mismatch")
    train_texts = [text for row in splits["train"] for text in (row["query_text"], row["candidate_text"])]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    vectorizer.fit(train_texts)
    models = {"tfidf_cosine": {}}
    for split in ("validation", "test"):
        rows = splits[split]
        query_features = vectorizer.transform([row["query_text"] for row in rows])
        candidate_features = vectorizer.transform([row["candidate_text"] for row in rows])
        scores = query_features.multiply(candidate_features).sum(axis=1).A1.tolist()
        models["tfidf_cosine"][split] = rank_metrics(rows, scores)
    if e5_model is not None:
        models["multilingual_e5_base"] = {}
        evaluation_rows = splits["validation"] + splits["test"]
        scores, model_checksum = _e5_scores(evaluation_rows, e5_model, batch_size)
        validation_count = len(splits["validation"])
        for split in ("validation", "test"):
            split_scores = scores[:validation_count] if split == "validation" else scores[validation_count:]
            models["multilingual_e5_base"][split] = rank_metrics(splits[split], split_scores)
    else:
        model_checksum = None
    return {
        "report_version": "retrieval-baselines.v1",
        "dataset_version": manifest.dataset_version,
        "dataset_content_sha256": manifest.content_sha256,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "frozen_evaluation_sha256": manifest.frozen_evaluation_sha256,
        "synthetic": manifest.synthetic,
        "relevance_grades": GRADES,
        "recall_mrr_ndcg_population": "queries_with_relevant_candidates",
        "precision_population": "all_queries",
        "tfidf_config": {"analyzer": "char", "ngram_range": [2, 5], "min_df": 2, "sublinear_tf": True},
        "e5_model_status": "EVALUATED" if e5_model is not None else "NOT_RUN",
        "e5_artifact_sha256": model_checksum,
        "models": models,
    }
