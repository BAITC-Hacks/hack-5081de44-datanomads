"""Aggregate token lengths from reviewed classifier train/validation splits."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum


LENGTH_LIMITS = (384, 512)


def _summarize(lengths: list[int]) -> dict:
    ordered = sorted(lengths)

    def percentile(percent: int) -> int:
        return ordered[max(0, (percent * len(ordered) + 99) // 100 - 1)]

    return {
        "sample_count": len(ordered),
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
        "exceeds_limit": {str(limit): sum(length > limit for length in ordered) for limit in LENGTH_LIMITS},
    }


def audit_token_lengths(package: Path, tokenizer_dir: Path) -> dict:
    manifest, splits = load_verified_classifier_package(package)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    if not tokenizer_dir.is_dir() or any(not (tokenizer_dir / name).is_file() for name in tokenizer_files):
        raise ValueError("tokenizer must be a complete local tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    summaries = {}
    for split in ("train", "validation"):
        lengths = defaultdict(list)
        for row in splits[split]:
            length = len(tokenizer.encode(row["text"], add_special_tokens=True, truncation=False))
            lengths["all"].append(length)
            lengths[row["language"]].append(length)
        summaries[split] = {language: _summarize(values) for language, values in sorted(lengths.items())}
    return {
        "report_version": "classifier-token-length-audit.v1",
        "dataset_version": manifest.dataset_version,
        "dataset_content_sha256": manifest.content_sha256,
        "frozen_evaluation_version": manifest.frozen_evaluation_version,
        "tokenizer_file_checksums": {name: checksum(tokenizer_dir / name) for name in tokenizer_files},
        "length_includes_special_tokens": True,
        "limits": list(LENGTH_LIMITS),
        "strategies_to_evaluate": ["head-384", "head-512", "head-tail-384", "head-tail-512"],
        "test_token_lengths_computed": False,
        "splits": summaries,
    }
