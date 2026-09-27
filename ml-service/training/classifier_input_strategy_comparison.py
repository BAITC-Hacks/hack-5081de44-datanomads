"""Select a classifier input strategy using validation-only artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


STRATEGIES = ("head-384", "head-tail-384", "head-512", "head-tail-512")
_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LINEAGE_FIELDS = (
    "model_family", "base_model", "dataset_version", "dataset_content_sha256",
    "frozen_evaluation_version", "base_model_checksum", "token_audit_sha256",
    "labels", "languages",
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("classifier manifest has duplicate JSON keys")
        result[key] = value
    return result


def _load_artifact(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("classifier manifest must be a JSON object")

    config = manifest.get("training_config")
    metrics = manifest.get("metrics")
    if not isinstance(config, dict) or not isinstance(metrics, dict):
        raise ValueError("classifier manifest is missing training config or metrics")
    if (config.get("reviewed_dataset") is not True or
            config.get("validation_only") is not True or
            metrics.get("status") != "validation_only" or
            "test" not in metrics or metrics["test"] is not None):
        raise ValueError("artifact must contain validation-only metrics with null test")

    strategy = config.get("input_length_strategy")
    max_length = config.get("max_length")
    if (strategy not in STRATEGIES or type(max_length) is not int or
            strategy.rsplit("-", 1)[-1] != str(max_length)):
        raise ValueError("artifact has an invalid input-length strategy")

    validation = metrics.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("artifact is missing validation metrics")
    macro_f1 = validation.get("macro_f1")
    if (type(macro_f1) not in (int, float) or not math.isfinite(macro_f1) or
            not 0 <= macro_f1 <= 1):
        raise ValueError("artifact has an invalid validation macro-F1")
    for field in ("sample_count", "scenario_count"):
        if type(validation.get(field)) is not int or validation[field] < 1:
            raise ValueError("artifact is missing validation sample counts")

    for field in _LINEAGE_FIELDS:
        value = manifest.get(field)
        if field in ("labels", "languages"):
            if (not isinstance(value, list) or not value or
                    any(not isinstance(item, str) or not item for item in value) or
                    len(value) != len(set(value))):
                raise ValueError("artifact has invalid label or language lineage")
        elif not isinstance(value, str) or not value:
            raise ValueError("artifact is missing required lineage")
    for field in ("dataset_content_sha256", "base_model_checksum", "token_audit_sha256"):
        if not _CHECKSUM.fullmatch(manifest[field]):
            raise ValueError("artifact has an invalid lineage checksum")
    for field in ("seed", "epochs", "batch_size"):
        if type(config.get(field)) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError("artifact has invalid training parameters")
    learning_rate = config.get("learning_rate")
    temperature = config.get("temperature")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (learning_rate, temperature)):
        raise ValueError("artifact has invalid training parameters")

    expected_checksum = manifest.get("artifact_checksum")
    if not isinstance(expected_checksum, str) or not _CHECKSUM.fullmatch(expected_checksum):
        raise ValueError("artifact has an invalid weight checksum")
    with (directory / "model.safetensors").open("rb") as stream:
        actual_checksum = f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"
    if actual_checksum != expected_checksum:
        raise ValueError("artifact weight checksum does not match manifest")

    candidate = {
        "input_length_strategy": strategy,
        "max_length": max_length,
        "validation_macro_f1": macro_f1,
        "artifact_checksum": actual_checksum,
    }
    return manifest, candidate


def compare_input_strategies(artifact_dirs: list[Path]) -> dict:
    """Recommend the best of four comparable artifacts without opening test data."""
    if len(artifact_dirs) != len(STRATEGIES):
        raise ValueError("exactly four classifier artifacts are required")

    reference_lineage = None
    reference_config = None
    reference_validation_counts = None
    candidates = {}
    for directory in artifact_dirs:
        manifest, candidate = _load_artifact(directory)
        strategy = candidate["input_length_strategy"]
        if strategy in candidates:
            raise ValueError("classifier input strategies must be unique")
        lineage = {field: manifest[field] for field in _LINEAGE_FIELDS}
        config = {key: value for key, value in manifest["training_config"].items()
                  if key not in ("max_length", "input_length_strategy", "temperature")}
        validation = manifest["metrics"]["validation"]
        validation_counts = (validation["sample_count"], validation["scenario_count"])
        if reference_lineage is None:
            reference_lineage = lineage
            reference_config = config
            reference_validation_counts = validation_counts
        elif (lineage != reference_lineage or config != reference_config or
              validation_counts != reference_validation_counts):
            raise ValueError("classifier artifacts do not share lineage and training parameters")
        candidates[strategy] = candidate

    if set(candidates) != set(STRATEGIES):
        raise ValueError("all four classifier input strategies are required")
    ordered = [candidates[strategy] for strategy in STRATEGIES]
    recommended = min(ordered, key=lambda item: (-item["validation_macro_f1"],
                                                  STRATEGIES.index(item["input_length_strategy"])))
    return {
        "report_version": "classifier-input-strategy-comparison.v1",
        "status": "VALIDATION_ONLY",
        "dataset_content_sha256": reference_lineage["dataset_content_sha256"],
        "base_model_checksum": reference_lineage["base_model_checksum"],
        "token_audit_sha256": reference_lineage["token_audit_sha256"],
        "validation_sample_count": reference_validation_counts[0],
        "validation_scenario_count": reference_validation_counts[1],
        "selection_policy": "max_validation_macro_f1_then_shorter_length_then_head",
        "recommendation": recommended,
        "candidates": ordered,
    }
