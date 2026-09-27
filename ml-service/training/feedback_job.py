"""Run the reviewed feedback export, dataset build and offline classifier training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE, VERSION_RE, build_candidate
from training.feedback_export import FEEDBACK_QUERY, export_feedback, load_review_links
from training.feedback_trainer import train_feedback_candidate


class FeedbackJobError(RuntimeError):
    """A stable worker error code that contains no ticket data."""


def _configured_paths() -> tuple[Path, Path, Path, Path]:
    names = (
        "PULSE_TRAINING_ROOT", "PULSE_TRAINING_REVIEW_LINKS",
        "PULSE_TRAINING_FROZEN_DATASET", "PULSE_TRAINING_PRODUCTION_MODEL",
    )
    values = [os.environ.get(name) for name in names]
    if not all(values):
        raise FeedbackJobError("TRAINER_NOT_CONFIGURED")
    root, links, frozen, production = (Path(value) for value in values)
    if not links.is_file() or not frozen.is_dir() or not production.is_dir():
        raise FeedbackJobError("TRAINING_INPUT_MISSING")
    if root.is_symlink():
        raise FeedbackJobError("INVALID_TRAINING_ROOT")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise FeedbackJobError("INVALID_TRAINING_ROOT")
    return root, links, frozen, production


async def train_classifier_job(pool: Any, payload: dict[str, Any]) -> dict:
    cycle_id = payload.get("cycle_id")
    dataset_version = payload.get("dataset_version")
    production_version = payload.get("production_model_version")
    candidate_version = payload.get("candidate_model_version")
    minimum = payload.get("min_samples")
    if (not all(isinstance(value, str) and ID_RE.fullmatch(value)
                for value in (cycle_id, production_version, candidate_version)) or
            not isinstance(dataset_version, str) or not VERSION_RE.fullmatch(dataset_version) or
            type(minimum) is not int or minimum < 1):
        raise FeedbackJobError("INVALID_JOB_PAYLOAD")
    root, review_links_path, frozen, production = _configured_paths()
    try:
        links = load_review_links(review_links_path)
    except (OSError, ValueError, TypeError):
        raise FeedbackJobError("INVALID_REVIEW_LINKS") from None
    async with pool.acquire() as connection:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            cycle_row = await connection.fetchrow(
                "SELECT state, production_model_version, candidate_dataset_version, candidate_model_version, min_feedback_count FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1",
                cycle_id,
            )
            cycle = dict(cycle_row) if cycle_row is not None else {}
            if cycle.get("state") != "TRAINING":
                raise FeedbackJobError("TRAINING_CYCLE_CHANGED")
            if (cycle.get("production_model_version") != production_version or
                    cycle.get("candidate_dataset_version") != dataset_version or
                    cycle.get("candidate_model_version") != candidate_version or
                    cycle.get("min_feedback_count") != minimum):
                raise FeedbackJobError("INVALID_JOB_PAYLOAD")
            rows = [dict(row) for row in await connection.fetch(FEEDBACK_QUERY, cycle_id)]

    export_path = root / "exports" / f"{cycle_id}.jsonl"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_report = export_feedback(rows, links, export_path, cycle_id=cycle_id,
                                    production_model_version=production_version)
    if export_report["status"] == "INSUFFICIENT_FEEDBACK":
        return {"state": "INSUFFICIENT_FEEDBACK", "cycle_id": cycle_id,
                "sample_count": 0, "required_samples": minimum,
                "rejected_counts": export_report["rejected_counts"]}
    dataset_report = build_candidate(
        export_path, frozen, root / "datasets", cycle_id=cycle_id,
        production_model_version=production_version, dataset_version=dataset_version,
        min_feedback_count=minimum,
    )
    if dataset_report["status"] == "INSUFFICIENT_FEEDBACK":
        return {"state": "INSUFFICIENT_FEEDBACK", "cycle_id": cycle_id,
                "sample_count": dataset_report["accepted_count"], "required_samples": minimum,
                "rejected_counts": dataset_report["rejected_counts"]}
    result = train_feedback_candidate(
        root / "datasets" / dataset_version, frozen, production,
        root / "models" / candidate_version, candidate_model_version=candidate_version,
    )
    return {"state": "COMPLETED", "cycle_id": cycle_id,
            "candidate_model_version": result["candidate_model_version"],
            "dataset_version": result["dataset_version"],
            "dataset_content_sha256": result["dataset_content_sha256"],
            "dataset_manifest_uri": str((root / "datasets" / dataset_version / "manifest.json").resolve()),
            "dataset_manifest_sha256": checksum(root / "datasets" / dataset_version / "manifest.json"),
            "sample_count": result["sample_count"],
            "manifest": {"artifact_uri": result["artifact_uri"] + "/manifest.json",
                         "artifact_checksum": result["artifact_checksum"]},
            "rejected_counts": dataset_report["rejected_counts"]}
