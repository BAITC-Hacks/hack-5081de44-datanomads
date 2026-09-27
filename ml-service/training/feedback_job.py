"""Run one reviewed feedback cycle with immutable, retryable local artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from app.schemas import ModelMetadata
from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum
from training.classifier_pair_eval import CriticalRegressionPolicy, compare_classifiers
from training.feedback_dataset import ID_RE, VERSION_RE, build_candidate
from training.feedback_export import FEEDBACK_QUERY, export_feedback, load_review_links
from training.feedback_trainer import train_feedback_candidate


class FeedbackJobError(RuntimeError):
    """A stable worker error code that contains no ticket data."""


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise FeedbackJobError("INVALID_TRAINING_ROOT")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir() or path.stat().st_mode & 0o077:
        raise FeedbackJobError("INVALID_TRAINING_ROOT")


def _configured_paths() -> tuple[Path, Path, Path, Path, Path]:
    names = (
        "PULSE_TRAINING_ROOT", "PULSE_TRAINING_REVIEW_LINKS",
        "PULSE_TRAINING_FROZEN_DATASET", "PULSE_TRAINING_PRODUCTION_MODEL",
        "PULSE_TRAINING_CRITICAL_POLICY",
    )
    values = [os.environ.get(name) for name in names]
    if not all(values):
        raise FeedbackJobError("TRAINER_NOT_CONFIGURED")
    root, links, frozen, production, policy = (Path(value) for value in values)
    if not links.is_file() or not frozen.is_dir() or not production.is_dir() or not policy.is_file():
        raise FeedbackJobError("TRAINING_INPUT_MISSING")
    _private_directory(root)
    return root, links, frozen, production, policy


def _file_inventory(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise FeedbackJobError("INVALID_TRAINING_ARTIFACT")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise FeedbackJobError("INVALID_TRAINING_ARTIFACT")
        if path.is_file():
            inventory[path.relative_to(directory).as_posix()] = checksum(path)
    return inventory


def _input_snapshot(links: Path, frozen: Path, production: Path, policy: Path) -> dict:
    for path in (links, policy):
        if path.is_symlink() or not path.is_file():
            raise FeedbackJobError("INVALID_TRAINING_INPUT")
    return {"review_links": checksum(links), "frozen": _file_inventory(frozen),
            "production": _file_inventory(production), "critical_policy": checksum(policy)}


def _cleanup_stale_stages(stage_root: Path, cycle_id: str) -> None:
    """Remove only our interrupted private stages, while holding the cycle lock."""
    if not shutil.rmtree.avoids_symlink_attacks:
        raise FeedbackJobError("STALE_STAGE_UNSAFE")
    stale = []
    for stage in stage_root.iterdir():
        if not stage.name.startswith(f"{cycle_id}-"):
            continue
        marker = stage / "stage_marker.json"
        if (stage.is_symlink() or not stage.is_dir() or stage.stat().st_uid != os.getuid() or
                stage.stat().st_mode & 0o077):
            raise FeedbackJobError("STALE_STAGE_UNSAFE")
        entries = list(stage.iterdir())
        # A crash can happen between mkdtemp and marker creation. Retain an
        # empty unmarked directory instead of guessing that it is safe to delete.
        if not entries:
            continue
        if marker.is_symlink() or not marker.is_file():
            raise FeedbackJobError("STALE_STAGE_UNSAFE")
        try:
            identity = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if entries == [marker]:
                continue
            raise FeedbackJobError("STALE_STAGE_UNSAFE") from None
        if identity != {"kind": "pulse-feedback-stage.v1", "cycle_id": cycle_id,
                        "directory": stage.name}:
            if entries == [marker]:
                continue
            raise FeedbackJobError("STALE_STAGE_UNSAFE")
        if any(child.name not in {"stage_marker.json", "exports", "datasets", "policies",
                                  "models", "reports"} for child in entries):
            raise FeedbackJobError("STALE_STAGE_UNSAFE")
        try:
            _file_inventory(stage)
        except FeedbackJobError:
            raise FeedbackJobError("STALE_STAGE_UNSAFE") from None
        stale.append(stage)
    for stage in stale:
        shutil.rmtree(stage)


@contextmanager
def _cycle_lock(root: Path, cycle_id: str) -> Iterator[None]:
    lock_dir = root / "locks"
    _private_directory(lock_dir)
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    descriptor = os.open(lock_dir / f"{cycle_id}.lock", flags, 0o600)
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise FeedbackJobError("INVALID_CYCLE_LOCK")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FeedbackJobError("TRAINING_CYCLE_BUSY") from None
        yield
    finally:
        os.close(descriptor)


def _publish_directory(stage: Path, destination: Path) -> None:
    """Linux renameat2 makes publication atomic without replacing an existing cycle."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise FeedbackJobError("ATOMIC_PUBLISH_UNAVAILABLE")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL):
            raise FeedbackJobError("ATOMIC_PUBLISH_UNAVAILABLE")
        if code == errno.EEXIST:
            raise FeedbackJobError("TRAINING_CYCLE_EXISTS")
        raise OSError(code, os.strerror(code))


def _result_for_cycle(artifact_dir: Path, published_dir: Path, training_result: dict, offline: dict,
                      rejected_counts: dict, cycle_id: str) -> dict:
    dataset_version = training_result["dataset_version"]
    model_version = training_result["candidate_model_version"]
    dataset_manifest = artifact_dir / "datasets" / dataset_version / "manifest.json"
    report_path = artifact_dir / "reports" / "offline.json"
    return {"state": "COMPLETED", "cycle_id": cycle_id,
            "candidate_model_version": model_version,
            "dataset_version": dataset_version,
            "dataset_content_sha256": training_result["dataset_content_sha256"],
            "dataset_manifest_uri": str((published_dir / "datasets" / dataset_version / "manifest.json").resolve()),
            "dataset_manifest_sha256": checksum(dataset_manifest),
            "sample_count": training_result["sample_count"],
            "manifest": {"artifact_uri": str((published_dir / "models" / model_version / "manifest.json").resolve()),
                         "artifact_checksum": training_result["artifact_checksum"]},
            "offline_metrics": offline,
            "offline_report_uri": str((published_dir / "reports" / "offline.json").resolve()),
            "offline_report_sha256": checksum(report_path),
            "rejected_counts": rejected_counts}


def _verify_cycle(cycle_dir: Path, payload: dict, snapshot: dict,
                  current_export_sha256: str, feedback_rows_sha256: str, frozen: Path) -> dict:
    try:
        if cycle_dir.is_symlink() or not cycle_dir.is_dir():
            raise ValueError("invalid cycle directory")
        record_path = cycle_dir / "cycle_record.json"
        if record_path.is_symlink():
            raise ValueError("invalid cycle record")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        inventory = _file_inventory(cycle_dir)
        inventory.pop("cycle_record.json")
        if (record["payload"] != payload or record["inputs"] != snapshot or
                record["files"] != inventory or
                record["export_sha256"] != current_export_sha256 or
                record["feedback_rows_sha256"] != feedback_rows_sha256):
            raise ValueError("cycle inputs or artifacts changed")
        result = record["result"]
        dataset = cycle_dir / "datasets" / payload["dataset_version"]
        model = cycle_dir / "models" / payload["candidate_model_version"]
        report = cycle_dir / "reports" / "offline.json"
        dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        model_manifest = ModelMetadata.model_validate_json((model / "manifest.json").read_text(encoding="utf-8"))
        offline = json.loads(report.read_text(encoding="utf-8"))
        expected = _result_for_cycle(cycle_dir, cycle_dir, {
            "dataset_version": dataset_manifest["candidate_dataset_version"],
            "candidate_model_version": model_manifest.model_version,
            "dataset_content_sha256": dataset_manifest["content_sha256"],
            "artifact_checksum": model_manifest.artifact_checksum,
            "sample_count": dataset_manifest["record_count"],
        }, offline, dataset_manifest["rejected_counts"], payload["cycle_id"])
        if (expected != result or model_manifest.artifact_checksum != checksum(model / "model.safetensors") or
                dataset_manifest["cycle_id"] != payload["cycle_id"] or
                dataset_manifest["production_model_version"] != payload["production_model_version"] or
                dataset_manifest["minimum_feedback_count"] != payload["min_samples"] or
                model_manifest.base_model != payload["production_model_version"] or
                (model_manifest.model_extra or {}).get("status") != "CANDIDATE" or
                offline["candidate"]["model_version"] != payload["candidate_model_version"] or
                offline["candidate"]["artifact_checksum"] != model_manifest.artifact_checksum or
                offline["production"]["model_version"] != payload["production_model_version"]):
            raise ValueError("cycle lineage changed")
        # The dataset verifier checks content SHA, feedback lineage and frozen exclusions.
        from training.feedback_dataset import load_verified_candidate
        load_verified_candidate(dataset, frozen)
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError, FeedbackJobError):
        raise FeedbackJobError("TRAINING_CYCLE_ARTIFACT_CHANGED") from None


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
    identity = {"cycle_id": cycle_id, "dataset_version": dataset_version,
                "production_model_version": production_version,
                "candidate_model_version": candidate_version, "min_samples": minimum}
    root, review_links_path, frozen, production, policy_path = _configured_paths()
    with _cycle_lock(root, cycle_id):
        stage_root = root / ".staging"
        _private_directory(stage_root)
        _cleanup_stale_stages(stage_root, cycle_id)
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
        feedback_rows_sha256 = "sha256:" + hashlib.sha256(json.dumps(
            rows, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":")).encode("utf-8")).hexdigest()

        if rows:
            try:
                policy_text = policy_path.read_text(encoding="utf-8")
                policy = CriticalRegressionPolicy.model_validate_json(policy_text)
            except (OSError, ValueError, TypeError):
                raise FeedbackJobError("INVALID_CRITICAL_POLICY") from None
            try:
                frozen_manifest, frozen_splits = load_verified_classifier_package(frozen)
                production_manifest = ModelMetadata.model_validate_json(
                    (production / "manifest.json").read_text(encoding="utf-8")
                )
                train_labels = {row["topic_id"] for row in frozen_splits["train"]}
                test_labels = {row["topic_id"] for row in frozen_splits["test"]}
                if (production_manifest.model_version != production_version or
                        (production_manifest.model_extra or {}).get("frozen_evaluation_version") !=
                        frozen_manifest.frozen_evaluation_version or
                        set(production_manifest.labels) != train_labels or
                        test_labels != train_labels or
                        production_manifest.artifact_checksum != checksum(production / "model.safetensors")):
                    raise ValueError("production artifact does not match frozen evaluation")
            except (OSError, ValueError, KeyError, TypeError):
                raise FeedbackJobError("INVALID_PRODUCTION_ARTIFACT") from None
            if not set(policy.critical_topics).issubset(train_labels):
                raise FeedbackJobError("INVALID_CRITICAL_POLICY")

        snapshot = _input_snapshot(review_links_path, frozen, production, policy_path)
        cycle_root = root / "cycles"
        _private_directory(cycle_root)
        destination = cycle_root / cycle_id
        with TemporaryDirectory(prefix=f"{cycle_id}-", dir=stage_root) as stage_name:
            stage = Path(stage_name)
            (stage / "stage_marker.json").write_text(json.dumps({
                "kind": "pulse-feedback-stage.v1", "cycle_id": cycle_id,
                "directory": stage.name,
            }), encoding="utf-8")
            export_path = stage / "exports" / "feedback.jsonl"
            export_path.parent.mkdir(mode=0o700)
            export_report = export_feedback(rows, links, export_path, cycle_id=cycle_id,
                                            production_model_version=production_version)
            if destination.exists() or destination.is_symlink():
                if export_report["status"] != "COMPLETED":
                    raise FeedbackJobError("TRAINING_CYCLE_ARTIFACT_CHANGED")
                return _verify_cycle(destination, identity, snapshot, checksum(export_path),
                                     feedback_rows_sha256, frozen)
            if export_report["status"] == "INSUFFICIENT_FEEDBACK":
                return {"state": "INSUFFICIENT_FEEDBACK", "cycle_id": cycle_id,
                        "sample_count": 0, "required_samples": minimum,
                        "rejected_counts": export_report["rejected_counts"]}
            dataset_report = build_candidate(
                export_path, frozen, stage / "datasets", cycle_id=cycle_id,
                production_model_version=production_version, dataset_version=dataset_version,
                min_feedback_count=minimum,
            )
            if dataset_report["status"] == "INSUFFICIENT_FEEDBACK":
                return {"state": "INSUFFICIENT_FEEDBACK", "cycle_id": cycle_id,
                        "sample_count": dataset_report["accepted_count"], "required_samples": minimum,
                        "rejected_counts": dataset_report["rejected_counts"]}
            policy_snapshot = stage / "policies" / "critical.json"
            policy_snapshot.parent.mkdir(mode=0o700)
            with policy_snapshot.open("x", encoding="utf-8") as stream:
                stream.write(policy_text)
            candidate_path = stage / "models" / candidate_version
            result = train_feedback_candidate(
                stage / "datasets" / dataset_version, frozen, production,
                candidate_path, candidate_model_version=candidate_version,
            )
            offline = compare_classifiers(frozen, production, candidate_path, policy_snapshot)
            report_path = stage / "reports" / "offline.json"
            report_path.parent.mkdir(mode=0o700)
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(offline, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                stream.write("\n")
            if _input_snapshot(review_links_path, frozen, production, policy_path) != snapshot:
                raise FeedbackJobError("TRAINING_INPUT_CHANGED")
            final_result = _result_for_cycle(stage, destination, result, offline,
                                             dataset_report["rejected_counts"], cycle_id)
            record = {"payload": identity, "inputs": snapshot,
                      "export_sha256": checksum(export_path),
                      "feedback_rows_sha256": feedback_rows_sha256,
                      "files": _file_inventory(stage),
                      "result": final_result}
            with (stage / "cycle_record.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                stream.write("\n")
            _publish_directory(stage, destination)
            return _verify_cycle(destination, identity, snapshot, record["export_sha256"],
                                 feedback_rows_sha256, frozen)
