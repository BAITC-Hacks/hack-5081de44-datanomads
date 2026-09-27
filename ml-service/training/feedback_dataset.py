"""Build a versioned classifier candidate from validated operator feedback."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from data.normalization.pii import scan_pii
from data.schemas.taxonomy import TOPIC_DEFINITIONS
from training.atomic_publish import publish_directory
from training.classifier_baselines import load_verified_classifier_package
from training.dataset_builder import checksum, normalized_text
from training.contracts import _checksum


TOPICS = {topic["id"] for topic in TOPIC_DEFINITIONS}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
VERSION_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


class PredictionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    topic_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class OperatorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_id: str
    action: Literal["confirm", "correct"]
    topic_id: str = Field(min_length=1)

    @field_validator("decision_id")
    @classmethod
    def valid_decision_id(cls, value: str) -> str:
        if not ID_RE.fullmatch(value):
            raise ValueError("decision_id must be stable")
        return value


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["learning-feedback-export.v1", "learning-feedback-export.v2"]
    feedback_id: str
    cycle_id: str
    ticket_id: str
    split_group: str
    source_dataset_version: str | None = None
    source_origin_kind: Literal["IMPORTED_DATASET", "RUNTIME_API"] = "IMPORTED_DATASET"
    is_synthetic: bool
    original_text: str = Field(min_length=1, max_length=10000)
    language: Literal["RU", "KZ", "MIXED"]
    production_model_version: str = Field(min_length=1)
    production_prediction: PredictionEvidence
    operator_confirmed_decision: OperatorDecision
    accepted_or_corrected: Literal["ACCEPTED", "CORRECTED"]
    feedback_created_at: AwareDatetime
    validation_status: str = Field(min_length=1)

    @field_validator("feedback_id", "cycle_id", "ticket_id", "split_group")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not ID_RE.fullmatch(value):
            raise ValueError("identifier must be stable")
        return value

    @model_validator(mode="after")
    def valid_source(self):
        imported = self.contract_version == "learning-feedback-export.v1"
        if imported != (self.source_origin_kind == "IMPORTED_DATASET"):
            raise ValueError("feedback origin and contract version differ")
        if imported != (self.source_dataset_version is not None):
            raise ValueError("feedback dataset lineage is invalid")
        if self.source_dataset_version is not None and not ID_RE.fullmatch(self.source_dataset_version):
            raise ValueError("source dataset version must be stable")
        return self


class CandidateSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feedback_id: str
    ticket_id: str
    split_group: str
    source_dataset_version: str | None = None
    source_origin_kind: Literal["IMPORTED_DATASET", "RUNTIME_API"] = "IMPORTED_DATASET"
    is_synthetic: bool
    text: str = Field(min_length=1, max_length=10000)
    language: Literal["RU", "KZ", "MIXED"]
    topic_id: str
    operator_confirmed_decision: OperatorDecision
    production_model_version: str
    production_prediction: PredictionEvidence
    accepted_or_corrected: Literal["ACCEPTED", "CORRECTED"]
    feedback_created_at: AwareDatetime

    @field_validator("feedback_id", "ticket_id", "split_group")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not ID_RE.fullmatch(value):
            raise ValueError("identifier must be stable")
        return value

    @model_validator(mode="after")
    def valid_source(self):
        imported = self.source_origin_kind == "IMPORTED_DATASET"
        if imported != (self.source_dataset_version is not None):
            raise ValueError("candidate source lineage is invalid")
        if self.source_dataset_version is not None and not ID_RE.fullmatch(self.source_dataset_version):
            raise ValueError("source dataset version must be stable")
        return self


class FeedbackCandidateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    manifest_version: Literal["feedback-candidate.v1"]
    source_contract_version: Literal["learning-feedback-export.v1", "learning-feedback-export.v2", "learning-feedback-export.mixed.v1"]
    candidate_dataset_version: str
    cycle_id: str
    production_model_version: str
    record_count: int = Field(ge=1)
    minimum_feedback_count: int = Field(ge=1)
    source_feedback_sha256: str
    source_feedback_ids: list[str] = Field(min_length=1)
    source_dataset_versions: list[str]
    runtime_ticket_ids: list[str] = Field(default_factory=list)
    origin_counts: dict[str, int]
    rejected_counts: dict[str, int]
    frozen_package: dict[str, str]
    split_policy: Literal["exported_split_group_frozen_id_text_exclusion.v1"]
    train_sha256: str
    content_sha256: str

    @field_validator("source_feedback_sha256", "train_sha256", "content_sha256")
    @classmethod
    def valid_checksum(cls, value: str) -> str:
        return _checksum(value)

    @field_validator("candidate_dataset_version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        if not VERSION_RE.fullmatch(value):
            raise ValueError("candidate dataset version is invalid")
        return value


def load_verified_candidate(package: Path, frozen_package: Path) -> tuple[FeedbackCandidateManifest, list[CandidateSample]]:
    """Reject changed feedback packages and frozen-test overlap before training."""
    manifest_path = package / "manifest.json"
    train_path = package / "train.jsonl"
    if (package.is_symlink() or manifest_path.is_symlink() or train_path.is_symlink() or
            {path.name for path in package.iterdir()} != {"manifest.json", "train.jsonl"}):
        raise ValueError("feedback candidate package layout is invalid")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_data = json.loads(manifest_text, object_pairs_hook=_unique_object)
    manifest = FeedbackCandidateManifest.model_validate_json(manifest_text)
    canonical = json.dumps({key: value for key, value in manifest_data.items() if key != "content_sha256"},
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if ("sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest() != manifest.content_sha256 or
            checksum(train_path) != manifest.train_sha256 or
            package.name != manifest.candidate_dataset_version or
            manifest.record_count < manifest.minimum_feedback_count):
        raise ValueError("feedback candidate manifest or training file checksum mismatch")
    frozen, blocked_ids, blocked_groups, blocked_texts = _frozen_exclusions(frozen_package)
    if manifest.frozen_package != frozen:
        raise ValueError("feedback candidate frozen evaluation lineage mismatch")
    samples = []
    seen_feedback = set()
    seen_tickets = set()
    seen_texts = set()
    with train_path.open(encoding="utf-8") as stream:
        for line in stream:
            json.loads(line, object_pairs_hook=_unique_object)
            sample = CandidateSample.model_validate_json(line)
            text = normalized_text(sample.text)
            if (not text or sample.feedback_id in seen_feedback or sample.ticket_id in seen_tickets or
                    text in seen_texts or sample.feedback_id in blocked_ids or sample.ticket_id in blocked_ids or
                    sample.split_group in blocked_groups or text in blocked_texts or
                    scan_pii(sample.text).detected or sample.topic_id not in TOPICS or
                    sample.topic_id != sample.operator_confirmed_decision.topic_id or
                    sample.production_model_version != manifest.production_model_version or
                    sample.accepted_or_corrected != ("ACCEPTED" if sample.operator_confirmed_decision.action == "confirm" else "CORRECTED")):
                raise ValueError("feedback candidate contains invalid or frozen training sample")
            seen_feedback.add(sample.feedback_id)
            seen_tickets.add(sample.ticket_id)
            seen_texts.add(text)
            samples.append(sample)
    if (len(samples) != manifest.record_count or
            sorted(seen_feedback) != manifest.source_feedback_ids or
            sorted({sample.source_dataset_version for sample in samples if sample.source_dataset_version is not None}) != manifest.source_dataset_versions or
            sorted(sample.ticket_id for sample in samples if sample.source_origin_kind == "RUNTIME_API") != manifest.runtime_ticket_ids or
            manifest.source_contract_version != (
                "learning-feedback-export.mixed.v1" if len({sample.source_origin_kind for sample in samples}) == 2
                else "learning-feedback-export.v2" if samples[0].source_origin_kind == "RUNTIME_API"
                else "learning-feedback-export.v1") or
            dict(sorted(Counter("synthetic" if sample.is_synthetic else "real" for sample in samples).items())) != manifest.origin_counts or
            any(count < 0 for count in manifest.rejected_counts.values())):
        raise ValueError("feedback candidate sample counts or lineage mismatch")
    return manifest, samples


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_feedback(path: Path) -> tuple[list[FeedbackRecord], Counter, str]:
    records = []
    rejected: Counter = Counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                rejected["INVALID_UTF8"] += 1
                continue
            if not line.strip():
                continue
            try:
                json.loads(line, object_pairs_hook=_unique_object)
            except (ValueError, json.JSONDecodeError):
                rejected["INVALID_JSON"] += 1
                continue
            try:
                records.append(FeedbackRecord.model_validate_json(line))
            except ValidationError:
                rejected["INVALID_SCHEMA"] += 1
    if not records and not rejected:
        raise ValueError("feedback export is empty")
    return records, rejected, "sha256:" + digest.hexdigest()


def _frozen_exclusions(package: Path) -> tuple[dict, set[str], set[str], set[str]]:
    manifest, splits = load_verified_classifier_package(package)
    frozen = json.loads((package / "frozen_evaluation.json").read_text(encoding="utf-8"))
    retrieval = [
        json.loads(line) for line in (package / "retrieval/test_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    blocked_ids = set(frozen["classifier_ids"]) | set(frozen["retrieval_entity_ids"]) | set(frozen["retrieval_pair_ids"])
    blocked_groups = set(frozen["classifier_groups"]) | set(frozen["retrieval_groups"])
    blocked_texts = {normalized_text(row["text"]) for row in splits["test"]}
    blocked_texts.update(normalized_text(row[field]) for row in retrieval for field in ("query_text", "candidate_text"))
    return {
        "dataset_version": manifest.dataset_version,
        "evaluation_version": manifest.frozen_evaluation_version,
        "content_sha256": manifest.content_sha256,
        "evaluation_sha256": manifest.frozen_evaluation_sha256,
        "manifest_sha256": checksum(package / "manifest.json"),
    }, blocked_ids, blocked_groups, blocked_texts


def _deduplicate(records: list[FeedbackRecord], rejected: Counter) -> list[FeedbackRecord]:
    by_feedback_id = defaultdict(list)
    for record in records:
        by_feedback_id[record.feedback_id].append(record)
    unique = []
    for feedback_id in sorted(by_feedback_id):
        versions = by_feedback_id[feedback_id]
        if any(version != versions[0] for version in versions[1:]):
            rejected["CONFLICTING_FEEDBACK_ID"] += len(versions)
            continue
        rejected["DUPLICATE_FEEDBACK_ID"] += len(versions) - 1
        unique.append(versions[0])
    by_ticket = defaultdict(list)
    for record in unique:
        by_ticket[record.ticket_id].append(record)
    latest = []
    for ticket_id in sorted(by_ticket):
        history = sorted(by_ticket[ticket_id], key=lambda row: (row.feedback_created_at, row.feedback_id))
        rejected["SUPERSEDED_TICKET_FEEDBACK"] += len(history) - 1
        latest.append(history[-1])
    return latest


def _sample(record: FeedbackRecord) -> dict:
    sample = {
        "feedback_id": record.feedback_id,
        "ticket_id": record.ticket_id,
        "split_group": record.split_group,
        "is_synthetic": record.is_synthetic,
        "text": record.original_text.strip(),
        "language": record.language,
        "topic_id": record.operator_confirmed_decision.topic_id,
        "operator_confirmed_decision": record.operator_confirmed_decision.model_dump(mode="json"),
        "production_model_version": record.production_model_version,
        "production_prediction": record.production_prediction.model_dump(mode="json"),
        "accepted_or_corrected": record.accepted_or_corrected,
        "feedback_created_at": record.feedback_created_at.isoformat(),
    }
    if record.source_origin_kind == "RUNTIME_API":
        sample["source_origin_kind"] = "RUNTIME_API"
    else:
        sample["source_dataset_version"] = record.source_dataset_version
    return sample


def build_candidate(
    feedback_path: Path, frozen_package: Path, output_root: Path, *,
    cycle_id: str, production_model_version: str, dataset_version: str, min_feedback_count: int,
) -> dict:
    if (not ID_RE.fullmatch(cycle_id) or not ID_RE.fullmatch(production_model_version) or
            not VERSION_RE.fullmatch(dataset_version) or min_feedback_count < 1):
        raise ValueError("invalid cycle, production model, dataset version or minimum feedback count")
    frozen, blocked_ids, blocked_groups, blocked_texts = _frozen_exclusions(frozen_package)
    records, rejected, source_sha = _read_feedback(feedback_path)
    input_record_count = len(records) + sum(rejected.values())
    matching_cycle = []
    for record in records:
        if record.cycle_id == cycle_id:
            matching_cycle.append(record)
        else:
            rejected["WRONG_CYCLE"] += 1
    latest = _deduplicate(matching_cycle, rejected)
    accepted = []
    seen_texts = set()
    for record in sorted(latest, key=lambda row: row.feedback_id):
        if record.production_model_version != production_model_version:
            rejected["WRONG_PRODUCTION_MODEL"] += 1
        elif record.validation_status != "VALID":
            rejected["UNVALIDATED_FEEDBACK"] += 1
        elif record.operator_confirmed_decision.topic_id not in TOPICS:
            rejected["UNKNOWN_CONFIRMED_TOPIC"] += 1
        elif record.accepted_or_corrected != ("ACCEPTED" if record.operator_confirmed_decision.action == "confirm" else "CORRECTED"):
            rejected["DECISION_ACTION_MISMATCH"] += 1
        elif scan_pii(record.original_text).detected:
            rejected["PII_DETECTED"] += 1
        elif record.ticket_id in blocked_ids or record.feedback_id in blocked_ids or record.split_group in blocked_groups:
            rejected["FROZEN_ID_OR_GROUP"] += 1
        elif normalized_text(record.original_text) in blocked_texts:
            rejected["FROZEN_TEXT"] += 1
        elif normalized_text(record.original_text) in seen_texts:
            rejected["DUPLICATE_TEXT"] += 1
        else:
            accepted.append(_sample(record))
            seen_texts.add(normalized_text(record.original_text))
    report = {
        "status": "INSUFFICIENT_FEEDBACK" if len(accepted) < min_feedback_count else "COMPLETED",
        "candidate_dataset_version": dataset_version,
        "cycle_id": cycle_id,
        "input_record_count": input_record_count,
        "accepted_count": len(accepted),
        "required_count": min_feedback_count,
        "rejected_counts": dict(sorted((reason, count) for reason, count in rejected.items() if count)),
        "source_feedback_sha256": source_sha,
        "frozen_evaluation_version": frozen["evaluation_version"],
    }
    if input_record_count != len(accepted) + sum(report["rejected_counts"].values()):
        raise ValueError("feedback accounting does not cover every input record")
    if report["status"] != "COMPLETED":
        return report
    destination = output_root / dataset_version
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("candidate dataset already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{dataset_version}-", dir=output_root) as stage_root:
        package = Path(stage_root) / dataset_version
        package.mkdir()
        train_path = package / "train.jsonl"
        with train_path.open("x", encoding="utf-8", newline="\n") as stream:
            for sample in accepted:
                stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        accepted_feedback_ids = {row["feedback_id"] for row in accepted}
        contract_versions = {record.contract_version for record in latest if record.feedback_id in accepted_feedback_ids}
        manifest = {
            "manifest_version": "feedback-candidate.v1",
            "source_contract_version": (next(iter(contract_versions)) if len(contract_versions) == 1
                                        else "learning-feedback-export.mixed.v1"),
            "candidate_dataset_version": dataset_version,
            "cycle_id": cycle_id,
            "production_model_version": production_model_version,
            "record_count": len(accepted),
            "minimum_feedback_count": min_feedback_count,
            "source_feedback_sha256": report["source_feedback_sha256"],
            "source_feedback_ids": sorted(row["feedback_id"] for row in accepted),
            "source_dataset_versions": sorted({row["source_dataset_version"] for row in accepted
                                               if row.get("source_dataset_version") is not None}),
            "origin_counts": dict(sorted(Counter("synthetic" if row["is_synthetic"] else "real" for row in accepted).items())),
            "rejected_counts": report["rejected_counts"],
            "frozen_package": frozen,
            "split_policy": "exported_split_group_frozen_id_text_exclusion.v1",
            "train_sha256": checksum(train_path),
        }
        runtime_ticket_ids = sorted(row["ticket_id"] for row in accepted
                                    if row.get("source_origin_kind") == "RUNTIME_API")
        if runtime_ticket_ids:
            manifest["runtime_ticket_ids"] = runtime_ticket_ids
        canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        manifest["content_sha256"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with (package / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        report["content_sha256"] = manifest["content_sha256"]
        publish_directory(package, destination)
        return report
