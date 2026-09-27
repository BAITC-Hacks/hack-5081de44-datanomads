"""Build immutable synthetic classifier/retrieval packages from reviewed records."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import unicodedata
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from data.normalization.pii import scan_pii
from data.schemas.taxonomy import REGION_DEFINITIONS, TOPIC_DEFINITIONS
from scripts.review_retrieval_relations import approved_records as approved_retrieval_records
from scripts.review_synthetic_classifier import approved_records as approved_classifier_records
from training.atomic_publish import publish_directory
from training.contracts import DatasetManifest


TOPICS = {topic["id"] for topic in TOPIC_DEFINITIONS}
REGIONS = {region["id"] for region in REGION_DEFINITIONS}
VERSION_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
SPLITS = ("train", "validation", "test")
RELATIONS = {"DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.casefold()).split())


class ReviewedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    synthetic: Literal[True]
    review_status: Literal["APPROVED"]
    reviewer_id: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    review_evidence_sha256: str

    @field_validator("reviewer_id")
    @classmethod
    def valid_reviewer(cls, value: str) -> str:
        if not ID_RE.fullmatch(value):
            raise ValueError("reviewer_id must be a stable identifier")
        return value

    @field_validator("review_evidence_sha256")
    @classmethod
    def valid_evidence(cls, value: str) -> str:
        if not SHA_RE.fullmatch(value):
            raise ValueError("review evidence must be a SHA-256 reference")
        return value


class ClassifierRecord(ReviewedRecord):
    variant_id: str
    scenario_id: str
    split_group: str
    language: Literal["RU", "KZ", "MIXED"]
    style: Literal["short", "conversational", "neutral"]
    text: str = Field(min_length=1)
    topic_id: str
    subtopic_id: str | None = None
    region_id: str | None = None
    generator_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    generator_seed: int = Field(ge=0)
    source_scenario_sha256: str

    @model_validator(mode="after")
    def valid_scenario(self):
        if (not ID_RE.fullmatch(self.variant_id) or not ID_RE.fullmatch(self.scenario_id) or
                self.split_group != self.scenario_id or self.topic_id not in TOPICS or
                (self.region_id is not None and self.region_id not in REGIONS) or
                not SHA_RE.fullmatch(self.source_scenario_sha256) or scan_pii(self.text).detected):
            raise ValueError("invalid scenario, topic, region, provenance or PII")
        return self


class RetrievalPair(ReviewedRecord):
    pair_id: str
    relation_group: str
    query_id: str
    candidate_id: str
    query_text: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)
    query_language: Literal["RU", "KZ", "MIXED"]
    candidate_language: Literal["RU", "KZ", "MIXED"]
    relation_label: Literal["DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED", "REPEAT"]
    source_relation_sha256: str

    @model_validator(mode="after")
    def valid_pair(self):
        if (any(not ID_RE.fullmatch(value) for value in (self.pair_id, self.relation_group, self.query_id, self.candidate_id)) or
                self.query_id == self.candidate_id or not SHA_RE.fullmatch(self.source_relation_sha256) or
                scan_pii(self.query_text).detected or scan_pii(self.candidate_text).detected):
            raise ValueError("invalid retrieval pair, provenance or PII")
        return self


def read_jsonl(path: Path, model_type: type[ClassifierRecord] | type[RetrievalPair]) -> list:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except (ValidationError, ValueError) as error:
            raise ValueError(f"input line {line_number}: invalid or unapproved record") from error
    if not rows:
        raise ValueError("input has no approved records")
    return rows


def assign_groups(groups: set[str], seed: int, salt: str, *, frozen_from: bool) -> dict[str, str]:
    values = sorted(groups)
    random.Random(f"{seed}:{salt}").shuffle(values)
    if len(values) < (2 if frozen_from else 3):
        raise ValueError(f"{salt}: too few independent groups")
    validation_count = max(1, round(len(values) * (0.2 if frozen_from else 0.15)))
    test_count = 0 if frozen_from else max(1, round(len(values) * 0.15))
    if len(values) - validation_count - test_count < 1:
        raise ValueError(f"{salt}: no training groups after split")
    assignments = {group: "train" for group in values}
    for group in values[:validation_count]:
        assignments[group] = "validation"
    for group in values[validation_count:validation_count + test_count]:
        assignments[group] = "test"
    return assignments


def split_classifier(rows: list[ClassifierRecord], seed: int, *, frozen_from: bool) -> dict[str, list[dict]]:
    topics = defaultdict(set)
    seen_variants = set()
    scenario_topics = {}
    seen_texts = set()
    for row in rows:
        if row.variant_id in seen_variants or normalized_text(row.text) in seen_texts:
            raise ValueError("duplicate classifier ID or text")
        seen_variants.add(row.variant_id)
        seen_texts.add(normalized_text(row.text))
        previous = scenario_topics.setdefault(row.scenario_id, row.topic_id)
        if previous != row.topic_id:
            raise ValueError("scenario has conflicting topic labels")
        topics[row.topic_id].add(row.scenario_id)
    if len(topics) < 10:
        raise ValueError("classifier requires at least 10 reviewed topics")
    assignments = {}
    for topic in sorted(topics):
        assignments.update(assign_groups(topics[topic], seed, f"classifier:{topic}", frozen_from=frozen_from))
    result = {split: [] for split in SPLITS}
    for row in sorted(rows, key=lambda item: item.variant_id):
        split = assignments[row.scenario_id]
        result[split].append({**row.model_dump(mode="json"), "split": split})
    for split in SPLITS[:2] if frozen_from else SPLITS:
        if "MIXED" not in {row["language"] for row in result[split]}:
            raise ValueError(f"{split}: no reviewed MIXED-language examples")
        for topic in topics:
            languages = {row["language"] for row in result[split] if row["topic_id"] == topic}
            if not {"RU", "KZ"}.issubset(languages):
                raise ValueError(f"{split}: insufficient RU/KZ coverage for {topic}")
    return result


def split_retrieval(rows: list[RetrievalPair], seed: int, *, frozen_from: bool) -> dict[str, list[dict]]:
    seen_pairs = set()
    entities = {}
    entity_groups = {}
    groups = set()
    for row in rows:
        if row.pair_id in seen_pairs:
            raise ValueError("duplicate retrieval pair ID")
        seen_pairs.add(row.pair_id)
        groups.add(row.relation_group)
        for entity_id, text in ((row.query_id, row.query_text), (row.candidate_id, row.candidate_text)):
            previous = entities.setdefault(entity_id, normalized_text(text))
            if previous != normalized_text(text):
                raise ValueError("retrieval entity has conflicting text")
            previous_group = entity_groups.setdefault(entity_id, row.relation_group)
            if previous_group != row.relation_group:
                raise ValueError("retrieval entity belongs to multiple relation groups")
    assignments = assign_groups(groups, seed, "retrieval", frozen_from=frozen_from)
    result = {split: [] for split in SPLITS}
    entity_splits = {}
    for row in sorted(rows, key=lambda item: item.pair_id):
        split = assignments[row.relation_group]
        for entity_id in (row.query_id, row.candidate_id):
            previous = entity_splits.setdefault(entity_id, split)
            if previous != split:
                raise ValueError("retrieval entity crosses split groups")
        result[split].append({**row.model_dump(mode="json"), "split": split})
    for split in SPLITS[:2] if frozen_from else SPLITS:
        labels = {row["relation_label"] for row in result[split]}
        if not RELATIONS.issubset(labels):
            raise ValueError(f"{split}: missing required retrieval relation labels")
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_frozen(package: Path, expected_version: str) -> tuple[dict, dict[str, bytes], DatasetManifest]:
    manifest = DatasetManifest.read(package / "manifest.json")
    if manifest.frozen_evaluation_version != expected_version or not manifest.frozen_evaluation_sha256:
        raise ValueError("frozen evaluation version or checksum does not match")
    frozen_path = package / "frozen_evaluation.json"
    if checksum(frozen_path) != manifest.frozen_evaluation_sha256:
        raise ValueError("frozen evaluation index checksum mismatch")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("version") != expected_version:
        raise ValueError("frozen evaluation index version mismatch")
    paths = ("classifier/test.jsonl", "retrieval/test_pairs.jsonl")
    files = {name: (package / name).read_bytes() for name in paths}
    if any(checksum(package / name) != manifest.split_file_checksums.get(name) or
           checksum(package / name) != frozen.get("file_checksums", {}).get(name) for name in paths):
        raise ValueError("frozen evaluation test file checksum mismatch")
    return frozen, files, manifest


def _verify_reviewed_records(rows: list, approved: list[dict], record_type: type, id_field: str) -> None:
    by_id = {getattr(row, id_field): row for row in rows}
    if len(by_id) != len(rows) or len(approved) != len(rows):
        raise ValueError("reviewed input does not match approved review records")
    for record in approved:
        expected = record_type.model_validate_json(json.dumps(record, ensure_ascii=False))
        if by_id.get(getattr(expected, id_field)) != expected:
            raise ValueError("reviewed input does not match approved review records")


def build_package(
    classifier_path: Path, retrieval_path: Path, scenario_source: Path, relation_source: Path,
    output_root: Path, dataset_version: str,
    frozen_evaluation_version: str, seed: int, *, classifier_candidates: Path,
    classifier_review: Path, retrieval_review: Path, frozen_from: Path | None = None,
) -> DatasetManifest:
    if not VERSION_RE.fullmatch(dataset_version) or not VERSION_RE.fullmatch(frozen_evaluation_version) or seed < 0:
        raise ValueError("invalid dataset version, frozen version or seed")
    classifier = read_jsonl(classifier_path, ClassifierRecord)
    retrieval = read_jsonl(retrieval_path, RetrievalPair)
    _verify_reviewed_records(
        classifier, approved_classifier_records(classifier_review, classifier_candidates, scenario_source),
        ClassifierRecord, "variant_id",
    )
    _verify_reviewed_records(
        retrieval, approved_retrieval_records(retrieval_review, relation_source),
        RetrievalPair, "pair_id",
    )
    scenario_checksum = checksum(scenario_source)
    relation_checksum = checksum(relation_source)
    if (any(row.source_scenario_sha256 != scenario_checksum for row in classifier) or
            any(row.source_relation_sha256 != relation_checksum for row in retrieval)):
        raise ValueError("reviewed record provenance checksum does not match source")
    frozen = None
    frozen_files = None
    previous_manifest = None
    if frozen_from is not None:
        frozen, frozen_files, previous_manifest = _load_frozen(frozen_from, frozen_evaluation_version)
        classifier_ids = set(frozen["classifier_ids"])
        classifier_groups = set(frozen["classifier_groups"])
        retrieval_ids = set(frozen["retrieval_pair_ids"])
        retrieval_groups = set(frozen["retrieval_groups"])
        retrieval_entities = set(frozen["retrieval_entity_ids"])
        if (any(row.variant_id in classifier_ids or row.scenario_id in classifier_groups or row.variant_id in retrieval_entities for row in classifier) or
                any(row.pair_id in retrieval_ids or row.relation_group in retrieval_groups or
                    row.query_id in retrieval_entities or row.candidate_id in retrieval_entities or
                    row.query_id in classifier_ids or row.candidate_id in classifier_ids for row in retrieval)):
            raise ValueError("candidate input overlaps frozen evaluation IDs or groups")
    classifier_splits = split_classifier(classifier, seed, frozen_from=frozen is not None)
    retrieval_splits = split_retrieval(retrieval, seed, frozen_from=frozen is not None)
    if frozen is not None and frozen_files is not None:
        classifier_splits["test"] = [json.loads(line) for line in frozen_files["classifier/test.jsonl"].splitlines() if line]
        retrieval_splits["test"] = [json.loads(line) for line in frozen_files["retrieval/test_pairs.jsonl"].splitlines() if line]
    entity_splits = {}
    text_splits = {}
    for split in SPLITS:
        for row in classifier_splits[split]:
            previous = entity_splits.setdefault(row["variant_id"], split)
            if previous != split:
                raise ValueError("classifier ID crosses frozen split")
            previous_text_split = text_splits.setdefault(normalized_text(row["text"]), split)
            if previous_text_split != split:
                raise ValueError("classifier text crosses split")
        for row in retrieval_splits[split]:
            for entity_id, text in ((row["query_id"], row["query_text"]), (row["candidate_id"], row["candidate_text"])):
                previous = entity_splits.setdefault(entity_id, split)
                if previous != split:
                    raise ValueError("retrieval entity crosses classifier or frozen split")
                previous_text_split = text_splits.setdefault(normalized_text(text), split)
                if previous_text_split != split:
                    raise ValueError("retrieval text crosses split")
    if frozen is not None:
        frozen_texts = {normalized_text(row["text"]) for row in classifier_splits["test"]}
        frozen_texts.update(normalized_text(row[field]) for row in retrieval_splits["test"] for field in ("query_text", "candidate_text"))
        if (any(normalized_text(row.text) in frozen_texts for row in classifier) or
                any(normalized_text(value) in frozen_texts for row in retrieval for value in (row.query_text, row.candidate_text))):
            raise ValueError("candidate text duplicates frozen evaluation text")

    destination = output_root / dataset_version
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"dataset package already exists: {dataset_version}")
    if output_root.exists():
        for existing_path in output_root.iterdir():
            manifest_path = existing_path / "manifest.json"
            if not manifest_path.is_file():
                continue
            existing = DatasetManifest.read(manifest_path)
            if existing.frozen_evaluation_version != frozen_evaluation_version:
                continue
            if frozen is None:
                raise ValueError("frozen evaluation version already exists; use --frozen-from")
            if existing.frozen_evaluation_sha256 != previous_manifest.frozen_evaluation_sha256:
                raise ValueError("frozen evaluation version has conflicting content")

    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{dataset_version}-", dir=output_root) as stage_dir:
        output = Path(stage_dir) / dataset_version
        output.mkdir()
        (output / "classifier").mkdir()
        (output / "retrieval").mkdir()
        file_rows = {
            f"classifier/{split}.jsonl": classifier_splits[split] for split in SPLITS
        } | {
            f"retrieval/{split}_pairs.jsonl": retrieval_splits[split] for split in SPLITS
        }
        for name, rows in file_rows.items():
            _write_jsonl(output / name, rows)
        file_checksums = {name: checksum(output / name) for name in sorted(file_rows)}
        if frozen is None:
            frozen = {
                "version": frozen_evaluation_version,
                "classifier_ids": sorted(row["variant_id"] for row in classifier_splits["test"]),
                "classifier_groups": sorted({row["scenario_id"] for row in classifier_splits["test"]}),
                "retrieval_pair_ids": sorted(row["pair_id"] for row in retrieval_splits["test"]),
                "retrieval_groups": sorted({row["relation_group"] for row in retrieval_splits["test"]}),
                "retrieval_entity_ids": sorted({row[field] for row in retrieval_splits["test"] for field in ("query_id", "candidate_id")}),
                "file_checksums": {name: file_checksums[name] for name in ("classifier/test.jsonl", "retrieval/test_pairs.jsonl")},
            }
        frozen_path = output / "frozen_evaluation.json"
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        membership = {
            "classifier": {split: sorted(row["variant_id"] for row in classifier_splits[split]) for split in SPLITS},
            "retrieval": {split: sorted(row["pair_id"] for row in retrieval_splits[split]) for split in SPLITS},
            "classifier_groups": {split: sorted({row["scenario_id"] for row in classifier_splits[split]}) for split in SPLITS},
            "retrieval_groups": {split: sorted({row["relation_group"] for row in retrieval_splits[split]}) for split in SPLITS},
        }
        membership_path = output / "membership.json"
        membership_path.write_text(json.dumps(membership, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_classifier = [row for split in SPLITS for row in classifier_splits[split]]
        all_retrieval = [row for split in SPLITS for row in retrieval_splits[split]]
        audit = {
            "synthetic": True,
            "review_status": "APPROVED",
            "classifier_counts": {split: len(classifier_splits[split]) for split in SPLITS},
            "retrieval_counts": {split: len(retrieval_splits[split]) for split in SPLITS},
            "relation_labels": dict(sorted(Counter(row["relation_label"] for row in all_retrieval).items())),
            "cross_split_classifier_groups": 0,
            "cross_split_retrieval_groups": 0,
            "cross_split_entity_ids": 0,
            "frozen_exclusion_passed": frozen is not None,
        }
        audit_path = output / "audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sources = {
            "classifier_reviewed": checksum(classifier_path),
            "classifier_candidates": checksum(classifier_candidates),
            "classifier_review": checksum(classifier_review),
            "retrieval_reviewed": checksum(retrieval_path),
            "retrieval_review": checksum(retrieval_review),
            "scenario_source": scenario_checksum,
            "relation_source": relation_checksum,
        }
        lineage = {}
        if previous_manifest is not None:
            sources["frozen_package"] = checksum(frozen_from / "manifest.json")
            lineage["parent_dataset_version"] = previous_manifest.dataset_version
        content = hashlib.sha256()
        for name, digest in sorted(file_checksums.items()):
            content.update(f"{name}\0{digest}\n".encode("utf-8"))
        for name, digest in (("membership.json", checksum(membership_path)), ("frozen_evaluation.json", checksum(frozen_path)), ("audit.json", checksum(audit_path))):
            content.update(f"{name}\0{digest}\n".encode("utf-8"))
        manifest = DatasetManifest(
            dataset_version=dataset_version,
            schema_version="unified-ticket.v1",
            created_at=datetime.now(timezone.utc),
            synthetic=True,
            seed=seed,
            sources=sorted(sources),
            source_checksums=sources,
            record_count=len(all_classifier),
            quarantine_count=0,
            languages=dict(Counter(row["language"] for row in all_classifier)),
            topics=dict(Counter(row["topic_id"] for row in all_classifier)),
            regions=dict(Counter(row["region_id"] or "UNSPECIFIED" for row in all_classifier)),
            split_policy="synthetic_scenario_and_relation_group.v1",
            split_group_key="scenario_id|relation_group",
            frozen_evaluation_version=frozen_evaluation_version,
            pii_policy_version="pii-minimization.v1",
            content_sha256="sha256:" + content.hexdigest(),
            retrieval_pair_count=len(all_retrieval),
            split_file_checksums=file_checksums,
            membership_sha256=checksum(membership_path),
            frozen_evaluation_sha256=checksum(frozen_path),
            audit_sha256=checksum(audit_path),
            lineage=lineage,
        )
        manifest.write(output / "manifest.json")
        publish_directory(output, destination)
        return manifest
