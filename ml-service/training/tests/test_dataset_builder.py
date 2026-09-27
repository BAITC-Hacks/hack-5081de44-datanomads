from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.schemas.taxonomy import TOPIC_DEFINITIONS
from scripts.pulse_sdg import CATALOG, DEFAULT_SEEDS, export_candidates, read_seeds, source_checksum
from scripts.review_retrieval_relations import CHECKS as RELATION_CHECKS
from scripts.review_retrieval_relations import export_approved as export_retrieval
from scripts.review_retrieval_relations import prepare as prepare_retrieval
from scripts.review_synthetic_classifier import CHECKS as CLASSIFIER_CHECKS
from scripts.review_synthetic_classifier import export_approved as export_classifier
from scripts.review_synthetic_classifier import prepare as prepare_classifier
from training.dataset_builder import build_package, checksum


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def fixture_inputs(root: Path, *, groups_per_topic: int, retrieval_groups: int, prefix: str,
                   topic_count: int = 10, include_repeat: bool = False) -> tuple:
    scenario_source = root / f"{prefix}_scenarios.jsonl"
    relation_source = root / f"{prefix}_relations.jsonl"
    catalog_path = root / f"{prefix}_test_catalog.json"
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    # The customer catalog has no telecom proposal; test it through a temporary
    # synthetic catalog entry without changing production source evidence.
    catalog["pairs"].append({
        "source_category": "TEST_SYNTHETIC_TELECOM",
        "source_service": "TEST_SYNTHETIC_TELECOM",
        "suggested_topic_id": "telecom",
        "suggested_subtopic_id": "test_telecom",
        "proposal_status": "CANDIDATE",
    })
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    seed_by_topic = {}
    for seed in read_seeds(DEFAULT_SEEDS).values():
        seed_by_topic.setdefault(seed["topic_id"], seed)
    for pair in catalog["pairs"]:
        topic = pair.get("suggested_topic_id")
        if (pair.get("proposal_status") == "CANDIDATE" and pair.get("suggested_subtopic_id") and
                topic and topic not in seed_by_topic):
            seed_by_topic[topic] = {
                "scenario_id": f"test_{topic}", "topic_id": topic,
                "subtopic_id": pair["suggested_subtopic_id"],
                "source_category": pair["source_category"],
                "source_service": pair["source_service"],
                "facts_ru": f"На объекте наблюдается неисправность, сценарий {topic}.",
            }
    if topic_count == len(TOPIC_DEFINITIONS):
        assert set(seed_by_topic) == {topic["id"] for topic in TOPIC_DEFINITIONS}
    scenarios = []
    candidate_rows = []
    for topic in sorted(seed_by_topic)[:topic_count]:
        for index in range(groups_per_topic):
            scenario_id = f"{prefix}_{topic}_{index}"
            seed = {**seed_by_topic[topic], "scenario_id": scenario_id}
            scenarios.append(seed)
            for language in ("RU", "KZ", "MIXED"):
                candidate_rows.append({
                    **seed,
                    "language": language,
                    "style": ("short", "conversational", "neutral")[index % 3],
                    "appeal_text": f"{seed['facts_ru']} {prefix} {index} {language}.",
                })
    write_jsonl(scenario_source, scenarios)
    classifier_candidates = root / f"{prefix}_candidates.jsonl"
    export_candidates(candidate_rows, {seed["scenario_id"]: seed for seed in scenarios},
                      classifier_candidates, "local_fixture", source_checksum(scenario_source), 109)
    classifier_review = root / f"{prefix}_classifier_review.jsonl"
    with patch("scripts.pulse_sdg.CATALOG", catalog_path):
        prepare_classifier(classifier_candidates, scenario_source, classifier_review)
    reviews = [json.loads(line) for line in classifier_review.read_text(encoding="utf-8").splitlines()]
    for row in reviews:
        row.update({"decision": "APPROVED", "reviewer_id": "reviewer_test",
                    "reviewed_at": "2026-09-27T12:00:00Z", "review_reason": "VERIFIED",
                    "checks": {check: True for check in CLASSIFIER_CHECKS}})
    write_jsonl(classifier_review, reviews)
    classifier_path = root / f"{prefix}_classifier.jsonl"
    with patch("scripts.pulse_sdg.CATALOG", catalog_path):
        export_classifier(classifier_review, classifier_candidates, scenario_source, classifier_path)

    retrieval = []
    relation_labels = {}
    for index in range(retrieval_groups):
        group = f"{prefix}_relation_{index}"
        labels = ["DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"]
        if include_repeat:
            labels.append("REPEAT")
        for label in labels:
            source_row = {
                "pair_id": f"{group}_{label.lower()}",
                "relation_group": group,
                "query_id": f"{group}_query",
                "candidate_id": f"{group}_{label.lower()}_candidate",
                "query_text": f"Обращение {prefix} {index}",
                "candidate_text": f"Проверка {prefix} {index} {label}",
                "query_language": "RU",
                "candidate_language": "RU",
                "query_context": f"Контекст обращения {prefix} {index}",
                "candidate_context": f"Контекст проверки {prefix} {index} {label}",
                "synthetic": True,
            }
            retrieval.append(source_row)
            relation_labels[source_row["pair_id"]] = label
    write_jsonl(relation_source, retrieval)
    retrieval_review = root / f"{prefix}_retrieval_review.jsonl"
    prepare_retrieval(relation_source, retrieval_review)
    reviews = [json.loads(line) for line in retrieval_review.read_text(encoding="utf-8").splitlines()]
    for row in reviews:
        label = relation_labels[row["source"]["pair_id"]]
        same = label == "DUPLICATE"
        repeat = label == "REPEAT"
        row.update({"decision": "APPROVED", "relation_label": label,
                    "reviewer_id": "reviewer_test", "reviewed_at": "2026-09-27T12:00:00Z",
                    "review_reason": "VERIFIED", "checks": {check: True for check in RELATION_CHECKS},
                    "same_region": True, "same_object": same or repeat, "same_issue": same or repeat,
                    "same_episode": same, "prior_episode_resolved": repeat})
    write_jsonl(retrieval_review, reviews)
    retrieval_path = root / f"{prefix}_retrieval.jsonl"
    export_retrieval(retrieval_review, relation_source, retrieval_path)
    classifier = [json.loads(line) for line in classifier_path.read_text(encoding="utf-8").splitlines()]
    approved_retrieval = [json.loads(line) for line in retrieval_path.read_text(encoding="utf-8").splitlines()]
    return (classifier_path, retrieval_path, scenario_source, relation_source,
            classifier, approved_retrieval, classifier_candidates, classifier_review, retrieval_review,
            catalog_path)


def build_fixture_package(inputs: tuple, output_root: Path, dataset_version: str,
                          frozen_evaluation_version: str, seed: int, *, frozen_from: Path | None = None):
    with patch("scripts.pulse_sdg.CATALOG", inputs[9]):
        return build_package(*inputs[:4], output_root, dataset_version, frozen_evaluation_version, seed,
                             classifier_candidates=inputs[6], classifier_review=inputs[7],
                             retrieval_review=inputs[8], frozen_from=frozen_from)


class DatasetBuilderTests(unittest.TestCase):
    def test_repeated_build_has_same_membership_and_frozen_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="initial")
            first = build_fixture_package(inputs, root / "one", "dataset_v1", "eval_v1", 109)
            second = build_fixture_package(inputs, root / "two", "dataset_v1", "eval_v1", 109)
            self.assertEqual(first.content_sha256, second.content_sha256)
            self.assertEqual(first.split_file_checksums, second.split_file_checksums)
            self.assertEqual(first.membership_sha256, second.membership_sha256)
            self.assertEqual(first.frozen_evaluation_sha256, second.frozen_evaluation_sha256)
            self.assertEqual(first.audit_sha256, checksum(root / "one/dataset_v1/audit.json"))
            self.assertEqual(first.record_count, 90)
            self.assertEqual(first.retrieval_pair_count, 9)
            membership = json.loads((root / "one/dataset_v1/membership.json").read_text(encoding="utf-8"))
            for task in ("classifier_groups", "retrieval_groups"):
                groups = [set(membership[task][split]) for split in ("train", "validation", "test")]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            with self.assertRaises(FileExistsError):
                build_fixture_package(inputs, root / "one", "dataset_v1", "eval_v1", 109)
            fresh = fixture_inputs(root / "one", groups_per_topic=3, retrieval_groups=3, prefix="fresh")
            with self.assertRaisesRegex(ValueError, "frozen evaluation version already exists"):
                build_fixture_package(fresh, root / "one", "dataset_v2", "eval_v1", 109)
            self.assertFalse((root / "one/dataset_v2").exists())

    def test_candidate_excludes_frozen_ids_groups_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="initial")
            build_fixture_package(initial, root, "dataset_v1", "eval_v1", 109)
            frozen_package = root / "dataset_v1"
            candidate = fixture_inputs(root, groups_per_topic=2, retrieval_groups=2, prefix="candidate")
            manifest = build_fixture_package(candidate, root, "dataset_v2", "eval_v1", 109, frozen_from=frozen_package)
            self.assertEqual(manifest.lineage["parent_dataset_version"], "dataset_v1")
            for relative in ("classifier/test.jsonl", "retrieval/test_pairs.jsonl", "frozen_evaluation.json"):
                self.assertEqual((frozen_package / relative).read_bytes(), (root / "dataset_v2" / relative).read_bytes())
            with self.assertRaisesRegex(ValueError, "overlaps frozen"):
                build_fixture_package(initial, root, "dataset_v3", "eval_v1", 109, frozen_from=frozen_package)
            self.assertFalse((root / "dataset_v3").exists())
            (frozen_package / "classifier/test.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test file checksum mismatch"):
                build_fixture_package(candidate, root, "dataset_v4", "eval_v1", 109, frozen_from=frozen_package)
            self.assertFalse((root / "dataset_v4").exists())

    def test_pending_or_wrong_provenance_cannot_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="pending")
            inputs[4][0]["review_status"] = "PENDING"
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "unapproved"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())
            inputs[4][0]["review_status"] = "APPROVED"
            inputs[4][0]["source_scenario_sha256"] = "sha256:" + "c" * 64
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "approved review records"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())
            inputs[4][0]["source_scenario_sha256"] = checksum(inputs[2])
            inputs[4][0]["text"] = "ИИН 000000000000"
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "invalid or unapproved"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())

    def test_changed_retrieval_export_cannot_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="relation")
            inputs[5][3]["query_id"] = inputs[5][0]["query_id"]
            inputs[5][3]["query_text"] = inputs[5][0]["query_text"]
            write_jsonl(inputs[1], inputs[5])
            with self.assertRaisesRegex(ValueError, "approved review records"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())

    def test_changed_review_queue_or_forged_approval_cannot_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="evidence")
            classifier = inputs[4]
            classifier[0]["review_evidence_sha256"] = "sha256:" + "f" * 64
            write_jsonl(inputs[0], classifier)
            with self.assertRaisesRegex(ValueError, "approved review records"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            classifier[0]["review_evidence_sha256"] = checksum(inputs[7])
            write_jsonl(inputs[0], classifier)

            review_rows = [json.loads(line) for line in inputs[7].read_text(encoding="utf-8").splitlines()]
            review_rows[0]["checks"]["facts_preserved"] = False
            write_jsonl(inputs[7], review_rows)
            with self.assertRaisesRegex(ValueError, "every human check"):
                build_fixture_package(inputs, root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())


if __name__ == "__main__":
    unittest.main()
