from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data.schemas.taxonomy import TOPIC_DEFINITIONS
from training.dataset_builder import build_package, checksum


REVIEW = {
    "synthetic": True,
    "review_status": "APPROVED",
    "reviewer_id": "reviewer_test",
    "reviewed_at": "2026-09-27T12:00:00Z",
    "review_evidence_sha256": "sha256:" + "b" * 64,
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def fixture_inputs(root: Path, *, groups_per_topic: int, retrieval_groups: int, prefix: str) -> tuple[Path, Path, Path, Path, list[dict], list[dict]]:
    scenario_source = root / f"{prefix}_scenarios.jsonl"
    relation_source = root / f"{prefix}_relations.jsonl"
    scenario_source.write_text(f"synthetic scenario source {prefix}\n", encoding="utf-8")
    relation_source.write_text(f"synthetic relation source {prefix}\n", encoding="utf-8")
    classifier = []
    for topic in sorted(topic["id"] for topic in TOPIC_DEFINITIONS)[:10]:
        for index in range(groups_per_topic):
            scenario_id = f"{prefix}_{topic}_{index}"
            for language in ("RU", "KZ", "MIXED"):
                classifier.append({
                    **REVIEW,
                    "variant_id": f"{scenario_id}_{language.lower()}",
                    "scenario_id": scenario_id,
                    "split_group": scenario_id,
                    "language": language,
                    "style": ("short", "conversational", "neutral")[index % 3],
                    "text": f"{'Проблема' if language == 'RU' else 'Мәселе'} {prefix} {topic} {index} {language}",
                    "topic_id": topic,
                    "subtopic_id": None,
                    "region_id": None,
                    "generator_model": "local_fixture",
                    "prompt_version": "v1",
                    "generator_seed": 109,
                    "source_scenario_sha256": checksum(scenario_source),
                })
    retrieval = []
    for index in range(retrieval_groups):
        group = f"{prefix}_relation_{index}"
        for label in ("DUPLICATE", "SIMILAR_BUT_NOT_DUPLICATE", "UNRELATED"):
            retrieval.append({
                **REVIEW,
                "pair_id": f"{group}_{label.lower()}",
                "relation_group": group,
                "query_id": f"{group}_query",
                "candidate_id": f"{group}_{label.lower()}_candidate",
                "query_text": f"Обращение {prefix} {index}",
                "candidate_text": f"Проверка {prefix} {index} {label}",
                "query_language": "RU",
                "candidate_language": "RU",
                "relation_label": label,
                "source_relation_sha256": checksum(relation_source),
            })
    classifier_path = root / f"{prefix}_classifier.jsonl"
    retrieval_path = root / f"{prefix}_retrieval.jsonl"
    write_jsonl(classifier_path, classifier)
    write_jsonl(retrieval_path, retrieval)
    return classifier_path, retrieval_path, scenario_source, relation_source, classifier, retrieval


class DatasetBuilderTests(unittest.TestCase):
    def test_repeated_build_has_same_membership_and_frozen_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="initial")
            first = build_package(*inputs[:4], root / "one", "dataset_v1", "eval_v1", 109)
            second = build_package(*inputs[:4], root / "two", "dataset_v1", "eval_v1", 109)
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
                build_package(*inputs[:4], root / "one", "dataset_v1", "eval_v1", 109)
            fresh = fixture_inputs(root / "one", groups_per_topic=3, retrieval_groups=3, prefix="fresh")
            with self.assertRaisesRegex(ValueError, "frozen evaluation version already exists"):
                build_package(*fresh[:4], root / "one", "dataset_v2", "eval_v1", 109)
            self.assertFalse((root / "one/dataset_v2").exists())

    def test_candidate_excludes_frozen_ids_groups_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="initial")
            build_package(*initial[:4], root, "dataset_v1", "eval_v1", 109)
            frozen_package = root / "dataset_v1"
            candidate = fixture_inputs(root, groups_per_topic=2, retrieval_groups=2, prefix="candidate")
            manifest = build_package(*candidate[:4], root, "dataset_v2", "eval_v1", 109, frozen_from=frozen_package)
            self.assertEqual(manifest.lineage["parent_dataset_version"], "dataset_v1")
            for relative in ("classifier/test.jsonl", "retrieval/test_pairs.jsonl", "frozen_evaluation.json"):
                self.assertEqual((frozen_package / relative).read_bytes(), (root / "dataset_v2" / relative).read_bytes())
            candidate[4][0]["scenario_id"] = json.loads((frozen_package / "frozen_evaluation.json").read_text())["classifier_groups"][0]
            candidate[4][0]["split_group"] = candidate[4][0]["scenario_id"]
            write_jsonl(candidate[0], candidate[4])
            with self.assertRaisesRegex(ValueError, "overlaps frozen"):
                build_package(*candidate[:4], root, "dataset_v3", "eval_v1", 109, frozen_from=frozen_package)
            self.assertFalse((root / "dataset_v3").exists())
            (frozen_package / "classifier/test.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test file checksum mismatch"):
                build_package(*candidate[:4], root, "dataset_v4", "eval_v1", 109, frozen_from=frozen_package)
            self.assertFalse((root / "dataset_v4").exists())

    def test_pending_or_wrong_provenance_cannot_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="pending")
            inputs[4][0]["review_status"] = "PENDING"
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "unapproved"):
                build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())
            inputs[4][0]["review_status"] = "APPROVED"
            inputs[4][0]["source_scenario_sha256"] = "sha256:" + "c" * 64
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "provenance checksum"):
                build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())
            inputs[4][0]["source_scenario_sha256"] = checksum(inputs[2])
            inputs[4][0]["text"] = "ИИН 000000000000"
            write_jsonl(inputs[0], inputs[4])
            with self.assertRaisesRegex(ValueError, "invalid or unapproved"):
                build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())

    def test_relation_entity_cannot_belong_to_two_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="relation")
            inputs[5][3]["query_id"] = inputs[5][0]["query_id"]
            inputs[5][3]["query_text"] = inputs[5][0]["query_text"]
            write_jsonl(inputs[1], inputs[5])
            with self.assertRaisesRegex(ValueError, "multiple relation groups"):
                build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            self.assertFalse((root / "dataset_v1").exists())


if __name__ == "__main__":
    unittest.main()
