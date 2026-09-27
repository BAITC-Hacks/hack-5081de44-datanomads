from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import unicodedata

from training.dataset_builder import checksum
from training.feedback_dataset import build_candidate, load_verified_candidate
from training.tests.test_dataset_builder import build_fixture_package, fixture_inputs


BASE = {
    "contract_version": "learning-feedback-export.v1",
    "cycle_id": "cycle_1",
    "source_dataset_version": "source_demo_v1",
    "is_synthetic": True,
    "language": "RU",
    "production_model_version": "classifier_production_v1",
    "production_prediction": {"topic_id": "water_supply", "confidence": 0.8},
    "accepted_or_corrected": "CORRECTED",
    "feedback_created_at": "2026-09-27T12:00:00+05:00",
    "validation_status": "VALID",
}


def feedback(index: int, *, confirmed_topic: str = "roads") -> dict:
    return {
        **BASE,
        "feedback_id": f"feedback_{index}",
        "ticket_id": f"ticket_{index}",
        "split_group": f"incident_{index}",
        "original_text": f"На дороге рядом с остановкой новая яма, случай {index}.",
        "operator_confirmed_decision": {
            "decision_id": f"decision_{index}",
            "action": "correct",
            "topic_id": confirmed_topic,
        },
    }


def runtime_feedback(index: int) -> dict:
    row = feedback(index)
    row["contract_version"] = "learning-feedback-export.v2"
    row["source_origin_kind"] = "RUNTIME_API"
    del row["source_dataset_version"]
    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FeedbackCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frozen_root = self.root / "frozen"
        inputs = fixture_inputs(self.root, groups_per_topic=3, retrieval_groups=3, prefix="initial")
        build_fixture_package(inputs, self.frozen_root, "reviewed_v1", "eval_v1", 109)
        self.frozen = self.frozen_root / "reviewed_v1"
        self.input = self.root / "feedback.jsonl"

    def build(self, output: Path, *, minimum: int = 1) -> dict:
        return build_candidate(
            self.input, self.frozen, output,
            cycle_id="cycle_1", production_model_version="classifier_production_v1",
            dataset_version="candidate_v1", min_feedback_count=minimum,
        )

    def test_candidate_is_reproducible_and_uses_operator_label(self) -> None:
        write_jsonl(self.input, [feedback(2, confirmed_topic="electricity"), feedback(1)])
        first = self.build(self.root / "one", minimum=2)
        second = self.build(self.root / "two", minimum=2)
        self.assertEqual(first["status"], "COMPLETED")
        self.assertEqual(first["input_record_count"], 2)
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        first_package = self.root / "one/candidate_v1"
        second_package = self.root / "two/candidate_v1"
        self.assertEqual((first_package / "manifest.json").read_bytes(), (second_package / "manifest.json").read_bytes())
        self.assertEqual((first_package / "train.jsonl").read_bytes(), (second_package / "train.jsonl").read_bytes())
        samples = [json.loads(line) for line in (first_package / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["topic_id"] for row in samples], ["roads", "electricity"])
        self.assertTrue(all(row["production_prediction"]["topic_id"] == "water_supply" for row in samples))
        self.assertTrue(all(row["topic_id"] == row["operator_confirmed_decision"]["topic_id"] for row in samples))
        manifest = json.loads((first_package / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_feedback_ids"], ["feedback_1", "feedback_2"])
        self.assertEqual(manifest["train_sha256"], checksum(first_package / "train.jsonl"))
        with self.assertRaises(FileExistsError):
            self.build(self.root / "one", minimum=2)

    def test_runtime_origin_is_versioned_without_claiming_an_imported_dataset(self) -> None:
        write_jsonl(self.input, [feedback(1), runtime_feedback(2)])
        report = self.build(self.root / "out", minimum=2)
        self.assertEqual(report["status"], "COMPLETED")
        package = self.root / "out/candidate_v1"
        manifest, samples = load_verified_candidate(package, self.frozen)
        self.assertEqual(manifest.source_contract_version, "learning-feedback-export.mixed.v1")
        self.assertEqual(manifest.source_dataset_versions, ["source_demo_v1"])
        self.assertEqual(manifest.runtime_ticket_ids, ["ticket_2"])
        self.assertEqual(samples[1].source_origin_kind, "RUNTIME_API")
        self.assertIsNone(samples[1].source_dataset_version)
        frozen = json.loads((self.frozen / "frozen_evaluation.json").read_text(encoding="utf-8"))
        blocked = runtime_feedback(3)
        blocked["split_group"] = frozen["classifier_groups"][0]
        write_jsonl(self.input, [blocked])
        rejected = self.build(self.root / "rejected")
        self.assertEqual(rejected["rejected_counts"], {"FROZEN_ID_OR_GROUP": 1})

    def test_rejects_invalid_labels_pii_model_and_frozen_overlap(self) -> None:
        rows = [feedback(index) for index in range(1, 6)]
        rows[0]["operator_confirmed_decision"]["topic_id"] = "unknown"
        rows[1]["original_text"] = "ИИН 000000000000"
        rows[2]["production_model_version"] = "another_model"
        frozen = json.loads((self.frozen / "frozen_evaluation.json").read_text(encoding="utf-8"))
        rows[3]["split_group"] = frozen["classifier_groups"][0]
        frozen_text = json.loads((self.frozen / "classifier/test.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
        rows[4]["original_text"] = frozen_text
        write_jsonl(self.input, rows)
        result = self.build(self.root / "out", minimum=1)
        self.assertEqual(result["status"], "INSUFFICIENT_FEEDBACK")
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["input_record_count"], sum(result["rejected_counts"].values()))
        self.assertEqual(result["rejected_counts"], {
            "FROZEN_ID_OR_GROUP": 1,
            "FROZEN_TEXT": 1,
            "PII_DETECTED": 1,
            "UNKNOWN_CONFIRMED_TOPIC": 1,
            "WRONG_PRODUCTION_MODEL": 1,
        })
        self.assertFalse((self.root / "out/candidate_v1").exists())

    def test_rejects_canonically_equivalent_frozen_text(self) -> None:
        rows = [json.loads(line) for line in (self.frozen / "classifier/test.jsonl").read_text(encoding="utf-8").splitlines()]
        frozen_text = next(row["text"] for row in rows if "ё" in row["text"].casefold())
        decomposed = unicodedata.normalize("NFD", frozen_text)
        self.assertNotEqual(decomposed, frozen_text)
        row = feedback(1)
        row["original_text"] = decomposed
        write_jsonl(self.input, [row])
        result = self.build(self.root / "out")
        self.assertEqual(result["rejected_counts"], {"FROZEN_TEXT": 1})
        self.assertFalse((self.root / "out/candidate_v1").exists())

    def test_deduplicates_feedback_id_and_keeps_latest_ticket_decision(self) -> None:
        first = feedback(1)
        later = feedback(2, confirmed_topic="electricity")
        later["ticket_id"] = first["ticket_id"]
        later["feedback_created_at"] = "2026-09-28T12:00:00+05:00"
        write_jsonl(self.input, [first, first, later])
        result = self.build(self.root / "out")
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["rejected_counts"], {
            "DUPLICATE_FEEDBACK_ID": 1,
            "SUPERSEDED_TICKET_FEEDBACK": 1,
        })
        sample = json.loads((self.root / "out/candidate_v1/train.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(sample["feedback_id"], later["feedback_id"])
        self.assertEqual(sample["topic_id"], "electricity")

    def test_other_cycle_cannot_supersede_matching_ticket(self) -> None:
        valid = feedback(1)
        other_cycle = feedback(2)
        other_cycle["ticket_id"] = valid["ticket_id"]
        other_cycle["cycle_id"] = "cycle_2"
        other_cycle["feedback_created_at"] = "2026-09-28T12:00:00+05:00"
        write_jsonl(self.input, [valid, other_cycle])
        result = self.build(self.root / "out")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["rejected_counts"], {"WRONG_CYCLE": 1})

    def test_conflicting_id_and_missing_decision_are_counted(self) -> None:
        first = feedback(1)
        changed = feedback(1, confirmed_topic="electricity")
        missing = feedback(2)
        missing["operator_confirmed_decision"] = {}
        self.input.write_text(
            json.dumps(first, ensure_ascii=False) + "\n" +
            json.dumps(changed, ensure_ascii=False) + "\n" +
            json.dumps(missing, ensure_ascii=False) + "\n" +
            "{bad json\n",
            encoding="utf-8",
        )
        with self.input.open("ab") as stream:
            stream.write(b"\xff\n")
        result = self.build(self.root / "out")
        self.assertEqual(result["status"], "INSUFFICIENT_FEEDBACK")
        self.assertEqual(result["rejected_counts"], {
            "CONFLICTING_FEEDBACK_ID": 2,
            "INVALID_JSON": 1,
            "INVALID_SCHEMA": 1,
            "INVALID_UTF8": 1,
        })

    def test_tampered_frozen_package_fails_closed(self) -> None:
        write_jsonl(self.input, [feedback(1)])
        (self.frozen / "classifier/test.jsonl").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.build(self.root / "out")
        self.assertFalse((self.root / "out/candidate_v1").exists())

    def test_training_input_verifier_checks_checksum_lineage_and_frozen_text(self) -> None:
        write_jsonl(self.input, [feedback(1), feedback(2)])
        self.build(self.root / "out", minimum=2)
        package = self.root / "out/candidate_v1"
        manifest, samples = load_verified_candidate(package, self.frozen)
        self.assertEqual(manifest.record_count, 2)
        self.assertEqual([sample.topic_id for sample in samples], ["roads", "roads"])

        train_path = package / "train.jsonl"
        original = train_path.read_bytes()
        train_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_verified_candidate(package, self.frozen)
        train_path.write_bytes(original)

        rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
        frozen_rows = [json.loads(line) for line in (self.frozen / "classifier/test.jsonl").read_text(encoding="utf-8").splitlines()]
        frozen_text = next(row["text"] for row in frozen_rows if "ё" in row["text"].casefold())
        rows[0]["text"] = unicodedata.normalize("NFD", frozen_text)
        write_jsonl(train_path, rows)
        manifest_path = package / "manifest.json"
        changed = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed["train_sha256"] = checksum(train_path)
        changed.pop("content_sha256")
        canonical = json.dumps(changed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        changed["content_sha256"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        manifest_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid or frozen"):
            load_verified_candidate(package, self.frozen)

    def test_mismatched_decision_action_is_counted(self) -> None:
        row = feedback(1)
        row["accepted_or_corrected"] = "ACCEPTED"
        write_jsonl(self.input, [row])
        report = self.build(self.root / "out")
        self.assertEqual(report["status"], "INSUFFICIENT_FEEDBACK")
        self.assertEqual(report["rejected_counts"], {"DECISION_ACTION_MISMATCH": 1})


if __name__ == "__main__":
    unittest.main()
