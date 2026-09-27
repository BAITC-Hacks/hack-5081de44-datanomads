from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast

from app.trained_classifier import TrainedClassifierService
from train_classifier import LABELS
from training.classifier_pair_eval import compare_classifiers
from training.dataset_builder import checksum
from training.feedback_dataset import build_candidate
from training.feedback_trainer import CandidateTrainingError, train_feedback_candidate
from test_dataset_builder import build_fixture_package, fixture_inputs
from test_feedback_dataset import feedback, write_jsonl


class FeedbackTrainerTests(unittest.TestCase):
    def test_trains_immutable_candidate_and_loads_for_sanity_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3,
                                    prefix="feedback-train", topic_count=16)
            build_fixture_package(inputs, root / "frozen", "reviewed_v1", "eval_v1", 109)
            frozen = root / "frozen/reviewed_v1"
            feedback_path = root / "feedback.jsonl"
            write_jsonl(feedback_path, [feedback(1), feedback(2)])
            build_candidate(feedback_path, frozen, root / "datasets", cycle_id="cycle_1",
                            production_model_version="classifier_production_v1",
                            dataset_version="candidate_v1", min_feedback_count=2)
            package = root / "datasets/candidate_v1"

            production = root / "production"
            production.mkdir()
            raw_tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "<s>": 2, "</s>": 3}, unk_token="[UNK]"))
            raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            raw_tokenizer.post_processor = processors.TemplateProcessing(
                single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)])
            tokenizer = XLMRobertaTokenizerFast(tokenizer_object=raw_tokenizer, unk_token="[UNK]",
                                                pad_token="[PAD]", cls_token="<s>", sep_token="</s>")
            tokenizer.save_pretrained(production)
            config = XLMRobertaConfig(vocab_size=4, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      max_position_embeddings=64, num_labels=len(LABELS),
                                      id2label={index: label for index, label in enumerate(LABELS)},
                                      label2id={label: index for index, label in enumerate(LABELS)})
            XLMRobertaForSequenceClassification(config).save_pretrained(production, safe_serialization=True)
            (production / "manifest.json").write_text(json.dumps({
                "model_version": "classifier_production_v1", "model_family": "xlm-roberta-sequence-classification",
                "base_model": "local-tiny", "dataset_version": "reviewed_v1",
                "frozen_evaluation_version": "eval_v1",
                "created_at": "2026-09-27T00:00:00Z", "metrics": {"status": "reviewed_synthetic_holdout_only"},
                "languages": ["RU", "KZ", "MIXED"], "labels": list(LABELS),
                "training_config": {"max_length": 32, "temperature": 1.0, "input_length_strategy": "head-tail-32"},
                "artifact_checksum": checksum(production / "model.safetensors"),
            }), encoding="utf-8")
            output = root / "candidate"
            result = train_feedback_candidate(package, frozen, production, output,
                                              candidate_model_version="classifier_candidate_v1",
                                              epochs=1, batch_size=2, learning_rate=0.0001)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertTrue((output / "model.safetensors").is_file())
            self.assertNotEqual(result["artifact_checksum"], checksum(production / "model.safetensors"))
            candidate = TrainedClassifierService(output)
            self.assertEqual(candidate.classify("Проверка модели.").needs_review, True)
            self.assertFalse(candidate.confidence_policy.confident_enabled)
            self.assertEqual(candidate.metadata.dataset_version, "candidate_v1")
            self.assertEqual(candidate.input_length_strategy, "head-tail-32")
            self.assertEqual(candidate.metadata.model_extra["base_model_artifact_checksum"],
                             checksum(production / "model.safetensors"))
            production_tokens = AutoTokenizer.from_pretrained(production, local_files_only=True)("Проверка модели.")
            candidate_tokens = AutoTokenizer.from_pretrained(output, local_files_only=True)("Проверка модели.")
            self.assertEqual(dict(production_tokens), dict(candidate_tokens))
            policy_path = root / "critical-policy.json"
            policy_path.write_text(json.dumps({
                "policy_version": "classifier-critical-regression.v1",
                "critical_topics": [LABELS[0]], "max_f1_drop": 0.1,
                "min_topic_support": 1, "min_total_samples": 16,
            }), encoding="utf-8")
            comparison = compare_classifiers(frozen, production, output, policy_path)
            self.assertEqual(comparison["sample_count"], 48)
            self.assertEqual(comparison["candidate"]["model_version"], "classifier_candidate_v1")
            self.assertEqual(comparison["production"]["model_version"], "classifier_production_v1")
            with self.assertRaises(FileExistsError):
                train_feedback_candidate(package, frozen, production, output,
                                         candidate_model_version="classifier_candidate_v1")
            with (package / "train.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(CandidateTrainingError, "INVALID_CANDIDATE_DATASET"):
                train_feedback_candidate(package, frozen, production, root / "another-candidate",
                                         candidate_model_version="classifier_candidate_v2")
            self.assertFalse((root / "another-candidate").exists())
            command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/train_feedback_candidate.py"),
                       "--dataset", str(package), "--frozen-from", str(frozen),
                       "--production-model", str(production), "--candidate-model-version", "classifier_candidate_v2",
                       "--output", str(root / "another-candidate")]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stderr)["error_code"], "INVALID_CANDIDATE_DATASET")
            self.assertNotIn("На дороге", completed.stderr)


if __name__ == "__main__":
    unittest.main()
