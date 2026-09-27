from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

from training.dataset_builder import build_package, checksum
from training.embedder_candidate import train_embedder_candidate, verify_embedder_candidate
from training.retrieval_baselines import evaluate_retrieval_baselines
from test_dataset_builder import fixture_inputs


class EmbedderCandidateTests(unittest.TestCase):
    def test_trains_local_candidate_and_rejects_tampered_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="embedder")
            dataset = build_package(*inputs[:4], root / "packages", "dataset_v1", "eval_v1", 109)
            package = root / "packages/dataset_v1"

            base = root / "local_base"
            base.mkdir()
            raw = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "<s>": 2, "</s>": 3,
                                              "query": 4, "passage": 5, "Обращение": 6, "Проверка": 7},
                                             unk_token="[UNK]"))
            raw.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
                                    bos_token="<s>", eos_token="</s>").save_pretrained(base)
            config = XLMRobertaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      max_position_embeddings=128)
            XLMRobertaModel(config).save_pretrained(base, safe_serialization=True)
            baseline = evaluate_retrieval_baselines(package, base, batch_size=2)
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")

            output = root / "candidate"
            result = train_embedder_candidate(package, base, baseline_path, output,
                                              model_version="embedder_candidate_v1",
                                              base_model_id="tiny_local_fixture", epochs=1,
                                              batch_size=2, learning_rate=0.0001)
            manifest = verify_embedder_candidate(output)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(manifest.embedding_dimension, 16)
            self.assertEqual(manifest.dataset_content_sha256, dataset.content_sha256)
            self.assertEqual(manifest.artifact_checksum, checksum(output / "model.safetensors"))
            self.assertEqual(manifest.status, "CANDIDATE")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["top_3_manual_review_status"], "PENDING")
            self.assertEqual(metrics["test"]["candidate_count"], 3)
            self.assertNotIn("Обращение", json.dumps(metrics, ensure_ascii=False))
            self.assertNotIn("query_id", json.dumps(metrics, ensure_ascii=False))
            command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/train_embedder_candidate.py"),
                       "verify", "--artifact", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "VERIFIED")
            with self.assertRaises(FileExistsError):
                train_embedder_candidate(package, base, baseline_path, output,
                                         model_version="embedder_candidate_v1",
                                         base_model_id="tiny_local_fixture", epochs=1)

            wrong_baseline = root / "wrong_baseline.json"
            baseline["e5_artifact_sha256"] = "sha256:" + "0" * 64
            wrong_baseline.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline report"):
                train_embedder_candidate(package, base, wrong_baseline, root / "wrong_candidate",
                                         model_version="embedder_candidate_v2",
                                         base_model_id="tiny_local_fixture", epochs=1)
            self.assertFalse((root / "wrong_candidate").exists())

            with (package / "retrieval/train_pairs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                train_embedder_candidate(package, base, baseline_path, root / "tampered_dataset_candidate",
                                         model_version="embedder_candidate_v3",
                                         base_model_id="tiny_local_fixture", epochs=1)
            self.assertFalse((root / "tampered_dataset_candidate").exists())

            with (output / "metrics.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_embedder_candidate(output)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stderr)["error_code"], "EMBEDDER_CANDIDATE_FAILED")


if __name__ == "__main__":
    unittest.main()
