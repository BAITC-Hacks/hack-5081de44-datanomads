from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from training.dataset_builder import build_package
from training.retrieval_baselines import evaluate_retrieval_baselines, rank_metrics
from test_dataset_builder import fixture_inputs


class RetrievalBaselineTests(unittest.TestCase):
    def test_ranking_excludes_negative_only_queries_from_recall_and_mrr(self) -> None:
        rows = [
            {"query_id": "q1", "query_text": "one", "candidate_id": "a", "relation_label": "DUPLICATE"},
            {"query_id": "q1", "query_text": "one", "candidate_id": "b", "relation_label": "SIMILAR_BUT_NOT_DUPLICATE"},
            {"query_id": "q1", "query_text": "one", "candidate_id": "c", "relation_label": "UNRELATED"},
            {"query_id": "q2", "query_text": "two", "candidate_id": "d", "relation_label": "UNRELATED"},
            {"query_id": "q2", "query_text": "two", "candidate_id": "e", "relation_label": "UNRELATED"},
        ]
        result = rank_metrics(rows, [0.9, 0.8, 0.1, 0.4, 0.2])
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["positive_query_count"], 1)
        self.assertEqual(result["negative_only_query_count"], 1)
        self.assertEqual(result["metrics"]["recall_at_1"], 0.5)
        self.assertEqual(result["metrics"]["mrr"], 1)
        self.assertEqual(result["metrics"]["precision_at_1"], 0.5)
        self.assertEqual(result["metrics"]["ndcg_at_5"], 1)
        self.assertEqual(result["top_3_review_queue"][0]["review_status"], "PENDING")
        self.assertNotIn("relation_label", result["top_3_review_queue"][0]["top_3"][0])

    def test_lexical_baseline_uses_verified_frozen_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="retrieval")
            manifest = build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            result = evaluate_retrieval_baselines(root / "dataset_v1")
            self.assertEqual(result["frozen_evaluation_sha256"], manifest.frozen_evaluation_sha256)
            self.assertEqual(result["e5_model_status"], "NOT_RUN")
            self.assertEqual(set(result["models"]), {"tfidf_cosine"})
            for split in ("validation", "test"):
                metrics = result["models"]["tfidf_cosine"][split]
                self.assertEqual(metrics["query_count"], 1)
                self.assertEqual(metrics["positive_query_count"], 1)
                self.assertEqual(metrics["candidate_count"], 3)
                self.assertEqual(len(metrics["top_3_review_queue"][0]["top_3"]), 3)
            self.assertEqual(result, evaluate_retrieval_baselines(root / "dataset_v1"))
            command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/evaluate_retrieval_baselines.py"),
                       "--dataset", str(root / "dataset_v1"), "--output", str(root / "report.json")]
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads((root / "report.json").read_text(encoding="utf-8")), result)
            with self.assertRaisesRegex(ValueError, "local model directory"):
                evaluate_retrieval_baselines(root / "dataset_v1", root / "missing")


if __name__ == "__main__":
    unittest.main()
