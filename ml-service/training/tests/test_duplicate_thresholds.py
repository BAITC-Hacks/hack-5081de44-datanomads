from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.dataset_builder import build_package
from training.duplicate_thresholds import calibrate_duplicate_threshold, evaluate_duplicate_thresholds
from test_dataset_builder import fixture_inputs


class DuplicateThresholdTests(unittest.TestCase):
    def test_precision_policy_keeps_repeat_and_similar_cases_negative(self) -> None:
        rows = [
            {"pair_id": "d1", "relation_label": "DUPLICATE"},
            {"pair_id": "d2", "relation_label": "DUPLICATE"},
            {"pair_id": "r1", "relation_label": "REPEAT"},
            {"pair_id": "s1", "relation_label": "SIMILAR_BUT_NOT_DUPLICATE"},
            {"pair_id": "u1", "relation_label": "UNRELATED"},
        ]
        policy = {"min_duplicate_count": 2, "min_nonduplicate_count": 2,
                  "min_predictions": 1, "min_precision": 0.9}
        result = calibrate_duplicate_threshold(rows, [0.92, 0.8, 0.89, 0.7, 0.2], policy)
        self.assertEqual(result["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(result["selected_threshold"], 0.92)
        self.assertEqual(result["curve"][1]["precision"], 0.5)
        self.assertEqual(result["curve"][1]["recall"], 0.5)
        self.assertEqual(calibrate_duplicate_threshold(rows, [0.92, 0.8, 0.89, 0.7, 0.2],
                         {**policy, "min_duplicate_count": 3})["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(calibrate_duplicate_threshold(rows, [0.92, 0.8, 0.89, 0.7, 0.2],
                          {**policy, "min_predictions": 3})["selected_threshold"])

    def test_report_uses_verified_package_and_keeps_test_out_of_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root, groups_per_topic=3, retrieval_groups=3, prefix="duplicate")
            manifest = build_package(*inputs[:4], root, "dataset_v1", "eval_v1", 109)
            package = root / "dataset_v1"
            model = root / "local_model"
            model.mkdir()
            (model / "manifest.json").write_text('{"model_version":"embedder_v1"}', encoding="utf-8")
            policy = root / "policy.json"
            policy.write_text(json.dumps({"policy_version": "duplicate-threshold.v1",
                                          "min_duplicate_count": 1, "min_nonduplicate_count": 1,
                                          "min_predictions": 1, "min_precision": 0.9}), encoding="utf-8")

            def synthetic_scores(rows, _model, _batch_size):
                scores = [0.9 if row["relation_label"] == "DUPLICATE" else 0.1 for row in rows]
                return scores, "sha256:" + "a" * 64

            with patch("training.duplicate_thresholds._e5_scores", side_effect=synthetic_scores):
                report = evaluate_duplicate_thresholds(package, model, policy, "embedder_v1")
                self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
                self.assertEqual(report["threshold_status"], "PROVISIONAL")
                self.assertEqual(report["frozen_evaluation_sha256"], manifest.frozen_evaluation_sha256)
                self.assertEqual(report["model_identity_status"], "VERSION_MATCHED")
                self.assertEqual(report["test"]["fp"], 0)
                with self.assertRaisesRegex(ValueError, "model version"):
                    evaluate_duplicate_thresholds(package, model, policy, "wrong_version")

            def test_false_positive_scores(rows, _model, _batch_size):
                scores = [0.9 if row["relation_label"] == "DUPLICATE" else
                          0.95 if row["split"] == "test" else 0.1 for row in rows]
                return scores, "sha256:" + "a" * 64

            with patch("training.duplicate_thresholds._e5_scores", side_effect=test_false_positive_scores):
                failed = evaluate_duplicate_thresholds(package, model, policy, "embedder_v1")
            self.assertEqual(failed["validation"]["selected_threshold"], 0.9)
            self.assertEqual(failed["status"], "TEST_PRECISION_BELOW_POLICY")
            self.assertEqual(failed["threshold_status"], "REJECTED")
            self.assertTrue(failed["test"]["false_positive_pairs"])
            self.assertNotIn("pair_id", failed["test"]["false_positive_pairs"][0])
            self.assertTrue(failed["test"]["false_positive_pairs"][0]["pair_sha256"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
