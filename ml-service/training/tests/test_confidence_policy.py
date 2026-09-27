from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from app.confidence import ConfidencePolicy, POLICY_VERSION
from app.schemas import ModelMetadata
from app.trained_classifier import TrainedClassifierService
from train_classifier import confidence_state_report, select_confidence_policy


def metadata(*, enabled: bool, status: str) -> ModelMetadata:
    return ModelMetadata(
        model_version="classifier-test",
        model_family="xlm-roberta-sequence-classification",
        base_model="local-test",
        dataset_version="dataset-test",
        created_at="2026-09-27T00:00:00Z",
        metrics={"status": status},
        artifact_checksum="sha256:" + "0" * 64,
        confidence_policy_version=POLICY_VERSION,
        confidence_thresholds={"low_confidence_below": 0.55, "confident_at_or_above": 0.8},
        confident_enabled=enabled,
    )


class ConfidencePolicyTests(unittest.TestCase):
    def test_runtime_uncertain_returns_two_alternatives(self) -> None:
        service = TrainedClassifierService.__new__(TrainedClassifierService)
        service.metadata = metadata(enabled=False, status="synthetic_holdout_only")
        service.metadata.labels = ["roads", "water_supply", "electricity"]
        service.model_version = "classifier-test"
        service.max_length = 16
        service.temperature = 1.0
        service.device = torch.device("cpu")
        service.confidence_policy = ConfidencePolicy.from_metadata(service.metadata)
        service.tokenizer = lambda text, **kwargs: {"input_ids": torch.tensor([[1, 2]])}
        service.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[2.0, 1.0, 0.0]]))
        result = service.classify("На дороге яма", language="RU", top_k=3)
        self.assertEqual(result.confidence_state, "UNCERTAIN")
        self.assertEqual(len(result.alternatives), 2)
        self.assertTrue(result.needs_review)

    def test_validation_selects_candidate_but_synthetic_stays_in_review(self) -> None:
        logits = torch.tensor([[3.0, 0.0]] * 20 + [[0.0, 3.0]] * 20)
        labels = torch.tensor([0] * 20 + [1] * 20)
        rows = [{"language": "RU"}] * 20 + [{"language": "KZ"}] * 20
        policy, evidence = select_confidence_policy(logits, labels, rows, 1.0)
        self.assertEqual(evidence["candidate"]["threshold"], 0.55)
        self.assertFalse(policy.confident_enabled)
        report = confidence_state_report(logits, rows, 1.0, policy)
        self.assertEqual(report["state_counts"], {"CONFIDENT": 0, "UNCERTAIN": 40, "LOW_CONFIDENCE": 0})
        self.assertEqual(report["needs_review_share"], 1.0)

    def test_bad_validation_cannot_select_confident_threshold(self) -> None:
        logits = torch.tensor([[3.0, 0.0]] * 20 + [[0.0, 3.0]] * 20)
        labels = torch.tensor([1] * 20 + [0] * 20)
        rows = [{"language": "RU"}] * 20 + [{"language": "KZ"}] * 20
        policy, evidence = select_confidence_policy(logits, labels, rows, 1.0)
        self.assertIsNone(evidence["candidate"])
        self.assertEqual(policy.confident_at_or_above, 1.0)

    def test_runtime_rejects_synthetic_confident_and_invalid_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved real holdout"):
            ConfidencePolicy.from_metadata(metadata(enabled=True, status="synthetic_holdout_only"))
        policy = ConfidencePolicy.from_metadata(metadata(enabled=False, status="synthetic_holdout_only"))
        self.assertEqual(policy.state(0.4), "LOW_CONFIDENCE")
        self.assertEqual(policy.state(0.9), "UNCERTAIN")
        approved = ConfidencePolicy.from_metadata(metadata(enabled=True, status="approved_real_holdout"))
        self.assertEqual(approved.state(0.9), "CONFIDENT")
        with self.assertRaisesRegex(ValueError, "invalid classifier confidence thresholds"):
            ConfidencePolicy(low_confidence_below=0.9, confident_at_or_above=0.8, confident_enabled=False)

    def test_legacy_artifact_keeps_conservative_policy(self) -> None:
        old = metadata(enabled=False, status="synthetic_holdout_only")
        old.model_extra.clear()
        policy = ConfidencePolicy.from_metadata(old)
        self.assertEqual(policy.state(0.4), "LOW_CONFIDENCE")
        self.assertEqual(policy.state(0.9), "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
