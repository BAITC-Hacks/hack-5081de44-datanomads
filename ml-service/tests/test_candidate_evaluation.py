from __future__ import annotations

from typing import Any, Callable

from app.schemas import (
    CandidateEvaluationRequest,
    CandidateOfflineSample,
    CandidateShadowSample,
)
from app.services import EvaluationService, make_services


def _request(
    offline_count: int = 30,
    shadow_count: int = 20,
    **overrides: Any,
) -> CandidateEvaluationRequest:
    offline_samples = [
        CandidateOfflineSample(
            ticket_id=f"ticket-{index}",
            text=f"label:{'water' if index % 2 == 0 else 'lighting'}",
            label="water" if index % 2 == 0 else "lighting",
            language="RU",
        )
        for index in range(offline_count)
    ]
    shadow_samples = [
        CandidateShadowSample(
            production_topic_id="water" if index % 2 == 0 else "lighting",
            candidate_topic_id="water" if index % 2 == 0 else "lighting",
            confirmed_topic_id="water" if index % 2 == 0 else "lighting",
            candidate_inference_status="COMPLETED",
        )
        for index in range(shadow_count)
    ]
    values = {
        "cycle_id": "cycle-16",
        "candidate_model_version": "candidate-16",
        "production_model_version": "production-16",
        "production_baseline_available": True,
        "candidate_dataset_version": "candidate-dataset-16",
        "frozen_evaluation_dataset_version": "evaluation-16",
        "promotion_policy_version": "policy-v1",
        "offline_samples": offline_samples,
        "shadow_samples": shadow_samples,
        "shadow_inference_failures": 0,
        "synthetic": False,
        "blind_ab_enabled": False,
    }
    values.update(overrides)
    return CandidateEvaluationRequest.model_validate(values)


def _evaluator(
    monkeypatch: Any,
    candidate_prediction: Callable[[str], str] | None = None,
) -> EvaluationService:
    *_, evaluator = make_services()

    def classify_version(text: str, model_version: str, artifact_checksum: str | None) -> str:
        label = text.removeprefix("label:")
        if model_version == "candidate-16" and candidate_prediction is not None:
            return candidate_prediction(label)
        return label

    monkeypatch.setattr(evaluator, "_classify_version", classify_version)
    return evaluator


def _candidate_with_balanced_errors(errors_per_class: int) -> Callable[[str], str]:
    seen = {"water": 0, "lighting": 0}

    def predict(label: str) -> str:
        seen[label] += 1
        if seen[label] <= errors_per_class:
            return "lighting" if label == "water" else "water"
        return label

    return predict


def _shadow_samples_with_candidate_errors(
    error_count: int,
) -> list[CandidateShadowSample]:
    samples = _request().shadow_samples
    return [
        sample.model_copy(update={"candidate_topic_id": "other"})
        if index < error_count
        else sample
        for index, sample in enumerate(samples)
    ]


def test_candidate_evaluation_passes_frozen_policy_gates_for_human_review(monkeypatch) -> None:
    result = _evaluator(monkeypatch).evaluate_candidate(_request())

    assert result["status"] == "COMPLETED"
    assert result["decision"] == "PENDING_HUMAN_DECISION"
    assert result["policy_version"] == "policy-v1"
    assert result["promotion_policy"]["thresholds"] == {
        "minimum_offline_samples": 30,
        "minimum_shadow_samples": 20,
        "maximum_macro_f1_regression": 0.02,
        "maximum_class_f1_regression": 0.05,
        "maximum_shadow_correction_rate_delta": 0.05,
        "maximum_shadow_inference_failures": 0,
    }
    assert result["offline_evaluation"]["sample_count"] == 30
    assert result["baseline_evaluation"]["sample_count"] == 30
    assert result["shadow_evaluation"]["sample_count"] == 20
    assert all(gate["status"] == "PASSED" for gate in result["gates"])


def test_candidate_evaluation_fails_on_class_regression_over_policy_threshold(monkeypatch) -> None:
    evaluator = _evaluator(
        monkeypatch,
        candidate_prediction=lambda label: "lighting" if label == "water" else label,
    )

    result = evaluator.evaluate_candidate(_request())

    assert result["decision"] == "FAIL"
    assert "water" in result["offline_evaluation"]["critical_regressions"]
    class_gate = next(gate for gate in result["gates"] if gate["key"] == "critical_class_regressions")
    assert class_gate["status"] == "FAILED"


def test_macro_f1_regression_at_policy_limit_passes(monkeypatch) -> None:
    result = _evaluator(
        monkeypatch,
        candidate_prediction=_candidate_with_balanced_errors(errors_per_class=2),
    ).evaluate_candidate(_request(offline_count=200))

    gates = {gate["key"]: gate for gate in result["gates"]}
    assert gates["macro_f1_non_inferiority"]["observed"] == -0.02
    assert gates["macro_f1_non_inferiority"]["status"] == "PASSED"
    assert gates["critical_class_regressions"]["status"] == "PASSED"
    assert result["decision"] == "PENDING_HUMAN_DECISION"


def test_macro_f1_regression_beyond_policy_limit_fails_without_class_gate_failure(
    monkeypatch,
) -> None:
    result = _evaluator(
        monkeypatch,
        candidate_prediction=_candidate_with_balanced_errors(errors_per_class=3),
    ).evaluate_candidate(_request(offline_count=200))

    gates = {gate["key"]: gate for gate in result["gates"]}
    assert gates["macro_f1_non_inferiority"]["observed"] == -0.03
    assert gates["macro_f1_non_inferiority"]["status"] == "FAILED"
    assert gates["critical_class_regressions"]["status"] == "PASSED"
    assert result["decision"] == "FAIL"


def test_shadow_correction_delta_at_policy_limit_passes(monkeypatch) -> None:
    result = _evaluator(monkeypatch).evaluate_candidate(
        _request(shadow_samples=_shadow_samples_with_candidate_errors(error_count=1))
    )

    correction_gate = next(
        gate for gate in result["gates"] if gate["key"] == "shadow_correction_rate_delta"
    )
    assert result["shadow_evaluation"]["correction_rate_delta"] == 0.05
    assert correction_gate["status"] == "PASSED"
    assert result["decision"] == "PENDING_HUMAN_DECISION"


def test_shadow_correction_delta_beyond_policy_limit_fails(monkeypatch) -> None:
    result = _evaluator(monkeypatch).evaluate_candidate(
        _request(shadow_samples=_shadow_samples_with_candidate_errors(error_count=2))
    )

    correction_gate = next(
        gate for gate in result["gates"] if gate["key"] == "shadow_correction_rate_delta"
    )
    assert result["shadow_evaluation"]["correction_rate_delta"] == 0.1
    assert correction_gate["status"] == "FAILED"
    assert result["shadow_evaluation"]["critical_regressions"] == [
        "SHADOW_CORRECTION_RATE"
    ]
    assert result["decision"] == "FAIL"


def test_candidate_evaluation_requires_minimum_offline_and_shadow_samples(monkeypatch) -> None:
    result = _evaluator(monkeypatch).evaluate_candidate(
        _request(offline_count=29, shadow_count=19)
    )

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    gate_statuses = {gate["key"]: gate["status"] for gate in result["gates"]}
    assert gate_statuses["offline_sample_count"] == "INSUFFICIENT_EVIDENCE"
    assert gate_statuses["shadow_sample_count"] == "INSUFFICIENT_EVIDENCE"


def test_synthetic_baseline_cannot_produce_promotable_evidence(monkeypatch) -> None:
    result = _evaluator(monkeypatch).evaluate_candidate(
        _request(production_baseline_is_synthetic=True)
    )

    assert result["synthetic"] is True
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["offline_evaluation"]["status"] == "INSUFFICIENT_DATA"
    assert result["offline_evaluation"]["synthetic"] is True
