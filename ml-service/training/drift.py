"""Compare PII-free monitoring snapshots without triggering retraining."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from training.dataset_builder import checksum
from training.contracts import _checksum
from training.feedback_dataset import ID_RE, TOPICS, _unique_object


CHAR_LIMITS = (128, 256, 512, 1024, 2048)
TOKEN_LIMITS = (128, 256, 384, 512, 1024)
CONFIDENCE_LIMITS = (0.2, 0.4, 0.6, 0.8)
LANGUAGES = {"RU", "KZ", "MIXED", "UNKNOWN"}
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}


class DriftSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    snapshot_version: Literal["pulse-drift-snapshot.v1"]
    source: Literal["postgres"]
    model_version: str
    window_start: AwareDatetime
    window_end: AwareDatetime
    ticket_count: int = Field(ge=0)
    character_lengths: list[int] = Field(min_length=6, max_length=6)
    token_lengths: list[int] | None
    tokenizer_file_checksums: dict[str, str] | None
    language_counts: dict[str, int]
    topic_counts: dict[str, int]
    topic_prediction_count: int = Field(ge=0)
    confidence_counts: list[int] = Field(min_length=5, max_length=5)
    prediction_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    corrected_count: int = Field(ge=0)
    similarity_feedback_count: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        if (not ID_RE.fullmatch(self.model_version) or self.window_start >= self.window_end or
                any(type(value) is not int or value < 0 for value in
                    [*self.character_lengths, *self.confidence_counts,
                     *self.language_counts.values(), *self.topic_counts.values()]) or
                set(self.language_counts) - LANGUAGES or
                set(self.topic_counts) - (TOPICS | {"unknown"}) or
                sum(self.character_lengths) != self.ticket_count or
                sum(self.language_counts.values()) != self.ticket_count or
                sum(self.topic_counts.values()) != self.topic_prediction_count or
                sum(self.confidence_counts) != self.prediction_count or
                max(self.prediction_count, self.topic_prediction_count) > self.ticket_count or
                self.corrected_count > self.decision_count):
            raise ValueError("drift snapshot counts, labels or window are invalid")
        if self.token_lengths is None:
            if self.tokenizer_file_checksums is not None:
                raise ValueError("tokenizer checksums without token lengths")
        elif (len(self.token_lengths) != 6 or
              any(type(value) is not int or value < 0 for value in self.token_lengths) or
              sum(self.token_lengths) != self.ticket_count or
              self.tokenizer_file_checksums is None or
              set(self.tokenizer_file_checksums) != TOKENIZER_FILES or
              any(not isinstance(value, str) or _checksum(value) != value
                  for value in self.tokenizer_file_checksums.values())):
            raise ValueError("drift token counts or tokenizer identity are invalid")
        return self


class DriftPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    policy_version: Literal["pulse-drift-policy.v1"]
    min_tickets: int = Field(ge=1)
    min_predictions: int = Field(ge=1)
    min_decisions: int = Field(ge=1)
    max_distribution_tv: float = Field(ge=0, le=1)
    max_correction_rate_increase: float = Field(ge=0, le=1)


def _distribution(baseline: list[int], recent: list[int], minimum: int,
                  maximum_tv: float) -> dict:
    baseline_count, recent_count = sum(baseline), sum(recent)
    if min(baseline_count, recent_count) < minimum:
        return {"status": "INSUFFICIENT_EVIDENCE", "baseline_count": baseline_count,
                "recent_count": recent_count, "total_variation": None}
    distance = round(0.5 * sum(abs(left / baseline_count - right / recent_count)
                               for left, right in zip(baseline, recent)), 6)
    return {"status": "DRIFT" if distance > maximum_tv else "STABLE",
            "baseline_count": baseline_count, "recent_count": recent_count,
            "total_variation": distance}


def _categories(baseline: dict[str, int], recent: dict[str, int], minimum: int,
                maximum_tv: float) -> dict:
    keys = sorted(set(baseline) | set(recent))
    return _distribution([baseline.get(key, 0) for key in keys],
                         [recent.get(key, 0) for key in keys], minimum, maximum_tv)


def compare_snapshots(baseline: DriftSnapshot, recent: DriftSnapshot,
                      policy: DriftPolicy) -> dict:
    if (baseline.model_version != recent.model_version or
            baseline.window_end > recent.window_start):
        raise ValueError("drift snapshots have different models or overlapping windows")
    signals = {
        "character_length": _distribution(baseline.character_lengths, recent.character_lengths,
                                          policy.min_tickets, policy.max_distribution_tv),
        "language": _categories(baseline.language_counts, recent.language_counts,
                                policy.min_tickets, policy.max_distribution_tv),
        "topic": _categories(baseline.topic_counts, recent.topic_counts,
                             policy.min_predictions, policy.max_distribution_tv),
        "confidence": _distribution(baseline.confidence_counts, recent.confidence_counts,
                                    policy.min_predictions, policy.max_distribution_tv),
    }
    signals["character_length"]["bin_upper_bounds"] = list(CHAR_LIMITS)
    signals["confidence"]["bin_upper_bounds"] = list(CONFIDENCE_LIMITS)
    for name, baseline_counts, recent_counts in (
            ("language", baseline.language_counts, recent.language_counts),
            ("topic", baseline.topic_counts, recent.topic_counts)):
        signals[name]["baseline_counts"] = baseline_counts
        signals[name]["recent_counts"] = recent_counts
    if (baseline.token_lengths is None or recent.token_lengths is None or
            baseline.tokenizer_file_checksums != recent.tokenizer_file_checksums):
        signals["model_token_length"] = {"status": "UNAVAILABLE_TOKENIZER_MISMATCH_OR_MISSING"}
    else:
        signals["model_token_length"] = _distribution(
            baseline.token_lengths, recent.token_lengths,
            policy.min_tickets, policy.max_distribution_tv,
        )
        signals["model_token_length"]["bin_upper_bounds"] = list(TOKEN_LIMITS)
    if min(baseline.decision_count, recent.decision_count) < policy.min_decisions:
        signals["correction_rate"] = {
            "status": "INSUFFICIENT_GROUND_TRUTH",
            "baseline_count": baseline.decision_count, "recent_count": recent.decision_count,
            "rate_increase": None,
        }
    else:
        baseline_rate = baseline.corrected_count / baseline.decision_count
        recent_rate = recent.corrected_count / recent.decision_count
        increase = round(recent_rate - baseline_rate, 6)
        signals["correction_rate"] = {
            "status": "DRIFT" if increase > policy.max_correction_rate_increase else "STABLE",
            "baseline_count": baseline.decision_count, "recent_count": recent.decision_count,
            "baseline_rate": round(baseline_rate, 6), "recent_rate": round(recent_rate, 6),
            "rate_increase": increase,
        }
    # The current schema records relation labels but not ranked candidates at
    # feedback time. Those labels alone cannot prove Recall@K or quality drift.
    signals["retrieval_quality"] = {
        "status": "UNAVAILABLE_NO_RANKED_RELEVANCE",
        "baseline_feedback_count": baseline.similarity_feedback_count,
        "recent_feedback_count": recent.similarity_feedback_count,
    }
    drifted = [name for name, signal in signals.items() if signal["status"] == "DRIFT"]
    checked = [name for name, signal in signals.items() if signal["status"] in {"DRIFT", "STABLE"}]
    partial = len(checked) != len(signals)
    return {
        "report_version": "pulse-drift-report.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": baseline.model_version,
        "policy": policy.model_dump(mode="json"),
        "baseline_window": [baseline.window_start.isoformat(), baseline.window_end.isoformat()],
        "recent_window": [recent.window_start.isoformat(), recent.window_end.isoformat()],
        "decision_cohort": "first_operator_decision_with_latest_prior_model_prediction_by_decision_time",
        "signals": signals,
        "drifted_signals": drifted,
        "status": "REVIEW_TRIGGER" if drifted else "PARTIAL_NO_DRIFT" if checked and partial
                  else "NO_DRIFT_DETECTED" if checked else "INSUFFICIENT_EVIDENCE",
        "automatic_retraining": False,
    }


def evaluate_drift(baseline_path: Path, recent_path: Path, policy_path: Path) -> dict:
    def read(path: Path, model):
        content = path.read_text(encoding="utf-8")
        json.loads(content, object_pairs_hook=_unique_object)
        return model.model_validate_json(content)

    baseline = read(baseline_path, DriftSnapshot)
    recent = read(recent_path, DriftSnapshot)
    policy = read(policy_path, DriftPolicy)
    report = compare_snapshots(baseline, recent, policy)
    report["baseline_sha256"] = checksum(baseline_path)
    report["recent_sha256"] = checksum(recent_path)
    report["policy_sha256"] = checksum(policy_path)
    return report
