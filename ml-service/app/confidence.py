"""Versioned confidence states for local classifier artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .schemas import ModelMetadata


POLICY_VERSION = "classifier-uncertainty.v1"


@dataclass(frozen=True)
class ConfidencePolicy:
    low_confidence_below: float
    confident_at_or_above: float
    confident_enabled: bool

    def __post_init__(self) -> None:
        if (not math.isfinite(self.low_confidence_below) or
                not math.isfinite(self.confident_at_or_above) or
                not 0 <= self.low_confidence_below <= self.confident_at_or_above <= 1):
            raise ValueError("invalid classifier confidence thresholds")

    @classmethod
    def from_metadata(cls, metadata: ModelMetadata) -> ConfidencePolicy:
        extra = metadata.model_extra or {}
        thresholds = extra.get("confidence_thresholds")
        if thresholds is None:
            if "confidence_policy_version" in extra or "confident_enabled" in extra:
                raise ValueError("incomplete classifier confidence policy")
            return cls(low_confidence_below=0.55, confident_at_or_above=1.0, confident_enabled=False)
        if (extra.get("confidence_policy_version") != POLICY_VERSION or
                not isinstance(thresholds, dict) or
                set(thresholds) != {"low_confidence_below", "confident_at_or_above"} or
                type(extra.get("confident_enabled")) is not bool or
                any(type(value) not in (int, float) for value in thresholds.values())):
            raise ValueError("unsupported or incomplete classifier confidence policy")
        if extra["confident_enabled"] and metadata.metrics.get("status") != "approved_real_holdout":
            raise ValueError("CONFIDENT requires approved real holdout evidence")
        return cls(
            low_confidence_below=float(thresholds["low_confidence_below"]),
            confident_at_or_above=float(thresholds["confident_at_or_above"]),
            confident_enabled=extra["confident_enabled"],
        )

    def state(self, confidence: float) -> str:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid classifier confidence")
        if confidence < self.low_confidence_below:
            return "LOW_CONFIDENCE"
        if self.confident_enabled and confidence >= self.confident_at_or_above:
            return "CONFIDENT"
        return "UNCERTAIN"
