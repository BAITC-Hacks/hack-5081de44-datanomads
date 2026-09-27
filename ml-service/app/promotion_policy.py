"""Immutable, versioned gates for candidate promotion review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    version: str
    minimum_offline_samples: int
    minimum_shadow_samples: int
    maximum_macro_f1_regression: float
    maximum_class_f1_regression: float
    maximum_shadow_correction_rate_delta: float

    def thresholds(self) -> dict[str, int | float]:
        return {
            "minimum_offline_samples": self.minimum_offline_samples,
            "minimum_shadow_samples": self.minimum_shadow_samples,
            "maximum_macro_f1_regression": self.maximum_macro_f1_regression,
            "maximum_class_f1_regression": self.maximum_class_f1_regression,
            "maximum_shadow_correction_rate_delta": self.maximum_shadow_correction_rate_delta,
            "maximum_shadow_inference_failures": 0,
        }


_POLICIES = {
    # Versions are immutable: a changed threshold set must use a new version.
    "policy-v1": PromotionPolicy(
        version="policy-v1",
        minimum_offline_samples=30,
        minimum_shadow_samples=20,
        maximum_macro_f1_regression=0.02,
        maximum_class_f1_regression=0.05,
        maximum_shadow_correction_rate_delta=0.05,
    ),
}


def promotion_policy(version: str) -> PromotionPolicy | None:
    return _POLICIES.get(version)


def gate(
    key: str,
    status: str,
    *,
    observed: int | float | None = None,
    threshold: int | float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"key": key, "status": status}
    if observed is not None:
        result["observed"] = observed
    if threshold is not None:
        result["threshold"] = threshold
    if reason is not None:
        result["reason"] = reason
    return result
