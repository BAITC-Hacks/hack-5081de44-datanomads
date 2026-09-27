"""Evaluate fresh paired classifier predictions against operator decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from training.dataset_builder import checksum
from training.feedback_dataset import ID_RE, TOPICS, OperatorDecision, PredictionEvidence, _unique_object


class ShadowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["classifier-shadow-input.v1"]
    feedback_id: str
    cycle_id: str
    ticket_id: str
    source_dataset_version: str
    is_synthetic: bool
    production_model_version: str
    candidate_model_version: str
    production_prediction: PredictionEvidence
    candidate_shadow_prediction: PredictionEvidence
    operator_confirmed_decision: OperatorDecision
    accepted_or_corrected: Literal["ACCEPTED", "CORRECTED"]
    feedback_created_at: AwareDatetime
    validation_status: Literal["VALID"]

    @model_validator(mode="after")
    def valid_decision(self):
        if (any(not ID_RE.fullmatch(value) for value in (
                self.feedback_id, self.cycle_id, self.ticket_id, self.source_dataset_version,
                self.production_model_version, self.candidate_model_version)) or
                self.operator_confirmed_decision.topic_id not in TOPICS or
                self.production_prediction.topic_id not in TOPICS | {"unknown"} or
                self.candidate_shadow_prediction.topic_id not in TOPICS | {"unknown"} or
                self.accepted_or_corrected != ("ACCEPTED" if self.operator_confirmed_decision.action == "confirm" else "CORRECTED")):
            raise ValueError("shadow row has invalid identity, topic or operator decision")
        return self


class ShadowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    policy_version: Literal["classifier-shadow-policy.v1"]
    promotion_policy_version: str = Field(min_length=1)
    window_start: AwareDatetime
    window_end: AwareDatetime
    min_samples: int = Field(ge=1)
    min_real_samples: int = Field(ge=1)
    critical_topics: list[str] = Field(min_length=1)
    min_topic_support: int = Field(ge=1)
    max_topic_agreement_drop: float = Field(ge=0, le=1)
    max_correction_rate_increase: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_window_and_topics(self):
        if (self.window_start >= self.window_end or
                len(self.critical_topics) != len(set(self.critical_topics)) or
                any(topic not in TOPICS for topic in self.critical_topics) or
                not ID_RE.fullmatch(self.promotion_policy_version)):
            raise ValueError("shadow policy window, critical topics or version is invalid")
        return self


def evaluate_shadow(rows_path: Path, policy_path: Path, *, cycle_id: str,
                    production_model_version: str, candidate_model_version: str) -> dict:
    if (not all(ID_RE.fullmatch(value) for value in
                (cycle_id, production_model_version, candidate_model_version)) or
            production_model_version == candidate_model_version):
        raise ValueError("shadow evaluation model or cycle version is invalid")
    policy = ShadowPolicy.model_validate_json(policy_path.read_text(encoding="utf-8"))
    rows = []
    seen_feedback = set()
    seen_tickets = set()
    with rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            json.loads(line, object_pairs_hook=_unique_object)
            row = ShadowRecord.model_validate_json(line)
            if (row.cycle_id != cycle_id or row.production_model_version != production_model_version or
                    row.candidate_model_version != candidate_model_version or
                    not policy.window_start <= row.feedback_created_at < policy.window_end or
                    row.feedback_id in seen_feedback or row.ticket_id in seen_tickets):
                raise ValueError("shadow rows differ in versions, window or identity")
            seen_feedback.add(row.feedback_id)
            seen_tickets.add(row.ticket_id)
            rows.append(row)
    per_topic: dict[str, list[ShadowRecord]] = defaultdict(list)
    for row in rows:
        per_topic[row.operator_confirmed_decision.topic_id].append(row)

    def agreement(items: list[ShadowRecord], model: str) -> float | None:
        if not items:
            return None
        return round(sum(getattr(row, model).topic_id == row.operator_confirmed_decision.topic_id
                         for row in items) / len(items), 6)

    production_agreement = agreement(rows, "production_prediction")
    candidate_agreement = agreement(rows, "candidate_shadow_prediction")
    real_rows = [row for row in rows if not row.is_synthetic]
    real_production_agreement = agreement(real_rows, "production_prediction")
    real_candidate_agreement = agreement(real_rows, "candidate_shadow_prediction")
    by_topic = {}
    regressions = []
    insufficient_topics = []
    for topic in sorted(per_topic):
        topic_rows = per_topic[topic]
        production = agreement(topic_rows, "production_prediction")
        candidate = agreement(topic_rows, "candidate_shadow_prediction")
        drop = round(production - candidate, 6)
        real_topic_rows = [row for row in topic_rows if not row.is_synthetic]
        real_production = agreement(real_topic_rows, "production_prediction")
        real_candidate = agreement(real_topic_rows, "candidate_shadow_prediction")
        by_topic[topic] = {"sample_count": len(topic_rows),
                           "real_sample_count": len(real_topic_rows),
                           "production_agreement": production,
                           "candidate_agreement": candidate, "candidate_agreement_drop": drop,
                           "real_production_agreement": real_production,
                           "real_candidate_agreement": real_candidate,
                           "real_candidate_agreement_drop": (
                               round(real_production - real_candidate, 6)
                               if real_production is not None and real_candidate is not None else None)}
    for topic in policy.critical_topics:
        topic_metrics = by_topic.get(topic)
        if topic_metrics is None or topic_metrics["real_sample_count"] < policy.min_topic_support:
            insufficient_topics.append(topic)
        elif topic_metrics["real_candidate_agreement_drop"] > policy.max_topic_agreement_drop:
            regressions.append(topic)
    origin_counts = dict(sorted(Counter("synthetic" if row.is_synthetic else "real" for row in rows).items()))
    sufficient = (len(rows) >= policy.min_samples and
                  origin_counts.get("real", 0) >= policy.min_real_samples and
                  not insufficient_topics)
    correction_rate_delta = (round(production_agreement - candidate_agreement, 6)
                             if production_agreement is not None and candidate_agreement is not None else None)
    real_correction_rate_delta = (round(real_production_agreement - real_candidate_agreement, 6)
                                  if real_production_agreement is not None and
                                  real_candidate_agreement is not None else None)
    global_regression = (real_correction_rate_delta is not None and
                         real_correction_rate_delta > policy.max_correction_rate_increase)
    status = "VALID" if sufficient else "INSUFFICIENT_EVIDENCE"
    decision = ("INSUFFICIENT_EVIDENCE" if not sufficient else
                "NO_GO_CRITICAL_REGRESSION" if regressions else
                "NO_GO_CORRECTION_RATE" if global_regression else "PENDING_HUMAN_REVIEW")
    ids_digest = hashlib.sha256(json.dumps(sorted(seen_feedback), ensure_ascii=False).encode("utf-8")).hexdigest()
    champion_reference = [
        {
            "feedback_id": row.feedback_id,
            "ticket_id": row.ticket_id,
            "source_dataset_version": row.source_dataset_version,
            "is_synthetic": row.is_synthetic,
            "production_prediction": row.production_prediction.model_dump(mode="json"),
            "operator_confirmed_decision": row.operator_confirmed_decision.model_dump(mode="json"),
            "feedback_created_at": row.feedback_created_at.astimezone(timezone.utc).isoformat(),
        }
        for row in sorted(rows, key=lambda item: item.feedback_id)
    ]
    reference_digest = hashlib.sha256(json.dumps(champion_reference, ensure_ascii=False,
                                               sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "report_version": "classifier-shadow-evaluation.v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id,
        "production_model_version": production_model_version,
        "candidate_model_version": candidate_model_version,
        "promotion_policy_version": policy.promotion_policy_version,
        "policy_sha256": checksum(policy_path),
        "policy": policy.model_dump(mode="json"),
        "input_sha256": checksum(rows_path),
        "window_start": policy.window_start.isoformat(),
        "window_end": policy.window_end.isoformat(),
        "sample_count": len(rows),
        "sample_ids_sha256": f"sha256:{ids_digest}",
        "champion_reference_sha256": f"sha256:{reference_digest}",
        "source_dataset_versions": sorted({row.source_dataset_version for row in rows}),
        "origin_counts": origin_counts,
        "gate_population": "real_only.v1",
        "production_agreement": production_agreement,
        "candidate_agreement": candidate_agreement,
        "production_correction_rate": round(1 - production_agreement, 6) if production_agreement is not None else None,
        "candidate_correction_rate": round(1 - candidate_agreement, 6) if candidate_agreement is not None else None,
        "correction_rate_delta": correction_rate_delta,
        "real_production_agreement": real_production_agreement,
        "real_candidate_agreement": real_candidate_agreement,
        "real_production_correction_rate": round(1 - real_production_agreement, 6) if real_production_agreement is not None else None,
        "real_candidate_correction_rate": round(1 - real_candidate_agreement, 6) if real_candidate_agreement is not None else None,
        "real_correction_rate_delta": real_correction_rate_delta,
        "global_regression": global_regression,
        "by_topic": by_topic,
        "insufficient_critical_topics": insufficient_topics,
        "critical_regressions": regressions,
        "blind_ab_enabled": False,
        "blind_ab_preference": None,
        "status": status,
        "decision": decision,
    }
