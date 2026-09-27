"""Deterministic local ML primitives used by the Pulse 109 demo.

The service has no network/model-download side effects at runtime. These
implementations are baselines that preserve the production API contract while
the real, versioned artifacts are prepared from the 109 dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from statsforecast.models import SeasonalNaive
from pydantic import ValidationError

from contracts import ContractValidationError, validate_document

from .constants import DEMO_IMPLEMENTATIONS, MODEL_VERSIONS, TOPICS, TOPIC_BY_ID, Topic
from .schemas import (
    Alternative,
    AnomalyPoint,
    AnomalyResponse,
    CandidateTrainingJob,
    CandidateTrainingResult,
    Classification,
    EvaluationRequest,
    EvaluationResponse,
    ForecastPoint,
    ForecastResponse,
    ModelManifestResponse,
    ModelMetadata,
    TimeSeriesPoint,
    TrainingRequest,
    TrainingResponse,
)
from .training.classifier import train_candidate_classifier


logger = logging.getLogger("pulse109.ml.registry")


_TOKEN_RE = re.compile(r"[\wа-яёәғқңөұүһі]+", flags=re.IGNORECASE | re.UNICODE)
_KZ_SPECIFIC = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_RU_SPECIFIC = set("ёэъЁЭЪ")
_TEST_RUNTIME_MODES = frozenset({"demo", "development", "test", "unit"})


def test_fake_trainer_requested() -> bool:
    return os.environ.get("PULSE_TEST_FAKE_TRAINER", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def test_fake_trainer_enabled() -> bool:
    runtime_mode = os.environ.get("PULSE_ENV", "demo").strip().lower()
    return test_fake_trainer_requested() and runtime_mode in _TEST_RUNTIME_MODES


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def detect_language(text: str, requested: str | None = None) -> str:
    """Return a stable RU/KZ/MIXED/UNKNOWN language state."""

    if requested:
        value = requested.strip().upper()
        aliases = {"RU": "RU", "RUS": "RU", "KZ": "KZ", "KK": "KZ", "KAZ": "KZ"}
        if value in aliases:
            return aliases[value]
        if value in {"MIXED", "UNKNOWN"}:
            return value

    cyrillic = sum(1 for char in text if "а" <= char.lower() <= "я" or char in _KZ_SPECIFIC)
    if not cyrillic:
        return "UNKNOWN"
    has_kz = any(char in _KZ_SPECIFIC for char in text)
    has_ru = any(char in _RU_SPECIFIC for char in text)
    if has_kz and has_ru:
        return "MIXED"
    return "KZ" if has_kz else "RU"


class ClassifierService:
    def __init__(
        self,
        topics: Iterable[Topic] = TOPICS,
        model_version: str = MODEL_VERSIONS["classifier"],
    ) -> None:
        self.topics = tuple(topics)
        self.model_version = model_version

    @staticmethod
    def _topic_name(topic: Topic, language: str) -> str:
        return topic.name_kz if language == "KZ" else topic.name_ru

    def classify(self, text: str, language: str | None = None, top_k: int = 3) -> Classification:
        normalized = " ".join(text.lower().split())
        detected = detect_language(text, language)
        scores: dict[str, float] = {topic.topic_id: 0.0 for topic in self.topics}
        matched = False
        for topic in self.topics:
            score = 0.0
            for keyword in topic.keywords:
                if keyword in normalized:
                    # Longer phrases are more specific than one-word matches.
                    score += 1.0 + min(len(keyword), 24) / 48.0
            if score:
                matched = True
                scores[topic.topic_id] = score

        if not matched:
            # Keep the fallback deterministic and transparent: ``other`` is the
            # prediction, while the digest supplies a repeatable low-confidence
            # tie-break for alternatives rather than using Python's random hash.
            winning = TOPIC_BY_ID["other"]
            scores[winning.topic_id] = 1.0
            confidence = 0.28
            state = "LOW_CONFIDENCE"
        else:
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            winner_id, winner_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else 0.0
            confidence = min(0.99, 0.56 + 0.22 * min(winner_score, 2.0) + 0.12 * (winner_score - second_score))
            if confidence >= 0.78:
                state = "CONFIDENT"
            elif confidence >= 0.58:
                state = "UNCERTAIN"
            else:
                state = "LOW_CONFIDENCE"

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        winner_id = ordered[0][0]
        winner = TOPIC_BY_ID[winner_id]
        max_score = max(scores.values()) or 1.0
        alternatives: list[Alternative] = []
        for topic_id, score in ordered[: max(1, min(top_k, len(ordered)))]:
            topic = TOPIC_BY_ID[topic_id]
            score_value = 0.0 if topic_id != winner_id and score == 0 else min(1.0, score / max_score)
            alternatives.append(
                Alternative(
                    topic_id=topic_id,
                    topic=self._topic_name(topic, detected),
                    score=round(score_value, 6),
                    confidence=round(confidence if topic_id == winner_id else score_value * confidence, 6),
                )
            )

        return Classification(
            language=detected,
            topic_id=winner.topic_id,
            topic=self._topic_name(winner, detected),
            label=winner.topic_id,
            confidence=round(confidence, 6),
            confidence_state=state,
            needs_review=state != "CONFIDENT",
            alternatives=alternatives,
            model_version=self.model_version,
        )


class EmbeddingService:
    """Small deterministic hashed embedding baseline.

    It is intentionally not presented as a fine-tuned E5 artifact.  Hashing
    keeps local development reproducible and allows Qdrant integration tests
    without downloading a multi-hundred-megabyte model.
    """

    def __init__(self, model_version: str = MODEL_VERSIONS["embedder"]) -> None:
        self.model_version = model_version

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.lower().strip()
        tokens = _TOKEN_RE.findall(normalized)
        # Character trigrams improve similarity for inflected RU/KZ words.
        for token in list(tokens):
            padded = f"^{token}$"
            tokens.extend(padded[index : index + 3] for index in range(max(0, len(padded) - 2)))
        return tokens

    def embed(self, text: str, dimension: int = 32, normalize: bool = True) -> list[float]:
        vector = [0.0] * dimension
        tokens = self._tokens(text)
        for position, token in enumerate(tokens):
            hashed = _digest(f"{position % 17}:{token}")
            bucket = int.from_bytes(hashed[:4], "big") % dimension
            sign = 1.0 if hashed[4] & 1 else -1.0
            weight = 1.0 / math.sqrt(1.0 + position * 0.02)
            vector[bucket] += sign * weight
        if normalize:
            norm = math.sqrt(sum(item * item for item in vector))
            if norm:
                vector = [item / norm for item in vector]
        return [round(item, 8) for item in vector]


def _safe_float(value: float) -> float:
    return 0.0 if not math.isfinite(value) else float(value)


class ForecastService:
    def __init__(self, model_version: str = MODEL_VERSIONS["forecast"]) -> None:
        self.model_version = model_version

    @staticmethod
    def _forecast_values(values: list[float], horizon: int, season_length: int) -> list[float]:
        if len(values) < season_length:
            return [round(max(0.0, _safe_float(values[-1])), 6)] * horizon
        predicted = SeasonalNaive(season_length=season_length).forecast(
            y=np.asarray(values, dtype=np.float64), h=horizon
        )["mean"]
        return [round(max(0.0, _safe_float(float(value))), 6) for value in predicted]

    @staticmethod
    def _backtest(values: list[float], season_length: int) -> dict[str, float | int | None]:
        if len(values) <= season_length:
            return {"mae": None, "rmse": None, "wape": None, "smape": None, "sample_count": 0, "window_count": 0}
        actual: list[float] = []
        predicted: list[float] = []
        window_count = 0
        for cutoff in range(season_length, len(values), season_length):
            window = values[cutoff : cutoff + season_length]
            actual.extend(window)
            predicted.extend(ForecastService._forecast_values(values[:cutoff], len(window), season_length))
            window_count += 1
        errors = [abs(a - p) for a, p in zip(actual, predicted)]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(errors))
        denominator = sum(abs(item) for item in actual)
        wape = sum(errors) / denominator if denominator else 0.0
        smape_terms = [
            2 * abs(a - p) / (abs(a) + abs(p))
            for a, p in zip(actual, predicted)
            if abs(a) + abs(p)
        ]
        smape = sum(smape_terms) / len(smape_terms) if smape_terms else 0.0
        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "wape": round(wape, 6),
            "smape": round(smape, 6),
            "sample_count": len(actual),
            "window_count": window_count,
        }

    @staticmethod
    def _next_timestamps(
        timestamps: list[datetime | str | None] | None, horizon: int
    ) -> list[datetime | str | None]:
        if not timestamps or timestamps[-1] is None:
            return [None] * horizon
        last = timestamps[-1]
        parsed: datetime | None = None
        if isinstance(last, datetime):
            parsed = last
        elif isinstance(last, str):
            try:
                parsed = datetime.fromisoformat(last.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        if parsed is None:
            return [None] * horizon
        return [parsed + timedelta(days=index + 1) for index in range(horizon)]

    def forecast(
        self,
        values: list[float],
        horizon: int,
        season_length: int,
        timestamps: list[datetime | str | None] | None = None,
    ) -> ForecastResponse:
        forecast_values = self._forecast_values(values, horizon, season_length)
        next_timestamps = self._next_timestamps(timestamps, horizon)
        points = [
            ForecastPoint(index=index, value=value, timestamp=next_timestamps[index])
            for index, value in enumerate(forecast_values)
        ]
        max_value = max(forecast_values) if forecast_values else 0.0
        expected_peaks = [index for index, value in enumerate(forecast_values) if value == max_value and value > 0]
        insufficient = len(values) < season_length
        return ForecastResponse(
            model_version=self.model_version,
            model="seasonal_naive",
            status="INSUFFICIENT_HISTORY" if insufficient else "OK",
            insufficient_history=insufficient,
            horizon=horizon,
            season_length=season_length,
            forecast=forecast_values,
            points=points,
            expected_peaks=expected_peaks,
            backtest=self._backtest(values, season_length),
        )


class AnomalyService:
    def __init__(self, model_version: str = MODEL_VERSIONS["anomaly"]) -> None:
        self.model_version = model_version

    def detect(
        self,
        values: list[float],
        window: int,
        threshold: float,
        timestamps: list[datetime | str | None] | None = None,
    ) -> AnomalyResponse:
        points: list[AnomalyPoint] = []
        for index, value in enumerate(values):
            history = values[max(0, index - window) : index]
            if len(history) < 2:
                expected = history[-1] if history else value
                score = 0.0
            else:
                expected = statistics.median(history)
                deviations = [abs(item - expected) for item in history]
                mad = statistics.median(deviations)
                if mad > 0:
                    scale = 1.4826 * mad
                    score = abs(value - expected) / scale
                else:
                    mean = statistics.fmean(history)
                    std = statistics.pstdev(history)
                    expected = mean
                    score = abs(value - expected) / std if std > 0 else (10.0 if value != expected else 0.0)
            timestamp = timestamps[index] if timestamps and index < len(timestamps) else None
            points.append(
                AnomalyPoint(
                    index=index,
                    value=round(_safe_float(value), 6),
                    expected=round(_safe_float(expected), 6),
                    score=round(_safe_float(score), 6),
                    is_anomaly=score >= threshold,
                    timestamp=timestamp,
                )
            )
        latest = points[-1]
        return AnomalyResponse(
            model_version=self.model_version,
            threshold=threshold,
            window=window,
            is_anomaly=latest.is_anomaly,
            latest=latest,
            anomalies=[point for point in points if point.is_anomaly],
            points=points,
        )


def _classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = sorted(set(actual) | set(predicted))
    if not actual:
        return {"accuracy": None, "macro_f1": None, "per_class_f1": {}, "confusion_matrix": []}
    correct = sum(a == p for a, p in zip(actual, predicted))
    per_class: dict[str, float] = {}
    matrix: list[list[int]] = []
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0
    for actual_label in labels:
        matrix.append([sum(a == actual_label and p == predicted_label for a, p in zip(actual, predicted)) for predicted_label in labels])
    return {
        "accuracy": round(correct / len(actual), 6),
        "macro_f1": round(sum(per_class.values()) / len(per_class), 6) if per_class else 0.0,
        "per_class_f1": per_class,
        "labels": labels,
        "confusion_matrix": matrix,
    }


class ModelRegistry:
    def __init__(self, path: Path | None = None, runtime_mode: str | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "artifacts" / "manifest.json"
        self.path = path or Path(os.environ.get("PULSE_MODEL_MANIFEST_PATH", default_path))
        self.runtime_mode = (runtime_mode or os.environ.get("PULSE_ENV", "demo")).strip().lower()
        self.manifest: ModelManifestResponse | None = None
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            validate_document(payload, "ModelManifest")
            manifest = ModelManifestResponse.model_validate(payload)
        except (OSError, ValueError, TypeError, ContractValidationError, ValidationError):
            self.load_error = "MODEL_MANIFEST_UNAVAILABLE_OR_INVALID"
            logger.error(
                "model_manifest_unavailable_or_invalid",
                extra={"error_code": self.load_error},
            )
            return

        self.manifest = manifest
        models = manifest.models
        if self.runtime_mode in {"demo", "development", "test"}:
            for model_type, implementation in DEMO_IMPLEMENTATIONS.items():
                model = models[model_type]
                if model.artifact_kind != "DETERMINISTIC_BASELINE":
                    self.load_error = "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED"
                    break
                if model.status != "DEMO_BASELINE" or model.implementation != implementation:
                    self.load_error = "MODEL_MANIFEST_RUNTIME_MISMATCH"
                    break
        elif self.runtime_mode in {"production", "prod"}:
            # The repository currently ships only deterministic runtime adapters.
            # A valid trained manifest must not make those baselines look trained.
            self.load_error = "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED"
        else:
            self.load_error = "INVALID_PULSE_ENV"

        if self.load_error is not None:
            logger.error(
                "model_runtime_not_ready",
                extra={"error_code": self.load_error},
            )

    @property
    def ready(self) -> bool:
        return self.manifest is not None and self.load_error is None

    def model_versions(self) -> dict[str, str]:
        if self.manifest is None:
            return {}
        return {key: model.model_version for key, model in self.manifest.models.items()}

    def get(self, model_type: str) -> ModelMetadata:
        if self.manifest is None or model_type not in self.manifest.models:
            raise KeyError(model_type)
        return self.manifest.models[model_type]


class TrainingService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.jobs: dict[str, TrainingResponse] = {}

    def train(self, request: TrainingRequest, model_type: str | None = None) -> TrainingResponse:
        selected = model_type or request.model_type
        sample_count = len(request.samples)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model_type": selected, "dataset": request.dataset_version, "samples": [sample.model_dump() for sample in request.samples]},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        job_id = f"train-{fingerprint[:16]}"
        candidate_version = f"{selected}-candidate-{fingerprint[:12]}"
        enough = sample_count >= request.min_samples and sample_count > 0
        if not enough:
            response = TrainingResponse(
                job_id=job_id,
                state="INSUFFICIENT_FEEDBACK",
                model_type=selected,
                dataset_version=request.dataset_version,
                sample_count=sample_count,
                metrics={"required_samples": request.min_samples, "reason": "insufficient_feedback"},
            )
        elif not test_fake_trainer_enabled():
            response = TrainingResponse(
                job_id=job_id,
                state="TRAINER_NOT_CONFIGURED",
                model_type=selected,
                dataset_version=request.dataset_version,
                sample_count=sample_count,
                metrics={"reason": "trained_model_pipeline_not_configured"},
            )
        else:
            labels = sorted({sample.get_label() for sample in request.samples if sample.get_label()})
            response = TrainingResponse(
                job_id=job_id,
                state="COMPLETED",
                model_type=selected,
                dataset_version=request.dataset_version,
                candidate_model_version=candidate_version,
                sample_count=sample_count,
                metrics={"sample_count": sample_count, "class_count": len(labels), "labels": labels, "offline_only": True},
                manifest={
                    "model_version": candidate_version,
                    "model_family": "deterministic-demo-candidate",
                    "base_model": request.base_model or "local-no-download",
                    "dataset_version": request.dataset_version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "training_config": request.config,
                    "artifact_checksum": hashlib.sha256(candidate_version.encode()).hexdigest(),
                },
            )
        self.jobs[job_id] = response
        return response

    def train_candidate(self, request: CandidateTrainingJob) -> CandidateTrainingResult:
        return train_candidate_classifier(request)

    def get(self, job_id: str) -> TrainingResponse | None:
        return self.jobs.get(job_id)


class EvaluationService:
    def __init__(self, registry: ModelRegistry, classifier: ClassifierService, forecast: ForecastService, anomaly: AnomalyService) -> None:
        self.registry = registry
        self.classifier = classifier
        self.forecast = forecast
        self.anomaly = anomaly
        self.evaluations: dict[str, EvaluationResponse] = {}

    def evaluate(self, request: EvaluationRequest, model_type: str | None = None) -> EvaluationResponse:
        selected = model_type or request.model_type
        payload = request.model_dump(mode="json")
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        evaluation_id = f"eval-{fingerprint[:16]}"
        metrics: dict[str, Any]
        sample_count: int
        status = "COMPLETED"
        if selected == "classifier":
            samples = [sample for sample in request.samples if sample.get_label()]
            texts = request.texts or [sample.text for sample in samples]
            labels = request.labels or [sample.get_label() for sample in samples]
            if not texts or len(texts) != len(labels):
                status = "INSUFFICIENT_DATA"
                sample_count = 0
                metrics = {"reason": "labelled_texts_required"}
            else:
                predicted = [self.classifier.classify(text).topic_id for text in texts]
                sample_count = len(texts)
                metrics = _classification_metrics([str(label) for label in labels], predicted)
        elif selected == "forecast":
            sample_count = len(request.values)
            if sample_count <= request.season_length:
                status = "INSUFFICIENT_DATA"
                metrics = {"reason": "history_shorter_than_season_length"}
            else:
                metrics = self.forecast._backtest(request.values, request.season_length)
        elif selected == "anomaly":
            sample_count = len(request.values)
            if sample_count < 2:
                status = "INSUFFICIENT_DATA"
                metrics = {"reason": "at_least_two_values_required"}
            else:
                result = self.anomaly.detect(request.values, request.season_length, 3.0)
                metrics = {"anomaly_count": len(result.anomalies), "rate": round(len(result.anomalies) / sample_count, 6)}
        else:
            sample_count = len(request.samples) or len(request.texts) or len(request.values)
            metrics = {"sample_count": sample_count, "offline_only": True}
        try:
            model_version = self.registry.get(selected).model_version
        except KeyError:
            model_version = MODEL_VERSIONS.get(selected, f"{selected}-unknown")
        response = EvaluationResponse(
            evaluation_id=evaluation_id,
            model_type=selected,
            dataset_version=request.dataset_version,
            status=status,
            sample_count=sample_count,
            metrics=metrics,
            model_version=model_version,
        )
        self.evaluations[evaluation_id] = response
        return response

    def get(self, evaluation_id: str) -> EvaluationResponse | None:
        return self.evaluations.get(evaluation_id)


def make_services() -> tuple[ModelRegistry, ClassifierService, EmbeddingService, ForecastService, AnomalyService, TrainingService, EvaluationService]:
    registry = ModelRegistry()
    versions = registry.model_versions()
    classifier = ClassifierService(model_version=versions.get("classifier", MODEL_VERSIONS["classifier"]))
    embedding = EmbeddingService(model_version=versions.get("embedder", MODEL_VERSIONS["embedder"]))
    forecast = ForecastService(model_version=versions.get("forecast", MODEL_VERSIONS["forecast"]))
    anomaly = AnomalyService(model_version=versions.get("anomaly", MODEL_VERSIONS["anomaly"]))
    training = TrainingService(registry)
    evaluation = EvaluationService(registry, classifier, forecast, anomaly)
    return registry, classifier, embedding, forecast, anomaly, training, evaluation
