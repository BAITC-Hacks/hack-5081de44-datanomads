"""Local, versioned classifier artifact for the internal inference API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .classifier_input import encode_classifier_texts
from .confidence import ConfidencePolicy
from .constants import TOPIC_BY_ID
from .schemas import Alternative, Classification, ModelMetadata
from .services import detect_language


class TrainedClassifierService:
    def __init__(self, model_dir: Path) -> None:
        metadata_path = model_dir / "manifest.json"
        self.metadata = ModelMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
        if self.metadata.metrics.get("status") == "validation_only":
            raise ValueError("validation-only classifier artifact cannot serve predictions")
        if set(self.metadata.labels) != set(TOPIC_BY_ID) - {"other"}:
            raise ValueError("classifier artifact labels do not match the canonical taxonomy")
        model_file = model_dir / "model.safetensors"
        with model_file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if self.metadata.artifact_checksum != f"sha256:{checksum}":
            raise ValueError("classifier artifact checksum mismatch")

        self.model_version = self.metadata.model_version
        self.max_length = int(self.metadata.training_config["max_length"])
        self.input_length_strategy = self.metadata.training_config.get(
            "input_length_strategy", f"head-{self.max_length}")
        if self.input_length_strategy not in (f"head-{self.max_length}", f"head-tail-{self.max_length}"):
            raise ValueError("classifier artifact has unsupported input length strategy")
        self.temperature = float(self.metadata.training_config["temperature"])
        if self.temperature < 1.0:
            raise ValueError("classifier temperature must be at least 1")
        self.confidence_policy = ConfidencePolicy.from_metadata(self.metadata)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
        model_labels = [self.model.config.id2label[index] for index in range(self.model.config.num_labels)]
        if model_labels != self.metadata.labels:
            raise ValueError("classifier model labels do not match artifact manifest")

    @torch.inference_mode()
    def classify(self, text: str, language: str | None = None, top_k: int = 3) -> Classification:
        detected = detect_language(text, language)
        encoded = encode_classifier_texts(self.tokenizer, [text], max_length=self.max_length,
                                          strategy=self.input_length_strategy, pad_to_max_length=False)
        logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits[0]
        probabilities = torch.softmax(logits.float() / self.temperature, dim=-1)
        values, indices = probabilities.topk(min(top_k, len(self.metadata.labels)))
        winner_id = self.metadata.labels[indices[0].item()]
        confidence = values[0].item()
        state = self.confidence_policy.state(confidence)
        if state == "UNCERTAIN":
            values, indices = values[:2], indices[:2]
        alternatives = []
        for value, index in zip(values.tolist(), indices.tolist()):
            topic_id = self.metadata.labels[index]
            topic = TOPIC_BY_ID[topic_id]
            alternatives.append(Alternative(
                topic_id=topic_id,
                topic=topic.name_kz if detected == "KZ" else topic.name_ru,
                score=round(value, 6),
                confidence=round(value, 6),
            ))
        topic = TOPIC_BY_ID[winner_id]
        return Classification(
            language=detected,
            topic_id=winner_id,
            topic=topic.name_kz if detected == "KZ" else topic.name_ru,
            label=winner_id,
            confidence=round(confidence, 6),
            confidence_state=state,
            needs_review=True,
            alternatives=alternatives,
            model_version=self.model_version,
        )
