from __future__ import annotations

import math
import asyncio

import httpx

from app.main import app


class ASGIClient:
    """Sync facade over httpx ASGITransport (works across httpx releases)."""

    async def _request(
        self,
        method: str,
        url: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, json=json, headers=headers)

    def request(
        self,
        method: str,
        url: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return asyncio.run(self._request(method, url, json, headers))

    def get(self, url: str) -> httpx.Response:
        return self.request("GET", url)

    def post(self, url: str, json: dict) -> httpx.Response:
        return self.request("POST", url, json)


client = ASGIClient()


def test_health_and_readiness() -> None:
    health = client.get("/healthz")
    ready = client.get("/readyz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_ru_kz_classifier_is_deterministic_and_has_topics() -> None:
    ru_payload = {"text": "В нашем доме нет горячей воды", "top_k": 3}
    first = client.post("/internal/v1/classify", json=ru_payload)
    second = client.post("/internal/v1/classify", json=ru_payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["language"] == "RU"
    assert first.json()["topic_id"] == "water_supply"
    assert len(first.json()["alternatives"]) == 3

    kz = client.post("/internal/v1/classify", json={"text": "Көшеде жарық жоқ"})
    assert kz.status_code == 200
    assert kz.json()["language"] == "KZ"
    assert kz.json()["topic_id"] == "street_lighting"


def test_mixed_unknown_and_low_confidence_classifications_require_review() -> None:
    for text, language in (("әлеуметтік мәселе", "MIXED"), ("ticket 123", "UNKNOWN")):
        response = client.post(
            "/internal/v1/classify",
            json={"text": text, "language": language},
        )
        assert response.status_code == 200
        prediction = response.json()["predictions"][0]
        assert prediction["language"] == language
        assert prediction["needs_review"] is True

    low_confidence = client.post(
        "/internal/v1/classify",
        json={"text": "unrecognized text", "language": "RU"},
    )
    assert low_confidence.status_code == 200
    prediction = low_confidence.json()["predictions"][0]
    assert prediction["confidence_state"] == "LOW_CONFIDENCE"
    assert prediction["needs_review"] is True


def test_inference_trace_headers_are_documented_and_request_id_is_preserved() -> None:
    header_values = {"x-request-id": "assist-preview-17", "x-trace-id": "trace-17"}
    response = client.request(
        "POST",
        "/internal/v1/classify",
        json={"text": "Нет воды", "request_id": header_values["x-request-id"]},
        headers=header_values,
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == header_values["x-request-id"]

    openapi = app.openapi()
    for path in ("/internal/v1/classify", "/internal/v1/embed"):
        parameters = openapi["paths"][path]["post"]["parameters"]
        documented = {parameter["name"] for parameter in parameters}
        assert {"x-request-id", "x-trace-id"}.issubset(documented)


def test_batch_embedding_is_repeatable_and_normalized() -> None:
    payload = {"texts": ["Нет воды", "Нет воды"], "dimension": 16}
    response = client.post("/internal/v1/embed", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == 16
    assert body["embeddings"][0] == body["embeddings"][1]
    assert math.isclose(sum(value * value for value in body["embeddings"][0]), 1.0, rel_tol=1e-5)


def test_forecast_seasonal_naive_and_short_history_state() -> None:
    response = client.post(
        "/internal/v1/forecast",
        json={"values": [1, 2, 3, 4, 5, 6, 7], "horizon": 9, "season_length": 7},
    )
    assert response.status_code == 200
    assert response.json()["forecast"] == [1, 2, 3, 4, 5, 6, 7, 1, 2]
    assert response.json()["status"] == "OK"

    for horizon in (30, 60, 90):
        forecast = client.post(
            "/internal/v1/forecast",
            json={"values": [1, 2, 3, 4, 5, 6, 7], "horizon": horizon, "season_length": 7},
        )
        assert forecast.status_code == 200
        assert len(forecast.json()["forecast"]) == horizon

    rolling = client.post(
        "/internal/v1/forecast",
        json={"values": [1, 2, 3, 4, 5, 6, 7] * 3, "horizon": 7, "season_length": 7},
    )
    assert rolling.status_code == 200
    assert rolling.json()["backtest"]["window_count"] == 2
    assert rolling.json()["backtest"]["sample_count"] == 14
    assert rolling.json()["backtest"]["mae"] == 0
    assert rolling.json()["model"] == "seasonal_naive"
    manifest = client.get("/internal/v1/models").json()
    assert manifest["models"]["forecast"]["model_version"] == rolling.json()["model_version"]

    short = client.post("/internal/v1/forecast", json={"values": [9, 10], "horizon": 2, "season_length": 7})
    assert short.status_code == 200
    assert short.json()["status"] == "INSUFFICIENT_HISTORY"
    assert short.json()["forecast"] == []
    assert short.json()["points"] == []
    assert short.json()["expected_peaks"] == []


def test_anomaly_flags_latest_spike() -> None:
    response = client.post(
        "/internal/v1/anomaly",
        json={"values": [10, 11, 10, 10, 12, 11, 60], "window": 5, "threshold": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is True
    assert body["latest"]["is_anomaly"] is True
    assert body["anomalies"][-1]["index"] == 6


def test_versioned_model_manifest_preserves_demo_fields_and_checks_explicit_version(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    manifest = client.get("/internal/v1/models")
    assert manifest.status_code == 200
    body = manifest.json()
    assert body["manifest_version"] == "1"
    assert body["schema_version"] == "model-manifest.v1"
    classifier = body["models"]["classifier"]
    assert classifier["model_version"] == "classifier-demo-2026-09-21-001"
    assert classifier["dataset_version"] == "demo-ru-kz-v1"
    assert classifier["artifact_checksum"] is None
    assert classifier["demo_artifact_id"].startswith("demo-baseline:")
    assert classifier["artifact_kind"] == "DETERMINISTIC_BASELINE"
    assert classifier["synthetic"] is True

    accepted = client.post(
        "/internal/v1/classify",
        json={"text": "Нет воды", "model_version": classifier["model_version"]},
    )
    rejected = client.post(
        "/internal/v1/classify",
        json={"text": "Нет воды", "model_version": "candidate-not-loaded"},
    )
    assert accepted.status_code == 200
    assert rejected.status_code == 404

    candidate_without_artifact = client.post(
        "/internal/v1/classify",
        json={
            "text": "Нет воды",
            "model_version": "candidate-not-loaded",
            "expected_artifact_checksum": f"sha256:{'0' * 64}",
        },
    )
    assert candidate_without_artifact.status_code == 404
    assert candidate_without_artifact.json()["detail"] == "CANDIDATE_MODEL_NOT_FOUND"


def test_training_evaluation_and_manifest(monkeypatch) -> None:
    training = client.post(
        "/internal/v1/training/classifier",
        json={
            "dataset_version": "feedback-1",
            "samples": [{"text": "Нет воды", "label": "water_supply"}],
            "min_samples": 1,
        },
    )
    assert training.status_code == 200
    assert training.json()["state"] == "TRAINER_NOT_CONFIGURED"
    assert training.json()["candidate_model_version"] is None
    job_id = training.json()["job_id"]
    assert client.get(f"/internal/v1/training/jobs/{job_id}").status_code == 200

    monkeypatch.setenv("PULSE_ENV", "test")
    monkeypatch.setenv("PULSE_TEST_FAKE_TRAINER", "true")
    fake_training = client.post(
        "/internal/v1/training/classifier",
        json={"dataset_version": "feedback-1", "samples": [{"text": "Нет воды", "label": "water_supply"}], "min_samples": 1},
    )
    assert fake_training.json()["state"] == "COMPLETED"
    monkeypatch.delenv("PULSE_TEST_FAKE_TRAINER")

    evaluation = client.post(
        "/internal/v1/evaluation/classifier",
        json={"texts": ["Нет воды", "Автобус не приехал"], "labels": ["water_supply", "public_transport"]},
    )
    assert evaluation.status_code == 200
    assert evaluation.json()["metrics"]["macro_f1"] == 1.0
    assert client.get(f"/internal/v1/evaluation/jobs/{evaluation.json()['evaluation_id']}").status_code == 200

    manifest = client.get("/internal/v1/models")
    assert manifest.status_code == 200
    assert len(manifest.json()["models"]["classifier"]["labels"]) >= 10
