# Pulse 109 Core API

Локальный P0 backend — один Axum-сервис с deterministic demo fixtures. По
умолчанию state хранится в памяти, поэтому запуск не требует PostgreSQL и не
маскирует demo-данные под реальные метрики. Контракт изолирован в `AppState`,
что оставляет простой путь для PostgreSQL repository в production.

## Запуск

```bash
cargo run --manifest-path backend/Cargo.toml
```

Сервис слушает `0.0.0.0:8080`. Настройки: `PULSE_HOST`, `PULSE_PORT` и
`PULSE_DEV_AUTH`. Для Docker:

```bash
docker build -t pulse109-core backend
docker run --rm -p 8080:8080 pulse109-core
```

## Контракт

- `GET /healthz`, `GET /readyz` — liveness/readiness.
- `GET/POST /api/v1/tickets`, `GET /api/v1/tickets/{ticket_id}` — тикеты и
  deterministic RU/KZ fixtures.
- `POST /api/v1/assist/preview` — принимает `ticket_id` или `{ "text": ... }`
  и возвращает prediction, confidence state, alternatives, service, похожие,
  duplicate/repeat candidates и response template.
- `POST /api/v1/assist/{ticket_id}/confirm` и `/correct` — фиксируют operator
  decision отдельно от AI prediction. Для клиентов, которым удобнее передавать
  id в JSON, доступны также `/api/v1/assist/confirm` и `/correct`.
- `GET /api/v1/analytics`, `GET /api/v1/forecast` — Situation Center.
- `GET /api/v1/alerts`, `GET /api/v1/alerts/{alert_id}` и `POST .../ack`.
- `GET/POST /api/v1/learning`, cycle feedback/promote/reject — controlled
  learning loop без автоматического promotion.
- `GET /api/v1/models`, `GET/POST /api/v1/models/{model_id}` — model registry.
- `GET /api/v1/openapi.json` и `/api/v1/docs` — OpenAPI-ish описание.

Development auth использует `x-pulse-role: OPERATOR|MANAGER|ML_REVIEWER|ADMIN`
и необязательный `x-user-id`. При `PULSE_DEV_AUTH=true` запрос без role header
работает как `ADMIN`, чтобы demo можно было открыть из браузера. Явно переданная
роль всегда проверяется.

Structured JSON logs содержат `service`, `request_id`, `trace_id`, `endpoint`,
`latency_ms`, `status` и не включают полный текст обращения. CORS разрешён для
локального frontend demo.

## Проверки

```bash
cargo fmt --manifest-path backend/Cargo.toml -- --check
cargo check --manifest-path backend/Cargo.toml
cargo test --manifest-path backend/Cargo.toml
```
