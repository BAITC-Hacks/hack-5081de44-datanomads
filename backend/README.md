# Pulse 109 Core API

Локальный P0 backend — один Axum-сервис с PostgreSQL как источником истины,
Qdrant для retrieval и ML-service для baseline classify/embed/forecast. В памяти
остаётся только явно выбранный `PULSE_STORAGE=memory` режим для unit/API тестов;
Compose и рабочий запуск используют PostgreSQL и возвращают ошибку при отказе
зависимости, а не подменяют её demo-данными.

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
- `GET/POST /api/v1/tickets`, `GET /api/v1/tickets/{ticket_id}` — тикеты,
  prediction и operator decision из PostgreSQL.
- `POST /api/v1/import` — validated CSV/TSV/JSON/JSONL/XLSX ingest с
  `data_import_runs`, `quarantine_rows`, dataset version и автоматической
  индексацией в Qdrant.
- `POST /api/v1/retrieval/reindex` и `DELETE /api/v1/tickets/{ticket_id}/vector`
  — очередь полного backfill и управляемое удаление vector; worker очищает
  stale points и строит collection из PostgreSQL.
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

В Compose Core route index доступен локально через
`http://127.0.0.1:8081/api/v1/docs` (JSON также доступен на
`/api/v1/openapi.json`). Порт привязан к loopback; публичный Nginx эти пути
не проксирует.

В demo/test режиме auth использует `x-pulse-role:
OPERATOR|MANAGER|ML_REVIEWER|ADMIN` и необязательный `x-user-id`. В normal
режиме Core принимает только заголовки, выставленные доверенным auth gateway:
`x-authenticated-role` и `x-authenticated-user`; `PULSE_DEV_AUTH=true` разрешён
только для `PULSE_ENV=demo|test|unit` и останавливает Core при включении в
normal/production режиме. Core не проверяет JWT: внешний gateway обязан
проверить identity token, удалить одноимённые заголовки от клиента и выставить
доверенные `x-authenticated-*` перед пересылкой запроса. Явно переданная роль
всегда проверяется.

Structured JSON logs содержат `service`, `request_id`, `trace_id`, `endpoint`,
`latency_ms`, `status` и не включают полный текст обращения. CORS разрешён для
локального frontend demo.

## Проверки

```bash
cargo fmt --manifest-path backend/Cargo.toml -- --check
cargo check --manifest-path backend/Cargo.toml
cargo test --manifest-path backend/Cargo.toml
```
