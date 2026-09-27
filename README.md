# Pulse 109

Pulse 109 — AI-слой над существующей системой 109 для маршрутизации обращений,
ассистирования оператору и Situation Center. Официальные статус обращения,
исполнитель, история действий и отправка ответа остаются во внешней системе
109; Pulse хранит локальное зеркало разрешённых данных, ML predictions,
operator feedback, аналитику, alerts, forecasts и версии моделей.

## P0 status

| Контур | Статус в репозитории |
| --- | --- |
| Compose topology, Nginx gateway и env contract | demo-контур реализован |
| Operator Workspace / Situation Center | demo-контур, графики и рабочие действия реализованы в `frontend/` |
| Rust Core API | demo-контур реализован в `backend/` |
| Python ML runtime/worker | deterministic classifier/embedder, StatsForecast baseline и PostgreSQL worker; реальный trainer пока отключён |
| Data importers и demo fixtures | import/quarantine и fixtures реализованы в `data/` |
| OpenAPI и инженерные контракты | Core path/method coverage проверяется контрактным тестом; ML схема экспортирована из FastAPI |

Compose-контракт запускает сервисы на frontend `0.0.0.0:5174`, Core API
`0.0.0.0:8080` и ML service `0.0.0.0:8000` внутри Compose. Локальный доступ
идёт через Nginx на `127.0.0.1:PULSE_HTTP_PORT`.

## Быстрый запуск demo

Требуется Docker Engine с Compose v2.

```bash
cp .env.example .env
docker compose --profile demo config
docker compose --profile demo up --build
```

`demo-seed` автоматически сверяет checked-in synthetic fixture с manifest и
идемпотентно импортирует её в PostgreSQL/Qdrant при запуске demo profile.

В другом терминале:

```bash
scripts/smoke
```

Открыть `http://localhost:8080`. Health endpoints: `/healthz` и `/readyz`.
OpenAPI-контракты находятся в `docs/openapi/`; Swagger/OpenAPI UI не
проксируется наружу public contour. В Compose ML Swagger доступен локально на
`http://127.0.0.1:8000/docs`, OpenAPI JSON — на
`http://127.0.0.1:8000/openapi.json`; Core route index — на
`http://127.0.0.1:8081/api/v1/docs`. Эти порты привязаны только к loopback, а
публичный Nginx возвращает 404 для документационных путей.

Для полного локального reset PostgreSQL/Qdrant нужен явный флаг:

```bash
PULSE_CONFIRM_RESET=1 scripts/demo-reset
```

Команда удаляет локальные volumes и заново собирает demo stack.

## Архитектура

```text
Browser → Nginx → React frontend
                 → Rust Core API → PostgreSQL (source of truth)
                                  → Qdrant (vector index)
                                  → Python ML service
ML worker → PostgreSQL-backed jobs (profile demo)
```

Frontend никогда не вызывает ML service напрямую. Core API выполняет
валидацию, auth/RBAC, транзакции, analytics, alert/learning lifecycle и
экспорт. ML service отвечает только за inference, embeddings, anomaly,
forecast, training/evaluation и model metadata.

Основной реализованный demo vertical slice:

```text
demo ticket → assist preview → AI prediction → operator confirm/correct
→ persisted feedback → analytics/alerts → controlled learning cycle → export
```

Controlled Learning Loop не является online self-learning:
`COLLECT → versioned dataset → offline TRAIN → shadow EVALUATE → human
PROMOTE/REJECT`. Candidate не заменяет production автоматически.
До подключения реального trainer цикл завершается явным
`TRAINER_NOT_CONFIGURED`; тестовый candidate не используется в обычном Compose.

## Документация

- [`docs/architecture.md`](docs/architecture.md) — сервисные границы и потоки;
- [`docs/data-contract.md`](docs/data-contract.md) — UnifiedTicket, import и
  quality gate;
- [`docs/contracts.md`](docs/contracts.md) — versioned Data/ML artifact contracts
  and standalone validation;
- [`docs/vko-109-data-audit.md`](docs/vko-109-data-audit.md) — границы
  предоставленной реальной выгрузки Восточно-Казахстанской области;
- [`docs/privacy.md`](docs/privacy.md) — PII, RBAC, logs и LLM boundary;
- [`docs/model-registry.md`](docs/model-registry.md) — immutable artifacts;
- [`docs/learning-loop.md`](docs/learning-loop.md) — lifecycle candidate model;
- [`docs/testing.md`](docs/testing.md) — проверки и acceptance evidence;
- [`docs/deployment.md`](docs/deployment.md) — demo/production checklist;
- [`docs/openapi/core.openapi.yaml`](docs/openapi/core.openapi.yaml) и
  [`docs/openapi/ml.openapi.yaml`](docs/openapi/ml.openapi.yaml) — API contracts.

## Ограничения и честный scope

- Demo fixture — synthetic/deterministic, не реальная статистика и не замена
  полного набора данных 20 регионов.
- Предоставленная выгрузка Восточно-Казахстанской области пригодна для проверки
  агрегатной динамики, но не содержит отдельного текста обращения или языка;
  подробности в `docs/vko-109-data-audit.md`.
- Без полного dataset нельзя честно утверждать held-out качество classifier,
  embeddings, spike detector или forecast; такие метрики должны иметь
  dataset/model/evaluation versions.
- Подключение pretrained `multilingual-e5-base` и дообучение classifier/embedder
  отложены по решению проекта. Demo retrieval использует deterministic
  embeddings; качество поиска на реальных обращениях пока не подтверждено.
- Seeded routing/priority mappings are `MANUAL` demo defaults, not official
  109 rules. Seeded response templates have `approved=false`; operators see
  `MANUAL_REQUIRED` until a reviewed template is explicitly approved.
- Production identity gateway/JWT и контракт синхронизации изменённых
  обращений внешней системы 109 не предоставлены. Demo RBAC использует
  `PULSE_DEV_AUTH`; внешняя публикация Compose без trusted identity запрещена.
- Forecast responses carry their own `forecast_model_version`; the embedding
  version is never used as forecast metadata.
- Optional LLM отключён по умолчанию; аналитика обязана работать через
  allow-listed `QueryIntent` и parameterized SQL.
- Compose demo — single-host среда с локальными volumes и placeholder
  credentials, не production hardening.
- Raw source files, secrets, PII и неманифестированные model artifacts не
  коммитятся.

## Проверки

```bash
docker compose --profile demo config
cargo test --manifest-path backend/Cargo.toml
npm run build --prefix frontend
scripts/smoke
```

Полный `docker compose ... up --build` зависит от доступного Docker Engine и
локального image cache. В текущем Compose Core использует PostgreSQL как source
of truth и Qdrant как vector index; deterministic in-memory store остаётся
только явно выбранным режимом `PULSE_STORAGE=memory` для unit/API тестов.
Synthetic demo не выдаётся за реальные данные или model quality.
