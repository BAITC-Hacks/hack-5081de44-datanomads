# Архитектура Pulse 109

Этот документ фиксирует P0-контур и границы ответственности сервисов. Pulse —
интеллектуальный слой над существующей системой 109, а не новая CRM: официальные
статус, исполнитель, история действий и отправка ответа остаются во внешней
системе 109.

## Контур выполнения

```text
Browser
  │ HTTP(S)
  ▼
nginx :8080 (public gateway)
  ├── /         ──► frontend :5174
  └── /api/*    ──► core-api :8080
                         ├──► postgres :5432  (Pulse source of truth)
                         ├──► qdrant :6333    (vectors + minimal payload)
                         └──► ml-service :8000
                                ├── classifier
                                ├── embeddings
                                ├── anomaly
                                └── forecast

ml-worker (profile demo)
  └── PostgreSQL-backed jobs / offline training / evaluation
```

`docker compose --profile demo up --build` поднимает локальный воспроизводимый
контур. Внешний LLM/API не является обязательной зависимостью.

## Границы сервисов

| Сервис | Внутренний порт | Ответственность | Не делает |
| --- | ---: | --- | --- |
| `frontend` | 5174 | React Operator Workspace и Situation Center | не вызывает ML напрямую |
| `core-api` | 8080 | auth/RBAC, валидация, orchestration, транзакции, аналитика, feedback, SSE, экспорт | не обучает модели в HTTP-запросе |
| `ml-service` | 8000 | inference, embeddings, forecast, anomaly, training/evaluation API, model metadata | не хранит бизнес-состояние и права пользователей |
| `ml-worker` | — | PostgreSQL-backed jobs, offline training/evaluation/reindex | не меняет production-модель без promotion |
| `postgres` | 5432 | нормализованные tickets, решения, feedback, alerts, model/learning metadata | не заменяется Qdrant |
| `qdrant` | 6333/6334 | vector index и фильтруемый минимальный payload | не является источником истины и не хранит PII |
| `nginx` | 80 | public gateway, request-id, SSE proxy, service isolation | не публикует `/internal/*` |

Путь между контейнерами использует имена Compose-сервисов. Frontend должен
слушать `0.0.0.0:5174`, Core API — `0.0.0.0:8080`, ML service —
`0.0.0.0:8000`. Эти порты не являются публичным API; наружу публикуется только
Nginx (`PULSE_HTTP_PORT`, по умолчанию 8080).

## Потоки данных

### Обращение и operator workflow

```text
UnifiedTicket
  → core-api validation
  → ML classification / embedding через internal API
  → PostgreSQL: AI prediction (immutable observation)
  → frontend: uncertainty, alternatives, service, priority, similar tickets
  → operator confirm/correct
  → PostgreSQL: operator_confirmed_decision + feedback
```

AI prediction и подтверждённое человеком решение — разные поля и разные
жизненные циклы. Исправление оператора не запускает немедленное переобучение.

### Retrieval

Текст → embedding → Qdrant top-K `ticket_id` → PostgreSQL fetch разрешённых
полей. В payload Qdrant допускаются только `ticket_id`, `region_id`, `topic_id`
и `created_at` (плюс технический vector id); исходный текст и PII остаются в
PostgreSQL/разрешённом представлении.

### Controlled Learning Loop

```text
COLLECT → versioned dataset → offline TRAIN → candidate EVALUATE/shadow
        → human PROMOTE или REJECT
```

Production model продолжает обслуживать запросы во время training. Состояние
цикла, dataset/model versions, метрики и решение сохраняются в PostgreSQL.

## Надёжность и наблюдаемость

- `/healthz` означает, что процесс жив; `/readyz` — что обязательные зависимости
  доступны.
- Core API пишет structured JSON с `request_id`, `trace_id`, `service`,
  `endpoint`, `latency_ms`, `model_version`, `status`, `error_code`.
- В приложенческий лог по умолчанию не попадают полный текст обращения, ИИН,
  телефон, имя, полный адрес и вложения.
- PostgreSQL-backed jobs используют lease/attempt и `FOR UPDATE SKIP LOCKED`;
  Redis, Kafka и отдельный message broker в P0 не нужны.

## Масштабирование и ограничения

P0 — modular monolith Core API плюс отдельный Python ML runtime. Kubernetes,
Kafka, RabbitMQ, Celery, Redis, ClickHouse, Elasticsearch, service mesh и
внешний SaaS не являются частью локального demo. Если появится необходимость
в дополнительной инфраструктуре, она оформляется отдельным ADR с конкретным
blocker и планом миграции.
