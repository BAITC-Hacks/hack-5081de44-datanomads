# Pulse 109 — правила проекта

## Приоритеты

Соблюдать порядок: correctness → simplicity → readability → reliability.
Изменять только файлы в области текущей задачи, сохранять пользовательские
изменения и не коммитить raw data, credentials или model artifacts с PII.

## Контракты

- PostgreSQL — source of truth для Pulse state; Qdrant хранит только vectors и
  минимальный allow-listed payload.
- `AI prediction` и `operator_confirmed_decision` всегда раздельны.
- Frontend вызывает только Core API. Core API — единственная граница auth/RBAC
  и единственный клиент ML service.
- `/healthz` — liveness, `/readyz` — readiness; public Nginx не публикует
  `/internal/*`.
- Feedback не запускает online retraining. Candidate model проходит
  `COLLECT → TRAINING → EVALUATE → human PROMOTE/REJECT`.

## Проверки перед commit

```bash
docker compose --profile demo config
cargo test --manifest-path backend/Cargo.toml
npm run build --prefix frontend
scripts/smoke
```

Запускайте только применимые проверки, но в итоговом сообщении указывайте
пропущенные и причины. При изменении OpenAPI синхронизируйте
`docs/openapi/core.openapi.yaml` и `docs/openapi/ml.openapi.yaml`.

## Безопасность

Не логировать полный ticket text, IIN, phone, name, full address или
attachments. Не отправлять PII внешнему LLM без явного обоснования. Не делать
free-form Text-to-SQL и не добавлять инфраструктуру «на будущее» (Kafka,
RabbitMQ, Redis, Kubernetes и т. п.) без ADR с конкретным blocker.
