# Тестирование и verification

Цель тестов — доказать границы P0, а не только успешный healthcheck. Synthetic
demo fixtures не заменяют held-out evaluation на реальных данных.

## Быстрый локальный контур

```bash
cp .env.example .env
docker compose --profile demo config
docker compose --profile demo up --build -d
scripts/smoke
```

После тестов логи и статусы проверяются командами:

```bash
docker compose --profile demo ps
docker compose --profile demo logs --tail=200 core-api ml-service ml-worker nginx
```

`docker compose ... config` — обязательная проверка разрешения путей, env и
health dependencies. Она не запускает контейнеры.

## Уровни

### Backend

```bash
cargo test --manifest-path backend/Cargo.toml
cargo fmt --manifest-path backend/Cargo.toml --all -- --check
```

Проверяются schema/validation, RBAC, state transitions, transaction boundaries,
error mapping и отсутствие смешения AI prediction/operator decision.

### Frontend

```bash
npm ci --prefix frontend
npm run build --prefix frontend
```

Проверяются реальные routes Operator Workspace/Situation Center, loading,
empty/error states, keyboard navigation, mobile layout и отсутствие прямого
вызова ML service.

### ML

```bash
python -m pytest ml-service/tests
```

Проверяются deterministic preprocessing, RU/KZ language cases, минимум 10
классов при наличии данных, embedding dimension, forecast horizon, model
manifest/checksum, no-PII payload и reproducible seed.

### API contract

OpenAPI-файлы в `docs/openapi/` валидируются YAML/OpenAPI parser-ом и
сверяются с route handlers. Любое изменение публичного request/response
требует обновления схемы и обратного smoke check.

### E2E / smoke

`scripts/smoke` проверяет `/healthz`, `/readyz`, overview и
`POST /api/v1/assist/preview` через Nginx, то есть проходит тот же public
gateway, что и браузер. Для CI base URL задаётся `PULSE_BASE_URL`.

Минимальный E2E:

```text
demo ticket → preview → prediction → operator confirm/correct
→ persisted feedback → close collect → candidate evaluation
→ human promote/reject → analytics → alert → forecast → export
```

## Data/PII safety tests

- bad CSV/invalid date/missing field отправляются в quarantine с причиной;
- frozen evaluation set не попадает в candidate train;
- полный ticket text, IIN, phone, name, address и attachments отсутствуют в
  default logs;
- Qdrant payload содержит только allow-listed metadata;
- optional LLM получает только строгий `QueryIntent` без PII;
- free-form Text-to-SQL и прямой frontend → ML route невозможны.

## Негативные сценарии

Тесты должны явно проверять: недоступный Qdrant/ML dependency → `readyz` не
готов; ML timeout → понятная ошибка без потери ticket state; недостаточный
feedback → `INSUFFICIENT_FEEDBACK`; candidate с critical regression нельзя
promote; rejected candidate не меняет production pointer; unauthorized role
получает 401/403; malformed QueryIntent не выполняет SQL.

## Acceptance evidence

Каждый demo run сохраняет commit SHA, compose config, model/dataset versions,
evaluation timestamp и команды проверки. Если полного production dataset нет,
README и отчёт прямо обозначают demo/synthetic scope и не называют его
реальной model quality.
