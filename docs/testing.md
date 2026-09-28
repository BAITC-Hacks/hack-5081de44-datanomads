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
health dependencies. Она не запускает контейнеры. `demo-seed` после старта
проверяет checksum fixture, загружает 160 synthetic tickets и завершается с
кодом 0; повторный запуск должен вернуть 160 duplicates и 0 imported rows.
Core healthcheck ждёт успешных миграций и `/readyz`; `ml-worker` и Nginx ждут
успешного завершения seed, поэтому стартовый reindex и public API не
опережают импорт.

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
Временной ряд и прогноз показывают графики ECharts из Core API; точные значения
остаются доступны в списке дат и таблице прогноза. При браузерной проверке
проверяйте обе страницы с demo-данными и ширину экрана 390 px.

### ML

```bash
cd ml-service
python -m contracts.validate --check-demo
cd ..
python -m pytest ml-service/tests
```

Проверяются deterministic preprocessing, RU/KZ language cases, минимум 10
классов при наличии данных, embedding dimension, forecast horizon, model
manifest/checksum, no-PII payload и reproducible seed.
Для forecast дополнительно проверяются weekly `SeasonalNaive`, несколько
rolling backtest окон, MAE/RMSE, состояние короткой истории и persisted
forecast-v1 → actual → forecast-v2 comparisons. Manager-signal policy uses both
versions' measured backtest MAE and requires verified real-data provenance.

### API contract

`core.openapi.yaml` сверяется с методами Axum routes и Core route index;
`ml.openapi.yaml` сверяется с `FastAPI.app.openapi()`. Smoke на запущенном
стеке проверяет, что публичный Nginx возвращает 404 на docs paths. Локально
ML Swagger и Core route index доступны на loopback-портах из `.env.example`.
Любое изменение request/response требует обновления схемы и обратного smoke
check.

### E2E / smoke

`scripts/smoke` проверяет `/healthz`, `/readyz`, overview и
`POST /api/v1/assist/preview` через Nginx, то есть проходит тот же public
gateway, что и браузер. Для CI base URL задаётся `PULSE_BASE_URL`.
Stateful E2E дополнительно держит `/api/v1/events` открытым и проверяет
`alerts.snapshot` и `alerts.changed` после подтверждения оповещения.

Минимальный E2E:

```text
demo ticket → preview → prediction → operator confirm/correct
→ persisted feedback → close COLLECT → candidate training and shadow window
→ fresh ticket shadow evidence → close evaluation → human promote/reject
→ analytics → alert → forecast → export
```

Полный stateful acceptance-контур запускается через public Nginx gateway:

```bash
PULSE_BASE_URL=http://localhost:8080 python scripts/e2e_acceptance.py
```

Для live PII sentinel и запрета утечки в логах запускайте smoke-проверку после
E2E. Она отправит только synthetic sentinel-строки, затем захватит логи Core и
ML контейнеров и проверит их до завершения; файл остаётся в игнорируемом
`.tmp/`:

```bash
PULSE_BASE_URL=http://localhost:8080 PULSE_ROLE_HEADER_PROBE=1 \
  python scripts/smoke_test.py --pii-probe --capture-compose-logs \
  --log-file .tmp/pulse109-compose.log --require-log-check
```

Для acceptance-gate с проверкой перезапуска Core и восстановления решения:

```bash
PULSE_BASE_URL=http://localhost:8080 python scripts/e2e_acceptance.py --restart-core
```

Он импортирует уникальный synthetic dataset и повторяет его для проверки
idempotency, затем проверяет PostgreSQL/Qdrant preview, refetch решения после
перезаписи, relation feedback, analytics drill-down, QueryIntent, forecast
30/60/90 и идемпотентную rolling forecast version, spike detector → manager monitoring period → ACK/CLOSE → SSE, PDF/XLSX, RBAC, Qdrant reindex и
learning-cycle. В normal mode ожидается реальный candidate artifact; test-only
fake trainer включается отдельно и не считается реальной ML-метрикой. Normal
acceptance также проверяет окно `EVALUATE`, checksum-pinned shadow prediction
для нового ticket, ссылку на его operator decision, сохранение production
model version и ручной переход в `DECISION` без promotion.

## Data/PII safety tests

- bad CSV/invalid date/missing field отправляются в quarantine с причиной;
- frozen evaluation IDs/version передаются candidate builder-у и не попадают в candidate train;
- payload `TRAIN_CLASSIFIER` содержит только version/artifact references и не содержит ticket text;
- normal `TRAIN_CLASSIFIER` job builds a checksummed candidate artifact and never changes the production pointer;
- shadow classification requires the registered artifact checksum and never falls back to production;
- evaluation reads do not close the window, while expiry/manual close advance it to `DECISION`;
- synthetic candidate artifacts are rejected by production shadow serving;
- fake trainer is rejected when `PULSE_ENV=production`, and synthetic candidates cannot be promoted there;
- candidate builder failure виден как `DATASET_BUILD_FAILED` и `background_jobs.FAILED`;
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

Observability checks validate structured JSON fields, safe trace correlation,
nonnegative finite `latency_ms`, and PII-free logs. Smoke reports nearest-rank
p50/p95 grouped by service and endpoint without imposing an SLA threshold or
using request/ticket/user IDs as metric labels. Readiness failures must include
dependency status and a safe error code; demo-only dependencies are explicitly
`not_applicable`.

## Acceptance evidence

Каждый demo run сохраняет commit SHA, compose config, model/dataset versions,
evaluation timestamp и команды проверки. Если полного production dataset нет,
README и отчёт прямо обозначают demo/synthetic scope и не называют его
реальной model quality.
