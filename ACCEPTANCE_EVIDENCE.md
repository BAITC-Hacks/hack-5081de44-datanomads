# Non-ML P0 acceptance evidence

Commit: `b2a37e9` (`fix(p0): close non-ML acceptance gaps`)

## Exact commands and results

```text
docker compose --profile demo config
PASS (213 rendered lines)

COMPOSE_PROJECT_NAME=pulse109-final PULSE_HTTP_PORT=18082 POSTGRES_PORT=15434 \
  QDRANT_HTTP_PORT=16337 QDRANT_GRPC_PORT=16338 \
  docker compose --profile demo up --build -d
PASS: fresh PostgreSQL, Qdrant and ML volumes; /readyz = ready

COMPOSE_PROJECT_NAME=pulse109-final PULSE_BASE_URL=http://127.0.0.1:18082 \
  python scripts/e2e_acceptance.py --restart-core
PASS: status=passed; normal learning result=TRAINER_NOT_CONFIGURED_AND_REJECTED

docker compose -p pulse109-final logs --no-color --timestamps \
  core-api ml-service ml-worker nginx > /tmp/pulse109-final.log
PULSE_BASE_URL=http://127.0.0.1:18082 \
  python scripts/smoke_test.py --log-file /tmp/pulse109-final.log --require-log-check
PASS: 65/65 checks; PII sentinels absent; structured observability present

cargo fmt --manifest-path backend/Cargo.toml -- --check
cargo test --manifest-path backend/Cargo.toml
PASS: 10 unit + 2 API tests; doc-tests 0

npm run build --prefix frontend
PASS: TypeScript/Vite production build

python -m unittest discover -s data/tests -p 'test_*.py'
python -m unittest discover -s tests/contract -p 'test_*.py'
PASS: 9 data tests + 5 contract tests

docker run --rm -v "$PWD/ml-service":/work -w /work -e PYTHONPATH=/app \
  pulse109/ml-service:local pytest -q tests
PASS: 6 ML-service tests

COMPOSE_PROJECT_NAME=pulse109-final-fake PULSE_TEST_FAKE_TRAINER=true \
  PULSE_HTTP_PORT=18083 POSTGRES_PORT=15435 QDRANT_HTTP_PORT=16339 \
  QDRANT_GRPC_PORT=16340 docker compose --profile demo up --no-build -d
COMPOSE_PROJECT_NAME=pulse109-final-fake PULSE_BASE_URL=http://127.0.0.1:18083 \
  python scripts/e2e_acceptance.py --restart-core
PASS: status=passed; learning result=PROMOTED_TEST_CANDIDATE
```

The clean PostgreSQL invariants after E2E were:

```text
routing_official=0
priority_official=0
templates_approved=0
templates_manual=34
forecast_model=forecast-seasonal-naive-2026-09-21-001
```

## Known external blockers

- Authoritative 109 routing/priority rules and response templates have not
  been supplied. Current seeded mappings are `MANUAL`; templates are
  `approved=false` and exposed as `MANUAL_DEMO`.
- Production trusted-auth-gateway/JWT integration still depends on the
  deployment's external identity contract; the acceptance stack uses its
  documented demo auth mode.

Real classifier/embedder training is **NOT implemented yet**. XLM-R/E5
fine-tuning, real model artifacts and held-out ML metrics remain explicitly
outside this closure. The deterministic classifier/embedder and Seasonal Naive
forecast are baselines only; the test-only fake trainer is never enabled in
normal Compose mode.

## Дополнительная проверка 2026-09-24

На отдельном `pulse109-freshcheck` stack команда
`docker compose --profile demo up --build -d` автоматически импортировала 160
synthetic tickets; повторный seed вернул 160 duplicates и 0 новых записей.
Forecast использовал 267 дней истории и 38 rolling backtest окон с версией
`forecast-statsforecast-seasonal-naive-2026-09-24-001`.

После миграции 008 на сохранённых volumes прошли `scripts/smoke` и
`scripts/e2e_acceptance.py`. E2E проверил `409` при изменении содержимого уже
зарегистрированной версии. Read-only SQL показал 160 связей demo dataset с
обращениями и quarantine snapshot `{"content_redacted": true}`. Дополнительно
прошли 10 Rust unit + 3 API tests, 15 data/contract tests, 6 ML tests, обе
OpenAPI YAML схемы успешно разобраны parser-ом. Тестовые контейнеры остановлены;
volumes оставлены без удаления.

## Финальная повторная проверка 2026-09-24

На отдельном свежем проекте `pulse109-finalcheck` команда
`docker compose --profile demo up --build -d` подняла frontend, Core API,
ML service/worker, PostgreSQL, Qdrant и Nginx. `demo-seed` автоматически
импортировал 160 synthetic tickets (20 регионов, 16 тем, RU/KZ).

Через public Nginx на `127.0.0.1:18185` прошли `scripts/smoke`,
`scripts/e2e_acceptance.py` (`status=passed`, normal learning:
`TRAINER_NOT_CONFIGURED_AND_REJECTED`) и `scripts/smoke_test.py` с проверкой
логов (`60/60`). Проверены импорт и конфликт изменённой source record,
PostgreSQL/Qdrant preview, решение оператора, feedback, learning state,
analytics, alerts/SSE, forecast, PDF/XLSX и RBAC.

Дополнительно прошли: 10 Rust unit + 3 API tests, 10 data tests, 7 contract
tests, 7 ML tests, TypeScript/Vite build, deterministic fixture check и
`docker compose --profile demo config`. В браузере проверены графики
временного ряда и 30 точек прогноза; при ширине 390 px горизонтального
переполнения и ошибок консоли нет. YAML Core и ML OpenAPI разобраны parser-ом;
Core OpenAPI покрывает все 50 операций Axum, ML YAML совпадает с FastAPI.

Оставшиеся внешние зависимости: реальные 109 dataset и source update contract,
официальные routing/priority rules и утверждённые response templates,
production identity gateway. Дообученный classifier/embedder и его held-out
метрики отложены отдельно; обычный runtime не выдаёт фиктивный candidate.
