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
