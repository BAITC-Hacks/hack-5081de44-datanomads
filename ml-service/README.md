# Pulse 109 ML Service

Изолированный FastAPI runtime для внутреннего ML-контура Pulse 109. Сервис
не содержит авторизации, бизнес-состояния, SQL или операторского workflow —
это ответственность Rust Core API.

## Что входит

- `/healthz` и `/readyz` для liveness/readiness;
- `POST /internal/v1/classify` — RU/KZ классификация по 16 темам, confidence
  state и alternatives. По умолчанию работает детерминированный baseline с
  дополнительным ответом `other`; локальный XLM-R включается явно;
- `POST /internal/v1/embed` — воспроизводимый локальный hashed embedding без
  загрузки внешней модели;
- `POST /internal/v1/forecast` — StatsForecast `SeasonalNaive` baseline и
  rolling-origin backtest (MAE/RMSE/WAPE/sMAPE, число окон и наблюдений);
  история короче сезона помечается `INSUFFICIENT_HISTORY`;
  горизонты до 366 точек (30/60/90 поддерживаются параметром `horizon`);
- `POST /internal/v1/anomaly` — rolling median/MAD anomaly detector;
- `POST /internal/v1/training[/<model_type>]` — inline-sample helper; без
  test-only fake отвечает `TRAINER_NOT_CONFIGURED`. Постоянный `TRAIN_CLASSIFIER`
  job проверяет immutable candidate dataset и вызывает версионированный
  Multinomial Naive Bayes trainer из `app.training`;
- `POST /internal/v1/evaluation[/<model_type>]` — classifier/forecast/anomaly
  baseline metrics;
- `GET /internal/v1/models` и `/internal/v1/models/<model_type>` — immutable
  version manifest. Shadow candidates are loaded on demand only when classify
  receives an explicit model version and expected artifact checksum; there is
  no fallback to the configured production classifier.
- The PostgreSQL worker can opt into one global recurring classifier cycle with
  `PULSE_LEARNING_AUTO_CYCLES_ENABLED=true`. It waits for an eligible terminal
  state, a registered frozen evaluation dataset and a production model; it does
  not enable the test fake trainer or bypass human promotion.
- The shared `drift-evidence.v1` contract carries aggregate detector output to
  Core's trusted `ML_SERVICE` drift-trigger endpoint. The deterministic runtime
  does not emit calibrated drift scores; its checked-in example is synthetic.

Это честный deterministic demo baseline, а не утверждение о качестве на
реальном dataset 109. `artifacts/manifest.json` явно содержит
`schema_version`, `dataset_version`, `model_version`, `artifact_kind`,
`synthetic`, `metrics`, labels и checksum-поле. Demo identifier не является
криптографическим checksum. Trained artifact обязан иметь `sha256:<64 hex>` и
artifact URI. The deterministic baseline remains the configured production/demo
model. Versioned Multinomial Naive Bayes candidate artifacts use the same
contracts and a checksum-verifying shadow adapter; they do not replace that
configured model until Core records an authorized human promotion.

## Локальный запуск

```bash
cd ml-service
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m contracts.validate --check-demo
uvicorn app.main:app --reload --no-access-log
```

Документация OpenAPI доступна на `http://127.0.0.1:8000/docs`, JSON-схема — на
`http://127.0.0.1:8000/openapi.json` при отдельном локальном запуске сервиса.
Compose не публикует этот порт на хост; из внутренней сети доступны
`http://ml-service:8000/docs` и `/openapi.json`. Публичный Nginx не
проксирует `/docs`, `/redoc` или `/openapi.json`.

## Примеры

```bash
curl -s http://localhost:8000/healthz

curl -s http://localhost:8000/internal/v1/classify \
  -H 'content-type: application/json' \
  -d '{"text":"На улице не горят фонари", "top_k": 3}'

curl -s http://localhost:8000/internal/v1/embed \
  -H 'content-type: application/json' \
  -d '{"text":"Нет воды в доме", "dimension": 32}'

curl -s http://localhost:8000/internal/v1/forecast \
  -H 'content-type: application/json' \
  -d '{"values":[10,12,11,13,10,12,14,11,13,12,14,12,13,15],"horizon":30,"season_length":7}'

curl -s http://localhost:8000/internal/v1/anomaly \
  -H 'content-type: application/json' \
  -d '{"values":[10,11,10,10,12,11,60],"window":5,"threshold":3}'
```

## Тесты

```bash
pytest -q
```

Для запуска в контейнере:

```bash
docker build -t pulse109-ml ./ml-service
docker run --rm -p 8000:8000 pulse109-ml
```
