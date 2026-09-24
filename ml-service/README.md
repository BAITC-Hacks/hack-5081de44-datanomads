# Pulse 109 ML Service

Изолированный FastAPI runtime для внутреннего ML-контура Pulse 109. Сервис
не содержит авторизации, бизнес-состояния, SQL или операторского workflow —
это ответственность Rust Core API.

## Что входит

- `/healthz` и `/readyz` для liveness/readiness;
- `POST /internal/v1/classify` — детерминированная RU/KZ demo-классификация по
  17 стабильным labels (16 тем и `other`), confidence state и alternatives;
- `POST /internal/v1/embed` — воспроизводимый локальный hashed embedding без
  загрузки внешней модели;
- `POST /internal/v1/forecast` — StatsForecast `SeasonalNaive` baseline и
  rolling-origin backtest (MAE/RMSE/WAPE/sMAPE, число окон и наблюдений);
  история короче сезона помечается `INSUFFICIENT_HISTORY`;
  горизонты до 366 точек (30/60/90 поддерживаются параметром `horizon`);
- `POST /internal/v1/anomaly` — rolling median/MAD anomaly detector;
- `POST /internal/v1/training[/<model_type>]` — возвращает
  `TRAINER_NOT_CONFIGURED` до подключения реального offline trainer; тестовый
  адаптер доступен только при `PULSE_TEST_FAKE_TRAINER=true`;
- `POST /internal/v1/evaluation[/<model_type>]` — classifier/forecast/anomaly
  baseline metrics;
- `GET /internal/v1/models` и `/internal/v1/models/<model_type>` — immutable
  version manifest.

Это честный deterministic demo baseline, а не утверждение о качестве на
реальном dataset 109. `artifacts/manifest.json` явно содержит
`dataset_version`, `model_version`, `metrics`, labels и checksum-поле. Реальные
fine-tuned artifacts могут быть подключены через тот же контракт после
подготовки versioned dataset.

## Локальный запуск

```bash
cd ml-service
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Документация OpenAPI доступна на `http://localhost:8000/docs`.

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
