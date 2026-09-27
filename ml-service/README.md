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
- `POST /internal/v1/training[/<model_type>]` — возвращает
  `TRAINER_NOT_CONFIGURED`: обычный feedback job обучается отдельным offline
  worker только при наличии проверенных входов; тестовый адаптер доступен
  только при `PULSE_TEST_FAKE_TRAINER=true`;
- `POST /internal/v1/evaluation[/<model_type>]` — classifier/forecast/anomaly
  baseline metrics;
- `GET /internal/v1/models` и `/internal/v1/models/<model_type>` — immutable
  version manifest.

`artifacts/manifest.json` описывает baseline. При включении XLM-R сервис
загружает локальный `manifest.json` и веса, сверяет labels и SHA-256, а в
`/healthz`, `/readyz`, `/models` и ответах классификации показывает версию
обученного артефакта. Если артефакт указан, но повреждён, сервис не стартует.

## Обученная модель для демо

Корпус генерируется отдельно; файлы и веса остаются локальными и не попадают
в Git. `classifier_v1` добавлял неподтверждённое время к части текстов;
для локального synthetic demo используйте `classifier_v2` (RU/KZ) или
`classifier_v3` (RU/KZ/MIXED). Для установленной RTX 3060:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers==4.57.6 -r ml-service/requirements.txt
.venv/bin/python scripts/generate_synthetic_classifier.py
.venv/bin/python ml-service/train_classifier.py \
  --demo-data-dir data/sdg/generated/classifier_v2 \
  --output-dir ml-service/artifacts/classifier-synthetic-v2
```

Указывайте новый `--output-dir`: trainer не перезаписывает существующий artifact.
Скрипт использует 4 480 train, 640 validation и 1 280 test примеров.
Сценарии между выборками не пересекаются. Артефакт и метрики сохраняются в
`ml-service/artifacts/classifier-synthetic-v2/`. `macro_f1` на test измеряет
только обобщение на новые **синтетические** сценарии; это не оценка качества
на реальных обращениях 109. Температура вероятностей подбирается на
validation, но калибровка на реальных данных отсутствует. Поэтому все
подсказки обученного кандидата требуют проверки оператора (`needs_review=true`).

Demo trainer также принимает локальный `data/sdg/generated/classifier_v3/` через
`--demo-data-dir` и сохраняет MIXED slice в метриках и manifest. Этот набор
остаётся `PENDING` и не подходит для reviewed training.

Локальный API с обученной моделью:

```bash
PULSE_CLASSIFIER_MODEL_DIR=ml-service/artifacts/classifier-synthetic-v2 \
  .venv/bin/uvicorn app.main:app --app-dir ml-service --host 127.0.0.1 --port 8000
```

`docker-compose.ml.yml` пока монтирует исторический артефакт `classifier-synthetic-v1`.
Он не проверяет корпус `classifier_v2`; переключение Docker demo на новый
артефакт относится к integration-task после его обучения и проверки. Без
override основной compose использует лёгкий baseline. Offline-обучение не
меняет операторские решения и не включает online retraining. Feedback jobs в
worker настраиваются отдельно, как описано в
[learning-loop.md](../docs/learning-loop.md).

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
