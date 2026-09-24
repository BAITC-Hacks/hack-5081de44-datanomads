# Реестр моделей и артефактов

## Правила

Каждая обученная модель получает immutable `model_version`. Имя версии не
переиспользуется, а production promotion меняет только указатель на уже
проверенный артефакт. Serving не читает «последний файл» из каталога.

Пример layout:

```text
artifacts/
├── classifier/
│   └── classifier-2026-09-21-001/
│       ├── model/
│       ├── tokenizer/
│       ├── manifest.yaml
│       ├── metrics.json
│       ├── training-config.yaml
│       └── SHA256SUMS
├── embedder/
└── forecast/
```

Baseline manifest поставляется в image по `/app/artifacts/manifest.json`.
Отдельный volume `/app/trained-artifacts` монтируется в `ml-service` и
`ml-worker` для будущих immutable versions; обновление image не перекрывается
старым содержимым volume. Артефакты не загружаются из непроверенного URL во
время inference.

## Manifest

Минимальный контракт `manifest.yaml`:

```yaml
model_version: classifier-2026-09-21-001
model_family: classifier
base_model: FacebookAI/xlm-roberta-base
dataset_version: dataset-2026-09-21-001
created_at: 2026-09-21T12:00:00Z
status: candidate # candidate | production | rejected | archived
languages: [ru, kz]
labels: [water_supply, outdoor_lighting, roads, other]
training_config: training-config.yaml
metrics: metrics.json
artifact_checksum: sha256:<hex>
evaluation_version: evaluation-2026-09-21-001
```

Фактические labels и base model отражают конкретный benchmark; пример выше —
контракт, а не заявление о достигнутом качестве. Для embedding дополнительно
фиксируются dimension, distance и preprocessing. Для forecast — horizon,
frequency и backtest window.

## Метаданные PostgreSQL

`model_versions` хранит минимум:

```text
model_version, model_family, base_model, dataset_version,
artifact_uri, artifact_checksum, status, created_at,
promoted_at, promoted_by, rejected_at, rejected_by
```

`model_evaluations` хранит:

```text
evaluation_id, model_version, evaluation_version,
split_version, metrics_json, critical_regressions,
sample_size, created_at, evaluator
```

Production pointer и audit trail обновляются одной транзакцией. Rejected
candidate остаётся доступным для аудита, но никогда не используется serving.

## Жизненный цикл

```text
TRAINED → CANDIDATE → EVALUATED
                    ├── PROMOTED → PRODUCTION
                    └── REJECTED
```

Promotion требует:

1. checksum и manifest успешно проверены;
2. dataset/evaluation split имеют immutable versions;
3. offline metrics и shadow metrics сохранены;
4. критические regressions просмотрены;
5. human reviewer с `ML_REVIEWER` или `ADMIN` подтвердил решение.

Candidate не заменяет production автоматически по факту окончания training.
Если evidence недостаточно, результат — `REJECTED` или `INSUFFICIENT_EVIDENCE`,
а текущая production-модель остаётся активной.

## Оценка

Classifier: macro-F1, per-class F1, RU/KZ разрезы, confidence calibration и
needs-review rate. Embedding: Recall@K/MRR на approved duplicate/repeat/similar
gold set. Forecast: MAE/RMSE/MAPE и покрытие интервала по rolling backtest.
Candidate сравнивается с baseline и production на одном immutable split.
