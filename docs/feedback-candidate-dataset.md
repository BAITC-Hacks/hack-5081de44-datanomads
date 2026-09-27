# Candidate dataset из learning feedback

[`build_feedback_candidate.py`](../scripts/build_feedback_candidate.py) принимает
JSONL-экспорт одного learning cycle. Экспорт должен быть сформирован после
проверки операторского решения и review текста: PostgreSQL остаётся source
of truth, а `original_text` в экспорте уже должен быть минимизирован по PII.
Скрипт повторно проверяет
известные форматы PII и не записывает отвергнутый текст в отчёт.

Каждая строка `learning-feedback-export.v1` содержит:

```text
contract_version, feedback_id, cycle_id, ticket_id, split_group,
source_dataset_version, is_synthetic, original_text, language,
production_model_version,
production_prediction: {topic_id, confidence},
operator_confirmed_decision: {decision_id, action, topic_id},
accepted_or_corrected, feedback_created_at, validation_status
```

Для обращений, созданных через Core API, используется
`learning-feedback-export.v2`: вместо `source_dataset_version` он содержит
`source_origin_kind="RUNTIME_API"`. Это не импортный dataset. В candidate
manifest такие строки перечислены отдельно в `runtime_ticket_ids`;
`source_dataset_versions` содержит только настоящие импортные версии. Пакет
может содержать v1 и v2 одновременно и фиксирует это в
`source_contract_version="learning-feedback-export.mixed.v1"`.

`split_group` должен обозначать проверенную группу одного инцидента;
экспортёр обязан получать её из подтверждённой связи, а не из похожести
текстов. Для импортного v1 `source_dataset_version` и `is_synthetic` должны
происходить из проверенной dataset lineage. Свободные комментарии, имена,
адреса, контакты и вложения в этот контракт не входят. `production_prediction` и
`operator_confirmed_decision` хранятся отдельно. Обучающая метка берётся
**только** из `operator_confirmed_decision.topic_id`.

Для локального экспорта из Core создайте вне Git JSONL с подтверждёнными
связями `feedback-review-link.v1`. Каждая строка содержит `db_ticket_id`,
стабильный `ticket_id`, `split_group`, `source_dataset_version`,
`text_review_sha256` (SHA-256 **точного текста из Core** в формате
`sha256:<64 lowercase hex>`), `review_status="APPROVED"`, `reviewer_id` и
`reviewed_at` с часовым поясом. Рецензент должен проверить PII и группу
инцидента; один regex сканер этого не доказывает. Связи без такого review
не экспортируются. `ticket_id` следует согласовать с идентификаторами
замороженного evaluation package, чтобы проверка пересечений работала.

Для API-created ticket без `dataset_ticket_links` нужен отдельный
`feedback-review-link.v2`. Обязательные поля: `db_ticket_id`, `ticket_id`,
`split_group`, `source_kind="RUNTIME_API"`, `source_system="api"`,
`external_ticket_id` (точный server-generated `api-<nanoseconds>`),
`is_synthetic` (подтверждённое рецензентом происхождение),
`text_review_sha256`, `review_status="APPROVED"`, `reviewer_id` и
`reviewed_at`. Рецензент утверждает и текст, и `is_synthetic`: созданный через
API текст не считается реальным автоматически. Экспортёр сверяет эти поля с
PostgreSQL, точный SHA-256 текста, отсутствие dataset links, близость времени
ID/создания (не более 5 минут) и запись `CREATE_TICKET` в `audit_log`. Запись
без audit event или review исключается. API tickets с заданным пользователем
`source` вместо стандартного `api` пока исключаются, поскольку их
происхождение не подтверждает этот узкий контракт. Импортный v1 путь и его
checksum-проверки сохраняются.

```bash
DATABASE_URL='postgresql://...' PYTHONPATH=.:ml-service \
  .venv/bin/python scripts/export_learning_feedback.py \
  --cycle-id cycle_1 --production-model-version classifier_production_v1 \
  --review-links data/processed/feedback/review-links.jsonl \
  --output data/processed/feedback/validated-feedback.jsonl
```

Скрипт читает PostgreSQL в read-only transaction. Он сверяет review hash
текста, импортную dataset lineage с checksums либо подтверждённое API
происхождение, `is_synthetic`, связанное решение оператора и сохранённую
production prediction. Generic feedback без
topic ID, API tickets без утверждённого v2 review и строки с PII отклоняются с
агрегатным кодом причины. Выходной JSONL создаётся эксклюзивно с правами
`0600`; при нуле допустимых строк файл не создаётся. В stdout выводятся
только счётчики. Review links и экспорт содержат чувствительные данные
или ссылки на них и не должны попадать в Git.

```bash
PYTHONPATH=ml-service .venv/bin/python scripts/build_feedback_candidate.py \
  --feedback /path/to/validated-feedback.jsonl \
  --frozen-from data/processed/reviewed-v1 \
  --cycle-id cycle_1 \
  --production-model-version classifier_production_v1 \
  --dataset-version candidate_v1 \
  --min-feedback-count 30
```

Builder проверяет полный checksum frozen package, затем исключает frozen
IDs, groups и точные нормализованные тексты classifier/retrieval test. При
повторном feedback ID сохраняется одна идентичная строка; конфликтующие
версии ID отклоняются целиком. Для одного ticket выбирается последнее
feedback по времени; остальные учитываются как `SUPERSEDED_TICKET_FEEDBACK`.
Каждая иная отброшенная строка получает код причины в агрегатных счётчиках.

При достаточном числе принятых строк создаётся один immutable каталог
`data/processed/feedback/<dataset_version>/` с `train.jsonl` и
`manifest.json`: checksum исходного экспорта, frozen evaluation, source
feedback IDs, состав synthetic/real, причины отказа и content checksum.
Повторная сборка в другом каталоге даёт те же байты. `COMPLETED` здесь
означает только сборку candidate dataset, а не обучение или promotion.
При `INSUFFICIENT_FEEDBACK` каталог не создаётся.

Перед обучением `load_verified_candidate(package, frozen_package)` повторно
проверяет checksum и структуру immutable пакета, соответствие сохранённому
frozen evaluation, число и происхождение строк, подтверждённые метки, PII и
пересечения с frozen IDs/groups/точными текстами. Подмена строк с последующим
пересчётом checksums тоже отклоняется при семантическом нарушении. Несогласованная
пара `accepted_or_corrected` / `operator_confirmed_decision.action` получает
отдельную причину `DECISION_ACTION_MISMATCH` ещё при сборке.

Локальный offline trainer принимает только такой проверенный пакет и ровно ту
версию production artifact, которая записана в manifest:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_feedback_candidate.py \
  --dataset data/processed/feedback/candidate_v1 \
  --frozen-from data/processed/reviewed-v1 \
  --production-model /path/to/production-classifier \
  --candidate-model-version classifier_candidate_v1 \
  --output ml-service/artifacts/classifier-feedback-candidate-v1
```

Trainer сохраняет base version/checksum, dataset checksum, frozen evaluation
version/checksum, seed, config и SHA-256 весов. Новый artifact проходит sanity
inference и остаётся `CANDIDATE` с отключённым `CONFIDENT`; после fine-tuning
калибровка исходной модели не считается доказанной. Ошибки CLI возвращаются как
`FAILED` с кодом стадии без текста обращения. При недостаточном числе валидных
строк builder возвращает `INSUFFICIENT_FEEDBACK` и не создаёт пакет, поэтому
trainer не запускается. Реальное качество оценивается отдельно на frozen test
и свежих операторских решениях.

Core ставит `TRAIN_CLASSIFIER` с `samples: []`, но настроенный offline worker
читает структурированный feedback из PostgreSQL по `cycle_id`, а не из этого
поля job payload. Generic feedback с пустым `production_prediction` или
решением без topic ID отвергается экспортёром. Worker затем собирает
candidate dataset, обучает модель и сохраняет offline comparison report;
этот путь требует `docker-compose.training.yml`, review links, frozen dataset,
production artifact и заранее утверждённый critical policy. Без них job
получает явный код ошибки. Два региональных CSV без текста обращения не дают
необходимых входов для реального обучения.

Файловое возобновление offline job не восстанавливает состояние PostgreSQL
после падения worker. Необходимые переходы `background_jobs` и
`learning_cycles` перечислены в
[`learning_worker_retry_handoff.json`](../data/contracts/learning_worker_retry_handoff.json).

Парный shadow export и сохранение его итогового отчёта выполняются отдельно
после evaluation window командами, описанными в
[`ml-service/training/README.md`](../ml-service/training/README.md). До
достаточного свежего real feedback promotion остаётся недоступным.
Проверка только ID/group/точного текста не доказывает отсутствие семантически
совпадающих инцидентов, поэтому upstream review `split_group` обязателен.
