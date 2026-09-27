# Candidate dataset из learning feedback

[`build_feedback_candidate.py`](../scripts/build_feedback_candidate.py) принимает
JSONL-экспорт одного learning cycle. Экспорт должен быть сформирован после
проверки в Core: PostgreSQL остаётся source of truth, а `original_text` в
экспорте уже должен быть минимизирован по PII. Скрипт повторно проверяет
форматы PII и не записывает отвергнутый текст в отчёт.

Каждая строка `learning-feedback-export.v1` содержит:

```text
contract_version, feedback_id, cycle_id, ticket_id, split_group,
source_dataset_version, is_synthetic, original_text, language,
production_model_version,
production_prediction: {topic_id, confidence},
operator_confirmed_decision: {decision_id, action, topic_id},
accepted_or_corrected, feedback_created_at, validation_status
```

`split_group` должен обозначать проверенную группу одного инцидента;
экспортёр обязан получать её из подтверждённой связи, а не из похожести
текстов. `source_dataset_version` и `is_synthetic` должны происходить из
проверенной dataset lineage. Свободные комментарии, имена, адреса, контакты
и вложения в этот контракт не входят. `production_prediction` и
`operator_confirmed_decision` хранятся отдельно. Обучающая метка берётся
**только** из `operator_confirmed_decision.topic_id`.

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

Текущий Core ставит `TRAIN_CLASSIFIER` с `samples: []`; generic feedback
может содержать пустой `production_prediction` и решение без topic ID.
Такие строки этот контракт отвергает. Нужен отдельный проверенный экспорт
из Core и подключение offline trainer; без них реальный candidate cycle не
считается завершённым. Проверка только ID/group/точного текста не может
доказать отсутствие семантически совпадающих инцидентов, поэтому upstream
review `split_group` остаётся обязательным.
