# Controlled Learning Loop

Learning loop — обязательный P0 wedge. Это offline lifecycle с человеческим
решением, а не online self-learning после каждого клика.

## Состояния

```text
COLLECT → TRAINING → EVALUATE → DECISION
                              ├── PROMOTED
                              └── REJECTED
```

- `COLLECT`: production продолжает обслуживать обращения; feedback только
  накапливается.
- `TRAINING`: валидный feedback замораживается в candidate dataset и запускает
  offline job.
- `EVALUATE`: candidate работает shadow рядом с production на свежих данных;
  production остаётся serving.
- `DECISION`: KPI и critical regressions доступны reviewer.
- `PROMOTED`: human reviewer утвердил candidate; указатель production обновлён
  атомарно.
- `REJECTED`: candidate отклонён или не набрал evidence; production не меняется.

## Минимальные данные

`learning_cycles`:

```text
cycle_id
state
collect_started_at, collect_ends_at
evaluation_started_at, evaluation_ends_at
production_model_version
candidate_dataset_version
candidate_model_version
min_feedback_count
promotion_policy_version
```

`learning_feedback`:

```text
cycle_id, ticket_id
production_prediction
operator_confirmed_decision
accepted_or_corrected
validation_status
feedback_created_at
```

Решение оператора попадает в `learning_feedback` только при открытом
`COLLECT` cycle. Core закрепляет за cycle версию модели из первой production
prediction и не смешивает feedback разных версий. Если открытого подходящего
cycle нет, операторское решение остаётся в PostgreSQL, но не включается в
обучающую выборку. Endpoint `/api/v1/learning/{cycle_id}/feedback` сохраняет
свободные заметки со статусом `UNVERIFIED`; они не увеличивают счётчик
структурированных решений и сами по себе не запускают training job. Закрытие
cycle и постановка job выполняются в одной транзакции после проверки этого
счётчика. PII, dataset lineage и исключение frozen evaluation проверяются
позже offline exporter и candidate builder до фактического обучения.

`candidate_evaluations`:

```text
cycle_id, offline_metrics, shadow_metrics,
critical_regressions, sample_size, decision, created_at
```

Каждая запись также имеет model/dataset/evaluation version и audit actor там,
где действие совершает человек.

## Гейты

1. **Collect close**: период завершён; feedback проходит schema, duplicate,
   PII и label validation.
2. **Minimum feedback**: если записей меньше `min_feedback_count`, цикл
   завершается `INSUFFICIENT_FEEDBACK`; candidate не создаётся.
3. **Dataset freeze**: candidate dataset получает version и checksum. Frozen
   test/evaluation set исключается из train.
4. **Training**: ML worker создаёт immutable candidate artifact и manifest.
5. **Evaluation**: сохраняются offline macro-F1/per-class F1, shadow agreement,
   correction-rate delta, sample size и critical regressions.
6. **Promotion policy**: thresholds фиксируются в `promotion_policy_version`
   до просмотра candidate metrics.
7. **Human decision**: только `ML_REVIEWER`/`ADMIN` переводит candidate в
   `PROMOTED` или `REJECTED`.

Пока реальный trainer не подключён, обычный ML runtime возвращает
`TRAINER_NOT_CONFIGURED` и не создаёт candidate. Тестовый fake trainer включается
только через `PULSE_TEST_FAKE_TRAINER=true` для проверки state machine; его
результат не является обученной моделью или валидной ML-метрикой. Само наличие
candidate artifact больше не создаёт фиктивный `READY_TO_REVIEW`. Promotion
требует сохранённые offline и shadow reports на тех же версиях моделей,
достаточный sample size, отсутствие критичных регрессий и совпадение checksums;
fake candidate не проходит этот gate.

В новой установке production pointer для встроенного rule-based classifier
совпадает с версией, которую сообщает ML `/readyz`. Этот pointer не обозначает
обученный artifact. Core `/readyz` проверяет совпадение версии в PostgreSQL и
ML runtime; после promotion нужно отдельно запустить serving утверждённой
версии, иначе readiness сообщает о расхождении.

Операторское исправление никогда не вызывает serving replacement или
автоматический retraining.

## PostgreSQL-backed jobs

P0 использует `background_jobs`:

```text
id, job_type, payload, state, attempt,
created_at, started_at, finished_at, error
```

Типы: `TRAIN_CLASSIFIER`, `BUILD_EMBEDDINGS`, `RUN_MODEL_EVALUATION`,
`BUILD_FORECAST`, `REINDEX_QDRANT`, `GENERATE_REPORT`. Worker получает задачу
в транзакции через `FOR UPDATE SKIP LOCKED`, ставит lease/attempt и явно
сохраняет terminal error. Redis/Kafka не требуются.

## Invariants

- `AI prediction` и `operator_confirmed_decision` хранятся раздельно.
- Один цикл имеет одну production baseline и не меняет её training job.
- Candidate не становится production без audit event и человеческого approval.
- Rejected candidate не откатывает production, потому что он не был production.
- Все переходы state machine валидируются Core API; ML service не может
  самостоятельно продвинуть цикл.
- Неуспешный job виден пользователю как ошибка/insufficient evidence, а не
  исчезает из очереди.

## Demo сценарий

Demo может закрывать collect явной кнопкой `Close cycle / Train candidate`.
Это тот же pipeline, что и автоматический переход по окончании периода:

```text
demo operator corrections
  → close COLLECT
  → version dataset
  → train candidate
  → evaluate shadow
  → reviewer: PROMOTE или REJECT
```

Synthetic fixtures и demo metrics маркируются отдельно и не интерпретируются
как качество модели на полном dataset всех 20 регионов.
