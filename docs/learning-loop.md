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
результат не является обученной моделью или валидной ML-метрикой.

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

## COLLECT window and configuration

Core creates one classifier cycle at a time and snapshots its start, end,
production model version, minimum feedback count, promotion policy version, and
demo manual-close setting. The PostgreSQL advisory lock serializes cycle
creation; a second active cycle is rejected. `PROMOTED`, `REJECTED`, and
`INSUFFICIENT_FEEDBACK` are terminal for this policy.

The Core API reads these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PULSE_LEARNING_CYCLE_DURATION_HOURS` | `168` | COLLECT window, from 1 hour to 10 years |
| `PULSE_LEARNING_MIN_FEEDBACK_COUNT` | `1` | Valid feedback rows required before training is queued |
| `PULSE_LEARNING_MANUAL_CLOSE` | enabled for `demo`/`test`/`unit`, disabled otherwise | Allows a reviewer to close before the end time |
| `PULSE_LEARNING_PROMOTION_POLICY_VERSION` | `policy-v1` | Policy version frozen on cycle creation |

The existing PostgreSQL worker checks expired COLLECT cycles on each poll. It
marks cycles below the threshold `INSUFFICIENT_FEEDBACK` without creating a
candidate or job; otherwise it atomically moves the cycle to `TRAINING` and
queues the existing classifier job. A feedback request tied to a closed,
expired, or non-COLLECT cycle receives `409`. Operator decisions remain saved
outside the learning dataset when there is no active COLLECT cycle; the Core
response reports `NO_ACTIVE_COLLECT_CYCLE` in that case. No retraining job is
created per operator click.

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
