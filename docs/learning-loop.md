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
- `TRAINING`: validated feedback проходит через Data/ML candidate dataset
  builder, затем checksum-verified dataset обучается отдельным candidate
  trainer; production остаётся serving.
- `EVALUATE`: candidate работает shadow рядом с production на свежих данных;
  production остаётся serving. Временные границы окна сохраняются в
  `evaluation_started_at` и `evaluation_ends_at`; новый ticket получает
  независимые production/candidate predictions с версиями и timestamp в
  `learning_cycle_shadow_predictions`. Ошибка candidate inference фиксируется
  безопасным кодом и не меняет production рекомендацию. Длительность evaluation
  окна равна фактической длительности COLLECT.
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

`learning_cycle_shadow_predictions` хранит evidence по cycle/ticket, обе версии
моделей, обе classification структуры, время inference и ссылку на
`operator_decisions.id`, если решение появилось позже. Запись не дублирует
ticket text и свободную заметку оператора. Получение evaluation не закрывает
окно: переход в `DECISION` происходит по истечении `evaluation_ends_at` или
через ручное закрытие тем же endpoint цикла. Promotion/rejection остаётся
отдельным человеческим решением. Blind A/B выключен
(`blind_ab_enabled = false`); preference signal не собирается.

Цикл связывает feedback, dataset и evaluation version; действие reviewer
сохраняет audit actor.

## Гейты

1. **Collect close**: период завершён; feedback проходит schema, duplicate,
   PII и label validation.
2. **Minimum feedback**: если записей меньше `min_feedback_count`, цикл
   завершается `INSUFFICIENT_FEEDBACK`; candidate не создаётся.
3. **Dataset freeze**: candidate dataset получает version и checksum. Frozen
   test/evaluation set исключается из train.
4. **Training**: ML worker создаёт immutable candidate artifact и manifest;
   модель регистрируется как `SHADOW`, production pointer не меняется.
5. **Evaluation**: Core сохраняет отдельные production/candidate predictions
   для fresh tickets и связывает появившиеся operator decisions; offline и
   shadow evidence доступны reviewer.
6. **Promotion policy**: thresholds фиксируются в `promotion_policy_version`
   до просмотра candidate metrics.
7. **Human decision**: только `ML_REVIEWER`/`ADMIN` переводит candidate в
   `PROMOTED` или `REJECTED`.

Постоянный `TRAIN_CLASSIFIER` job обучает отдельный candidate из immutable
dataset artifact. Inline-sample training endpoint остаётся
`TRAINER_NOT_CONFIGURED`, если включён только deterministic baseline. Тестовый
fake trainer допускается лишь в demo/development/test runtime через
`PULSE_TEST_FAKE_TRAINER=true`; его результат не является обученной моделью или
валидной production-метрикой.

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
| `PULSE_LEARNING_EVALUATION_DATASET_VERSION` | unset | Registered dataset version whose ticket IDs are frozen for evaluation exclusion |

The configured evaluation dataset version must already be registered in
`dataset_versions` and have `dataset_ticket_links`. Core copies its ticket IDs
to `learning_cycle_evaluation_tickets` when the cycle is created. A reviewer
may override the version in the cycle creation request.

The PostgreSQL worker checks expired COLLECT cycles on each poll. Below the
feedback threshold it records `INSUFFICIENT_FEEDBACK` without creating a
candidate or job. Otherwise it queues `BUILD_CANDIDATE_DATASET` with validated
feedback IDs and the frozen evaluation version/IDs. The Data/ML builder writes
an ID-only feedback export, resolves normalized ticket text from PostgreSQL,
excludes the frozen evaluation IDs, then returns a candidate dataset version,
artifact URI, and checksums. The ML worker registers that lineage and queues
`TRAIN_CLASSIFIER` in one transaction. That job points to the candidate dataset,
production baseline, training config version, and output artifact URI. The ML
package verifies both dataset checksums and writes a candidate model artifact
with a versioned manifest. Training completion stores the model as `CANDIDATE`
and advances the cycle to `EVALUATE`; it does not update the production pointer.
Jobs contain no ticket text or feedback comments. Builder failures mark the
cycle `DATASET_BUILD_FAILED`; trainer failures are stored as sanitized job and
cycle error codes.

The `PULSE_TEST_FAKE_TRAINER` adapter only runs in `demo`, `development`, `test`,
or `unit` runtime modes. Synthetic candidate datasets cannot be promoted in
production runtime.

If no frozen evaluation dataset is configured, the build job fails visibly with
`FROZEN_EVALUATION_SET_NOT_CONFIGURED`; the worker does not invent a holdout.
A feedback request tied to a closed, expired, or non-COLLECT cycle receives
`409`. Operator decisions remain saved outside the learning dataset when there
is no active COLLECT cycle; the Core response reports
`NO_ACTIVE_COLLECT_CYCLE` in that case. No retraining job is created per
operator click.

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
