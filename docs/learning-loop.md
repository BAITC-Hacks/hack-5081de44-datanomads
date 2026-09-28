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
- `EVALUATE`: до пяти зарегистрированных candidate versions работают в shadow
  рядом с единственной production-моделью на свежих данных; только production
  влияет на ответ. Временные границы окна сохраняются в `evaluation_started_at`
  и `evaluation_ends_at`; новый ticket получает отдельную production prediction
  и по одной версии candidate prediction с timestamp в
  `learning_cycle_shadow_predictions`. Ошибка inference кандидата фиксируется
  безопасным кодом и не меняет production рекомендацию. Длительность evaluation
  окна равна фактической длительности COLLECT.
- `DECISION`: worker оценивает каждого зарегистрированного candidate на одном
  frozen наборе и его собственных shadow predictions. Reviewer сравнивает версии,
  видит sample sizes, policy thresholds и gates каждой версии, затем указывает
  конкретную версию при promotion.
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

Связь цикла с версиями хранится в `learning_cycle_candidates` (`cycle_id`,
`model_version`, status и actor регистрации). Она не меняет единственный
`model_versions.status = 'PRODUCTION'`. В EVALUATE новая версия должна уже
находиться в registry как `CANDIDATE` или `SHADOW`, иметь checksum артефакта и
registered dataset lineage; reviewer может добавить её к cycle до закрытия
evaluation window только на стадиях COLLECT или TRAINING. Регистрация закрывается
при переходе в EVALUATE, чтобы все candidates получили одинаковые shadow
наблюдения. При закрытии окна Core ставит отдельную job оценки каждой версии. В
одном cycle допускается максимум пять версий, чтобы ограничить inference и
evaluation fan-out.

`learning_feedback`:

```text
cycle_id, ticket_id
production_prediction
operator_confirmed_decision
accepted_or_corrected
validation_status
feedback_created_at
```

`model_evaluations`:

```text
learning_cycle_id, evaluation_payload, offline_metrics, shadow_metrics,
critical_regressions, sample_size, decision, created_at
```

`learning_cycle_shadow_predictions` хранит evidence по cycle/ticket/candidate
version, production и candidate classification структуры, время inference и ссылку на
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
   для fresh tickets и связывает появившиеся operator decisions. После закрытия
   окна worker передаёт ML evaluator frozen holdout и связанные shadow decisions;
   ML считает обе модели на одном holdout и сохраняет полный result с FK на
   `learning_cycles.id`.
6. **Promotion policy**: версия policy фиксируется в cycle до оценки. `policy-v1`
   задаёт immutable thresholds: offline macro-F1 не ниже baseline −0.02, падение
   F1 любого класса не больше 0.05, минимум 30 offline samples, минимум 20
   подтверждённых shadow decisions, рост correction rate не больше +0.05 и
   ноль candidate inference failures. Изменение чисел требует новой версии.
7. **Human decision**: только версия, чьи собственные gates пройдены на том же
   frozen evidence set, может быть выбрана `ML_REVIEWER`/`ADMIN` для promotion.
   При нескольких eligible версиях API требует передать выбранный
   `candidate_model_version`. Цикл связывается с выбранной версией, она получает
   `PRODUCTION`, а прежний champion архивируется в одной транзакции. Reject
   завершает все candidate links этого цикла и не меняет production pointer.

Promotion остаётся одним явным шагом `ML_REVIEWER` или `ADMIN` с обязательным
audit event. Текущая policy не требует независимого второго согласования, поэтому
demo не добавляет многошаговый approval workflow до появления такого
операционного требования.

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

Типы: `TRAIN_CLASSIFIER`, `BUILD_CANDIDATE_DATASET`, `CANDIDATE_EVALUATION`,
`BUILD_EMBEDDINGS`, `RUN_MODEL_EVALUATION`, `BUILD_FORECAST`, `REINDEX_QDRANT`,
`GENERATE_REPORT`. Worker получает задачу
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

## Automatic recurring schedule

The ML worker can start one global classifier cycle automatically when
`PULSE_LEARNING_AUTO_CYCLES_ENABLED=true`. It defaults to disabled. The worker
uses the same COLLECT duration, minimum feedback count, promotion policy version
and frozen evaluation dataset settings as Core; every scheduled start snapshots
the current production model and frozen evaluation ticket IDs and writes an
audit event as `learning-scheduler`.

The schedule starts the first cycle only when a production model and a registered
evaluation dataset with linked tickets are available. Production refuses a
synthetic evaluation dataset. It starts the next cycle only after
`PROMOTED`, `REJECTED` or `INSUFFICIENT_FEEDBACK`; it waits while a cycle is
active or in `DECISION`, and it stops after dataset/training failures for an
operator to review. The scheduler and manual Core endpoint share the same
PostgreSQL advisory lock, so two workers or a manager request cannot create
overlapping cycles. Evaluation remains a separate persisted `EVALUATE` window
that begins after candidate training and ends before reviewer decision.

The schedule is global because this classifier currently has one serving
production version. No region/topic cycle configuration is exposed until there
is an operational reason to train independent models for those slices.

## Drift-triggered candidate cycles

The Core API accepts `drift-evidence.v1` only from the trusted `ML_SERVICE`
identity. Evidence is aggregate-only and identifies the model, detector,
metric, non-overlapping baseline/observed windows, score, threshold and sample
counts. Core rejects evidence that does not exceed its supplied threshold or
minimum count, repeated IDs with different payloads, and synthetic evidence
outside explicit demo/test/unit modes. The complete evidence payload is stored
with the trigger; audit metadata contains only allow-listed model/detector/metric
and score fields.

ML reviewers and admins can dismiss a pending trigger or open a COLLECT cycle.
Opening a cycle requires that its evidence model is still the production
champion, no other cycle is active, the configured frozen evaluation dataset is
registered and linked, and production evidence is not synthetic. Trigger review
and cycle creation share the learning-cycle transaction lock and are committed
with their audit events. Only the currently implemented `policy-v1` promotion
policy can open a cycle. This action only opens a candidate cycle; evaluation
and explicit human promotion remain required to change the production pointer.

The checked-in drift evidence example is synthetic and exists only to validate
the shared contract. The current deterministic ML runtime does not produce
calibrated drift scores; a Data/ML detector must publish real evidence through
the trusted Core boundary before this workflow can trigger on live drift.

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
- Cycle candidate проходит только через `/api/v1/learning/candidate/promote`;
  общий model promotion endpoint отклоняет связанные с cycle версии.
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
