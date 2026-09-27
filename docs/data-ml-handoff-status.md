# Data/ML handoff: состояние на 2026-09-28

Этот файл сверяет раздел 15 плана `PULSE109_DATA_ML_EXECUTION_PLAN.md` с фактическими артефактами рабочего дерева. `Готово` означает, что требуемое доказательство существует; `Частично` — доказательство только для части заявленной области; `Код` — реализованный путь без нужного результата; `Ожидает` — отсутствующий вход или проверку. Синтетические demo-метрики не считаются качеством на обращениях заказчика.

## Data

| Требование | Статус | Доказательство или следующий вход |
| --- | --- | --- |
| Synthetic raw fallback для 7 systems | Готово | `python3 scripts/generate_synthetic_sources.py --check`: 7 synthetic fixtures; [генератор](../scripts/generate_synthetic_sources.py), локальный `data/synthetic_raw/v1/manifest.json`. Профили помечены `SYNTHETIC_TEST_ONLY`. |
| Verified real source profiles для полученных источников | Ожидает | Два региональных CSV не сопоставлены с семью `source_system`; [профиль полей](customer-field-availability.md). Остальные пять выгрузок не получены. |
| Data Quality report | Частично | [Аудиты ВКО](../data/reports/vko_109_source_audit.json) и [Алматинской области](../data/reports/almaty_109_source_audit.json) совпадают по SHA-256 с локальными файлами. Отчёта по проверенному нормализованному real corpus нет. |
| PII-safe normalized corpus | Ожидает | [Манифест demo](../data/manifests/demo-2026-09-21.json) описывает только synthetic fixture; `data/processed/` не содержит real corpus. Исходного текста обращений нет в двух CSV. |
| Final taxonomy + labeling guide | Ожидает | [Labeling guide](../data/labeling-guide.md) и [очередь taxonomy review](../data/catalogs/almaty_2025_taxonomy_review.json) есть; предложения не утверждены человеком. |
| Classifier train/validation/test manifests | Ожидает | Локальный `data/sdg/generated/classifier_v2/manifest.json` относится к 6 400 synthetic `PENDING` кандидатам. [Builder](../ml-service/training/dataset_builder.py) готов к reviewed records; утверждённого пакета нет. |
| Retrieval train/validation/test manifests | Ожидает | [Synthetic relation seeds](../data/sdg/pilot_relations.jsonl) и локальная очередь `data/reviews/retrieval-pilot.jsonl` на 15 пар готовы; все решения `PENDING`, утверждённого relation package нет. |
| Frozen evaluation versions | Код | [Dataset builder](../ml-service/training/dataset_builder.py) создаёт immutable frozen membership; конкретной reviewed version в `data/processed/` нет. |
| No leakage evidence | Код | Проверки group/text leakage есть в builder и тестах. Evidence для фактического reviewed package появится только после его сборки. |

## Classifier

Локальный `ml-service/artifacts/classifier-synthetic-v1/` содержит веса и manifest для прежнего synthetic demo. Это не reviewed classifier handoff. Новый `classifier_v2` исправляет неподтверждённые временные вставки, но модель на нём ещё не обучалась.

| Требование | Статус | Доказательство или следующий вход |
| --- | --- | --- |
| Fine-tuned artifact | Ожидает | Есть только synthetic demo artifact; нужен обученный кандидат на reviewed dataset. |
| 10+ topics | Частично | Demo artifact имеет 16 labels; утверждённая taxonomy и проверенные метки отсутствуют. |
| RU/KZ | Частично | Demo artifact и synthetic corpus покрывают RU/KZ; нет оценки на проверенных обращениях. |
| Macro/per-class/weighted F1 | Код | [Trainer](../ml-service/train_classifier.py) вычисляет все три; в существующем demo manifest нет weighted F1, reviewed report отсутствует. |
| Confusion matrix | Код | Trainer вычисляет матрицу; сохранённой матрицы для reviewed holdout нет. |
| Calibration | Код | Validation temperature и calibration report реализованы; real/reviewed calibration evidence нет. |
| Confidence thresholds | Код | Selection policy реализована; пороги на reviewed validation не выбраны. |
| RU/KZ slices | Частично | Есть demo slices; reviewed RU/KZ slices отсутствуют. |
| Latency benchmark | Код | [Candidate evaluator](../ml-service/training/classifier_candidate_eval.py) измеряет latency; соответствующего reviewed report нет. |
| Checksum + manifest | Частично | У demo artifact есть checksum/manifest; immutable reviewed [bundle](../ml-service/training/classifier_bundle.py) пока не создан. |

## Retrieval

| Требование | Статус | Доказательство или следующий вход |
| --- | --- | --- |
| Pretrained E5 baseline result | Код | [Evaluator](../ml-service/training/retrieval_baselines.py) есть; сохранённого отчёта на reviewed relations нет. |
| Fine-tuned embedder artifact | Код | [Trainer и verifier](../ml-service/training/embedder_candidate.py) есть; artifact отсутствует. |
| Recall@1/3/5 + MRR | Код | Метрики реализованы в evaluator; результата на reviewed holdout нет. |
| Precision@K/nDCG | Код | Метрики реализованы; результата на reviewed holdout нет. |
| Top-3 manual review | Ожидает | Workflow создаёт `PENDING` очередь; экспертных решений нет. |
| Duplicate precision | Код | [Threshold evaluator](../ml-service/training/duplicate_thresholds.py) есть; reviewed relation evidence нет. |
| Selected duplicate/repeat thresholds | Ожидает | Порог duplicate и эмпирическое repeat window не утверждены; [runtime handoff](../data/contracts/repeat_runtime_handoff.json) сохраняет этот статус. |
| False-positive analysis | Ожидает | Нужны проверенные hard negatives и экспертный разбор ошибок. |
| Checksum + manifest | Код | [Bundle verifier](../ml-service/training/embedder_candidate.py) реализован; реального embedder bundle нет. |

## Controlled Learning

| Требование | Статус | Доказательство или следующий вход |
| --- | --- | --- |
| Feedback dataset builder | Код | [Builder](../ml-service/training/feedback_dataset.py) реализован; versioned reviewed feedback dataset отсутствует. |
| Frozen-eval exclusion | Код | Проверка реализована в builder и тестах; нет фактической frozen version для цикла. |
| Real classifier trainer | Код | [Offline trainer](../ml-service/training/feedback_trainer.py) реализован; нет запуска на достаточном approved feedback. |
| Candidate artifact creation | Код | [Cycle job](../ml-service/training/feedback_job.py) и [bundle](../ml-service/training/feedback_bundle.py) реализованы; кандидат не опубликован. |
| Offline production/candidate evaluator | Код | [Evaluator](../ml-service/training/classifier_candidate_eval.py) реализован; отчёта на одинаковом reviewed slice нет. |
| Shadow metrics evaluator | Код | [Shadow evaluator](../ml-service/training/shadow_eval.py) реализован; нет проверенного runtime feedback window. |
| Critical regression report | Код | Формат и вычисление предусмотрены evaluator; фактического отчёта нет. |
| Machine-readable promotion evidence | Код | [Comparison contract](../ml-service/training/challenger_eval.py) и candidate reports есть как код; готового promotion report нет. Human promotion остаётся отдельным решением. |

## Forecast и Spike

| Требование | Статус | Доказательство или следующий вход |
| --- | --- | --- |
| 30/60/90 rolling backtest package | Частично | [Отчёт ВКО](../data/reports/vko_109_forecast_candidates.json) содержит 21/20/19 окон на реальных агрегатных датах. Это proxy общей нагрузки области, без утверждённых `region × topic` labels; `runtime_eligible=false`. |
| Baseline comparison | Частично | В том же отчёте Prophet сравнивается с weekly seasonal naive на одинаковых окнах; выбора production model нет. |
| Honest insufficient-history behavior | Готово | [Отчёт Алматинской области](../data/reports/almaty_109_forecast_candidates.json): `INSUFFICIENT_HISTORY`, ноль допустимых окон. |
| Spike threshold calibration package | Ожидает | [Exploration ВКО](../data/reports/vko_109_spike_exploration.json) и [Алматинской области](../data/reports/almaty_109_spike_exploration.json) имеют `NO_REVIEWED_INCIDENT_LABELS`. Локально созданы полные очереди на 159 и 40 дат и detector-blind шаблоны на 975 и 259 оценённых дат; всё `PENDING`. Precision/recall и выбранного порога нет. |
| Validation limitations clearly marked | Готово | [Описание spike evidence](spike-evidence.md) и forecast/spike reports явно отмечают proxy scope, пропущенные дни и отсутствие incident ground truth. |

## Входы для закрытия checklist

Нужны подтверждённые профили полученных источников, исходные тексты обращений либо человечески проверенный synthetic corpus, решения по taxonomy/classifier/retrieval и независимый реестр инцидентов. Для Learning Loop дополнительно нужен approved feedback и сопоставимые production/candidate predictions в одном evaluation window. `com_exp` не используется как исходный текст: заказчик подтвердил, что это поле для задачи бесполезно.

Проверено 2026-09-28: `generate_synthetic_sources.py --check` (7 источников), `generate_demo_data.py --check` (160 synthetic строк), SHA-256 обоих локальных CSV против source audit reports. Созданы игнорируемые Git `data/reviews/{vko,almaty}_spike_review_2026-09-28.json` и соответствующие `*_incident_registry_template_2026-09-28.json`; шаблоны не содержат alert scores и не проходят evaluator до review. Также создана локальная очередь `data/reviews/retrieval-pilot.jsonl`: все 15 пар валидны и `PENDING`, экспорт approved-пар отклонён из-за отсутствия человеческих решений. Эти проверки не заменяют human review и не подтверждают качество моделей на реальных обращениях.
