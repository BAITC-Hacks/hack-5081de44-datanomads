# Документация Pulse 109

Начните с [главного README](../README.md): там схема сервисов, фактический
стек и быстрый запуск. Этот каталог содержит подробные контракты, инструкции
и evidence. Документы с датой в имени фиксируют отдельный прогон или
выступление; для текущего запуска используйте инструкции ниже.

## Запуск и сдача

- [Deployment](deployment.md) — локальный Docker demo, запуск на выделенном
  сервере, ограничения внешнего доступа, обновление и восстановление.
- [Testing](testing.md) — уровни проверок, smoke, API contract и E2E.
- [Сценарий Demo Day](demo-brief-2026-09-29.md) — тайминг ролика и маршрут
  по синтетическому набору версии `demo-2026-09-29.v1`.
- [Отчёт о проверке 28 сентября](test-report-2026-09-28.md) и
  [acceptance evidence](../ACCEPTANCE_EVIDENCE.md) — исторические результаты
  конкретных прогонов, а не гарантия для нового сервера.

## Устройство системы и границы

- [Архитектура](architecture.md) — сервисы и потоки данных.
- [Контракты API](openapi/core.openapi.yaml) — Core OpenAPI;
  [ML OpenAPI](openapi/ml.openapi.yaml) — внутренний ML API.
- [UnifiedTicket и импорт](data-contract.md),
  [Data/ML handoff](contracts.md) и [статус handoff](data-ml-handoff-status.md).
- [Privacy](privacy.md) — PII, роли, журнал и граница внешнего LLM.
- [Model registry](model-registry.md) и
  [controlled learning loop](learning-loop.md) — версии, evidence и human
  promotion.

## Исследования и ограничения данных

Отчёты в этом разделе показывают происхождение решений и открытые вопросы.
Они не подтверждают качество модели на полном наборе обращений 109.

- [Выгрузка ВКО](vko-109-data-audit.md),
  [таксономия Алматы](almaty-2025-taxonomy-review.md),
  [доступность полей](customer-field-availability.md).
- [Routing и priority](routing-priority-evidence.md),
  [качество retrieval](retrieval-gold-pilot.md),
  [источники сигналов](spike-evidence.md),
  [повторные обращения](recurrence-monitoring.md).
- [Прогноз](rolling-forecast.md), [дефицит ресурсов](capacity-gap.md),
  [контроль сигналов](signal-control.md),
  [проверка результата](outcome-verification.md).
- [Feedback dataset](feedback-candidate-dataset.md),
  [memory подтверждённых решений](verified-resolution-memory.md),
  [SDG pilot](nemo-sdg-pilot.md).
- [Адаптер STT](stt-adapter.md) и
  [генеративный ответ](generative-response-draft.md) — описанные зависимости
  и гейты, а не включённые по умолчанию функции demo.
