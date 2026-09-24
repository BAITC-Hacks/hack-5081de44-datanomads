# Контракт данных

## Принцип

PostgreSQL — source of truth для Pulse state. Внешняя система 109 остаётся
официальным источником самого обращения и его официального статуса. Qdrant —
индекс производного embedding, а не хранилище обращений.

Raw-файлы не коммитятся в Git. Каждый источник проходит собственный importer:

```text
source file → source-specific parser → schema validation → normalization
            → PII minimization → UnifiedTicket → PostgreSQL
```

Ошибочные строки не удаляются молча: они помещаются в `data/quarantine/` с
причиной и ссылкой на исходный файл/строку.
В PostgreSQL `quarantine_rows` хранит код причины, разрешённое имя поля и
маркер redaction. Переданный через API произвольный текст ошибки и содержимое
строки туда не копируются. PII-safe snapshot доступен в локальном JSONL
importer-а для разбора источника.

## UnifiedTicket

Минимальные поля (имя в API/хранилище — `snake_case`):

| Поле | Тип | Обязательность | Правило |
| --- | --- | --- | --- |
| `external_ticket_id` | string | да | стабильный id в исходной системе |
| `source_system` | enum/string | да | например `IKOMEK109`, `AIKEY`, `OPEN_CITY` |
| `region_id` | string | да | канонический код региона; demo покрывает 20 регионов |
| `created_at` | RFC3339 timestamp | да | исходное время, с timezone |
| `original_text` | string | да | исходный текст; доступ контролируется RBAC |
| `language` | `ru` / `kz` / `unknown` | да | значение модели/источника, не выдумывается |
| `topic_raw` | string/null | нет | исходное направление без потери значения |
| `topic_id` | string/null | нет | каноническая taxonomy после mapping |
| `service_raw` | string/null | нет | исходное наименование исполнителя |
| `service_id` | string/null | нет | канонический service id |
| `priority` | enum/null | нет | исходное или подтверждённое значение |
| `status` | string/null | нет | статус источника, не статус Pulse |

Подключаемые только при наличии в источнике поля:

```text
district, address, coordinates, object, channel,
closed_at, deadline_at, resolution_text, official_response,
attachments, assignment_history
```

Служебные производные поля не заменяют исходные:

```text
pulse_prediction
operator_confirmed_decision
model_versions
needs_review
embedding_ref
duplicate_feedback
repeat_feedback
created_in_pulse_at
updated_in_pulse_at
```

Пример безопасного envelope:

```json
{
  "external_ticket_id": "demo-001",
  "source_system": "DEMO",
  "region_id": "KZ-01",
  "created_at": "2026-09-21T10:00:00Z",
  "original_text": "Не работает освещение во дворе",
  "language": "ru",
  "topic_raw": "Наружное освещение",
  "topic_id": "outdoor_lighting",
  "service_raw": null,
  "service_id": null,
  "priority": null,
  "status": "open"
}
```

## Импорт и quality gate

Для каждого набора фиксируются `source_system`, схема, checksum, период,
количество строк и importer version. До обучения строится отчёт, включающий:

- покрытие регионов, источников и временного диапазона;
- заполненность полей, уникальность id и дубликаты строк;
- невалидные даты, сдвинутые колонки и повреждённые CSV-строки;
- распределение языков, тем, статусов и priority;
- наличие текста, исполнителя, решения и времени закрытия;
- PII scan и возможность собрать duplicate/repeat gold set.

`dataset_versions.content_sha256` фиксирует нормализованное содержимое импорта.
Повторная отправка той же версии с другим содержимым или manifest отклоняется
с `409`; `dataset_ticket_links` сохраняет связь версии с обращениями даже при
идемпотентном повторном импорте. Для версий, созданных до миграции 008, checksum
заполняется при первом повторном импорте с совпадающим manifest.
Новая версия может ссылаться на уже импортированное обращение, если его
канонические поля не изменились. Изменённая запись с тем же source ID получает
`409`, чтобы новая версия не ссылалась молча на старое содержимое. Синхронизация
изменений официального обращения требует отдельного контракта источника.

Причины quarantine стандартизируются: `BAD_CSV_STRUCTURE`, `INVALID_DATE`,
`MISSING_REQUIRED_FIELD`, `UNKNOWN_SCHEMA`, `PII_REVIEW`. Quarantine не
попадает в train/evaluation без явного решения data steward.

## Taxonomy

Финальная taxonomy появляется только после аудита всех доступных источников.
Начальные кандидаты верхнего уровня: `water_supply`, `sewerage`,
`electricity`, `outdoor_lighting`, `heating`, `gas_supply`, `roads`,
`public_transport`, `waste`, `landscaping`, `buildings`, `healthcare`,
`veterinary`, `environment`, `education`, `telecom`, а также `OTHER` и
`UNKNOWN`. Production labels фиксируются только при достаточных RU/KZ
примерах и проверяемой однозначности.

Mapping сохраняет обе стороны:

```text
source_system + raw_direction → canonical_topic + canonical_subtopic
```

## Память и версии

Минимальные PostgreSQL-сущности:

```text
tickets, ticket_predictions, operator_decisions,
topics, topic_source_mappings, services, routing_rules,
similarity_feedback, relation_feedback, response_templates,
alerts, alert_ticket_links, learning_cycles, learning_feedback,
dataset_versions, model_versions, model_evaluations,
background_jobs, users, audit_log
```

Prediction, operator decision и feedback имеют собственные timestamps,
`model_version`/`dataset_version` и автора. Evaluation/test split получает
immutable version и исключается из candidate training data.

## Demo fixtures

Demo-данные детерминированы и обезличены: RU/KZ тексты, 20 регионов, минимум
10 тем, похожие обращения, duplicate/repeat candidates, spike и forecastable
history. Они помечаются `dataset_kind=synthetic_demo` и никогда не выдаются за
реальные метрики или покрытие полного набора данных кейса.
