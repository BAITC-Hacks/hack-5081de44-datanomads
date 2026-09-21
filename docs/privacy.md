# Privacy и защита данных

Документ задаёт инженерные ограничения для demo/P0. Он не заменяет правовую
оценку, договоры с владельцами данных и требования действующего права.

## Минимизация

В Pulse импортируются только поля, которые нужны для маршрутизации,
retrieval, аналитики и контролируемого обучения. Исходные поля сохраняются
отдельно от производных предсказаний и доступны по роли.

По умолчанию не логируются и не отправляются в telemetry:

- полный текст обращения;
- ИИН/иные идентификаторы личности;
- телефон и e-mail;
- имя;
- полный адрес и точные координаты;
- вложения и их содержимое;
- свободный текст ошибки, если он может содержать PII.

В application logs допускаются только технические поля:

```text
timestamp, level, request_id, trace_id, service, endpoint,
latency_ms, model_version, status, error_code, ticket_id, user_id
```

`ticket_id` и `user_id` должны быть внутренними идентификаторами или
псевдонимами. Для отладки используется redacted/hashed значение, а не
исходный идентификатор.

## Контроль доступа

| Роль | Минимальные права |
| --- | --- |
| `OPERATOR` | разрешённые обращения, assist, confirm/correct, relation feedback |
| `MANAGER` | analytics, alerts, forecast, reports |
| `ML_REVIEWER` | candidate evaluation, learning cycles, promote/reject, model metadata |
| `ADMIN` | управление пользователями, политика доступа и model management |

Core API является единой границей auth/RBAC. Frontend не вызывает ML service
напрямую, а ML service не принимает решения о правах пользователя.

## Pipeline данных

1. Importer валидирует схему, типы и обязательные поля.
2. PII scan отправляет подозрительные строки в quarantine с причиной
   `PII_REVIEW`; строка не попадает молча в train.
3. Нормализатор сохраняет `raw`-значение и каноническое поле отдельно, если
   это разрешено политикой источника.
4. Dataset/evaluation split получает immutable version и checksum.
5. Training получает только одобренную минимизированную выборку.
6. Удаление/экспорт выполняется через Core API с audit log и проверкой роли.

## Qdrant

Qdrant хранит embedding и минимальный payload для фильтрации:
`ticket_id`, `region_id`, `topic_id`, `created_at`. Полный текст, имя,
адрес, контакты, вложения и иные PII в payload не дублируются. Результаты
поиска — только candidate IDs; разрешённые поля догружаются из PostgreSQL.

## LLM и внешние сервисы

Natural-language analytics использует optional intent parser, который возвращает
строгий `QueryIntent`. LLM не имеет прямого доступа к PostgreSQL, не строит
свободный Text-to-SQL и не выполняет SQL. Core API валидирует intent, подставляет
параметры в allow-listed parameterized queries и возвращает число, таблицу и
график. При недоступном LLM аналитика работает по явным фильтрам.

PII не отправляется внешнему LLM/API без отдельного, обоснованного и
аудируемого разрешения. По умолчанию внешний provider отключён (`disabled`).

## Demo и секреты

- Фикстуры demo обезличены, помечены `synthetic_demo` и не являются реальными
  метриками.
- Секреты хранятся в `.env`/secret store и не коммитятся; `.env.example`
  содержит только demo placeholders.
- Перед внешним доступом меняются `POSTGRES_PASSWORD`, ключи интеграций,
  origins и политика ingress.
- Public Nginx не проксирует `/internal/*`, `/docs` и ML administration API.

## Аудит и инциденты

Изменения operator decision, feedback, model promotion/rejection, export и
доступ к чувствительному тексту фиксируются в `audit_log` с actor, timestamp,
request id и reason. При утечке или ошибочной выдаче доступ приостанавливается,
сохраняются технические логи без PII и проводится review затронутых dataset и
model versions.
