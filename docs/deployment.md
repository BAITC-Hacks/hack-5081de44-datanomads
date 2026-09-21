# Deployment

## Demo/local

Требования: Docker Engine и Compose v2, доступ к локальному registry/cache и
свободные порты из `.env`. Запуск:

```bash
cp .env.example .env
docker compose --profile demo config
docker compose --profile demo up --build
```

После запуска:

```bash
curl -fsS http://localhost:8080/healthz
curl -fsS http://localhost:8080/readyz
scripts/smoke
```

Остановка без удаления данных:

```bash
docker compose --profile demo down --remove-orphans
```

Полный reset demo (удаляет PostgreSQL/Qdrant volumes) требует явного
подтверждения:

```bash
PULSE_CONFIRM_RESET=1 scripts/demo-reset
```

Не запускайте reset в окружении с нужными данными. Raw dataset не монтируется
в compose автоматически; demo fixture должен быть deterministic и
обезличенным.

## Сервисный контракт

| Container | Listen | Compose dependency |
| --- | --- | --- |
| `postgres` | 5432 | — |
| `qdrant` | 6333/6334 | — |
| `ml-service` | 8000 | basic health не зависит от БД |
| `core-api` | 8080 | postgres, qdrant, ml-service |
| `frontend` | 5174 | core-api |
| `ml-worker` | — | postgres, qdrant, ml-service; profile `demo` |
| `nginx` | 80 → host `PULSE_HTTP_PORT` | frontend/core-api |

Внутренние ML endpoints не публикуются на host и не проксируются Nginx.

## Production checklist

Перед внешним доступом необходимо:

1. заменить demo password и все placeholder secrets через secret manager;
2. ограничить `CORS_ALLOWED_ORIGINS`, host ports и network ingress;
3. включить TLS перед Nginx (или доверенный ingress) и проверить forwarded
   headers;
4. настроить backup/restore PostgreSQL и Qdrant snapshot policy;
5. применить миграции до включения Core API и проверить `/readyz`;
6. загрузить только проверенные immutable model artifacts и сверить SHA256;
7. настроить retention и redaction для structured logs/audit log;
8. выполнить API contract, PII, negative и E2E checks;
9. проверить rollback на предыдущий Core API image и production model pointer.

Compose demo не является production hardening: он использует локальные
volumes, single replicas и placeholder credentials.

## Миграции и данные

Миграции применяются отдельной явной командой/entrypoint Core API после
доступности PostgreSQL, не «тихой» модификацией схемы на каждом запросе.
Перед миграцией делаются backup и dry-run в staging. Изменение UnifiedTicket,
prediction/decision разделения, model metadata или learning state требует
совместимого обновления API/OpenAPI и проверок старых записей.

## Rollback и восстановление

- Core API/frontend/ML images versioned по commit SHA.
- Production model откатывается на предыдущий verified `model_version`, а не
  заменой файла в mounted directory.
- Rejected candidate не требует rollback production.
- При потере Qdrant индекса его можно пересобрать из PostgreSQL; PostgreSQL
  остаётся источником истины.
- После restore проверяются counts, checksums, model pointer, audit log и
  `/readyz`, затем запускается smoke.
