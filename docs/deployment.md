# Развёртывание Pulse 109

## Demo/local

Требования: Docker Engine и Compose v2, доступ к локальному registry/cache и
свободный порт 8012 (или `PULSE_HTTP_PORT` из `.env`). Запуск:

```bash
cp .env.example .env
docker compose --profile demo config
docker compose --profile demo up --build
```

Контейнер `demo-seed` ждёт `/readyz`, проверяет checksum synthetic fixture и
идемпотентно загружает обращения через Core API. Его exit code должен быть 0.
Core healthcheck ждёт успешные миграции и готовность PostgreSQL, Qdrant и ML;
`ml-worker` и публичный Nginx запускаются только после успешного завершения
seed. Поэтому первичный reindex не конкурирует с import, а публичные запросы
не приходят до появления demo-данных.

После запуска:

```bash
curl -fsS http://localhost:8012/healthz
curl -fsS http://localhost:8012/readyz
scripts/smoke
```

Остановка без удаления данных:

```bash
docker compose --profile demo down --remove-orphans
```

Полный reset demo (удаляет PostgreSQL, Qdrant и ML artifact volumes) требует
явного подтверждения:

```bash
PULSE_CONFIRM_RESET=1 scripts/demo-reset
```

Не запускайте reset в окружении с нужными данными. Raw dataset не монтируется
в compose автоматически; demo fixture должен быть deterministic и
обезличенным.

## Demo на выделенном сервере через Docker

Это способ показать **синтетический** проект на хакатоне. Для него нужны Linux
сервер с Docker Engine и Compose v2, доменное имя, HTTPS сертификат и внешний
reverse proxy с доступом только для приглашённых зрителей. Пример ниже
использует Nginx на хосте; он направляет трафик в Compose Nginx на
`127.0.0.1:8012`. Compose публикует только этот loopback-порт. В firewall
откройте только SSH и порты 80/443 для ingress.

1. Получите репозиторий на сервере, перейдите в его корень и создайте
   локальный `.env`:

   ```bash
   cp .env.example .env
   chmod 600 .env
   ```

2. В `.env` замените `POSTGRES_PASSWORD=pulse_demo_only` на отдельный пароль
   этого сервера и укажите точный browser origin, например
   `CORS_ALLOWED_ORIGINS=https://demo.example.org`. Оставьте
   `PULSE_ENV=demo`, `PULSE_DEV_AUTH=true` и
   `OPTIONAL_LLM_PROVIDER=disabled`. Пароль, сертификат и `.env` не добавляйте
   в Git. Если PostgreSQL volume уже инициализирован, смена переменной
   `POSTGRES_PASSWORD` не меняет пароль внутри существующей БД: потребуется
   отдельная управляемая ротация.

3. Проверьте конфигурацию, запустите stack и дождитесь seed:

   ```bash
   docker compose --profile demo config --quiet
   docker compose --profile demo up --build -d
   docker compose --profile demo ps
   scripts/smoke
   ```

   `demo-seed` должен завершиться с кодом `0`, а `scripts/smoke` — строкой
   `Pulse 109 smoke checks passed.`. Проверка использует loopback на самом
   сервере и не требует публичного URL.

   Если Core остаётся `unhealthy`, прочитайте JSON `/readyz` из контейнера:

   ```bash
   docker compose --profile demo exec -T core-api python3 - <<'PY'
   import urllib.error
   import urllib.request

   try:
       response = urllib.request.urlopen('http://127.0.0.1:8080/readyz')
   except urllib.error.HTTPError as error:
       response = error
   print(response.read().decode())
   PY
   ```

   При ответе 503 проверьте `checks` в теле ответа. В частности,
   `PRODUCTION_CLASSIFIER_ARTIFACT_UNVERIFIABLE` означает недостоверный
   production model pointer в существующем PostgreSQL volume; не удаляйте
   volume для устранения этой ошибки без проверки сохранённых решений.

4. Настройте HTTPS ingress. Ниже фрагмент конфигурации **хостового** Nginx;
   замените домен и пути к уже полученному сертификату. Создайте отдельный
   файл Basic Auth (`htpasswd -cB /etc/nginx/pulse109.htpasswd demo` запускают
   с правами администратора), проверьте `nginx -t` и перезагрузите Nginx.

   ```nginx
   server {
       listen 80;
       server_name demo.example.org;
       return 301 https://$host$request_uri;
   }

   server {
       listen 443 ssl;
       server_name demo.example.org;
       ssl_certificate /etc/letsencrypt/live/demo.example.org/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/demo.example.org/privkey.pem;

       auth_basic "Pulse 109 demo";
       auth_basic_user_file /etc/nginx/pulse109.htpasswd;

       location / {
           proxy_pass http://127.0.0.1:8012;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_http_version 1.1;
           proxy_buffering off;
           proxy_read_timeout 1h;
       }
   }
   ```

5. Проверьте, что HTTPS требует пароль, после входа открывается приложение, а
   `/readyz` возвращает `ready`. Публичные `/internal/*`, `/docs` и
   `/api/v1/docs` должны возвращать 404 через application gateway. Перед
   передачей ссылки жюри пройдите маршрут: очередь → подсказка → решение
   оператора → обзор руководителя.

Basic Auth ограничивает доступ ко **всему demo**, но роли внутри приложения
по-прежнему выбираются посетителем: `PULSE_DEV_AUTH=true` не является
production identity. На сервер загружается только synthetic fixture. Для
работы с настоящими обращениями нужны внешний identity contract и отдельный
защищённый контур.

При обновлении сначала сохраните данные, если в demo появились нужные
решения, затем выполните `git pull --ff-only`,
`docker compose --profile demo up --build -d` и `scripts/smoke` на сервере.
Миграции применяются при старте Core; обратная совместимость схемы должна быть
проверена до отката. `scripts/demo-reset` удаляет volumes и для обновления не
нужен.

## Сервисный контракт

| Container | Listen | Compose dependency |
| --- | --- | --- |
| `postgres` | 5432 | — |
| `qdrant` | 6333/6334 | — |
| `ml-service` | 8000 | basic health не зависит от БД |
| `core-api` | 8080 | postgres, qdrant, ml-service |
| `frontend` | 5174 | core-api |
| `ml-worker` | — | postgres, qdrant, ml-service; profile `demo` |
| `demo-seed` | — | core-api; profile `demo`, exits after import |
| `nginx` | 80 → host `127.0.0.1:8012` (`PULSE_HTTP_PORT`) | frontend/core-api |

ML docs и Core route index доступны только из внутренней Compose-сети или при
отдельном запуске сервисов. Nginx возвращает 404 на docs paths и не
проксирует внутренние ML endpoints. Сервисы внутри Compose продолжают
обращаться друг к другу по внутренней сети. Для внешнего доступа нужен
отдельный доверенный ingress.

## Production checklist

Перед внешним доступом необходимо:

1. заменить demo password и все placeholder secrets через secret manager;
2. задать точный список browser origins в `CORS_ALLOWED_ORIGINS` (через запятую
   для нескольких origin); wildcard запрещён, пустой список отключает CORS.
   По умолчанию разрешён только `http://localhost:8012`, а единственный
   Compose host port привязан к `127.0.0.1`;
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

Core применяет embedded migrations при старте после доступности PostgreSQL и
до открытия HTTP listener; `/readyz` остаётся нездоровым, пока версия миграций
не совпадает с бинарём. Это поддерживает свежие и существующие named volumes.
Перед production migration делаются backup и dry-run в staging. Изменение UnifiedTicket,
prediction/decision разделения, model metadata или learning state требует
совместимого обновления API/OpenAPI и проверок старых записей.

## Rollback и восстановление

- Demo Compose собирает локальные images с тегом `:local`; автоматического
  rollback по commit SHA нет. Для контролируемого production отката нужны
  отдельно сохранённые immutable images и проверенная совместимость миграций.
- Production model откатывается на предыдущий verified `model_version`, а не
  заменой файла в mounted directory.
- Rejected candidate не требует rollback production.
- При потере Qdrant индекса его можно пересобрать из PostgreSQL; PostgreSQL
  остаётся источником истины.
- После restore проверяются counts, checksums, model pointer, audit log и
  `/readyz`, затем запускается smoke.
