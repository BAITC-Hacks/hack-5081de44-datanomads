# Pulse 109 acceptance checks

Проверки в этой директории намеренно не импортируют `backend`, `frontend` или
`ml-service`. Они проверяют только внешний HTTP/OpenAPI-контракт, compose-файл,
структурированные логи и синтетический fixture.

Быстрый запуск до поднятия сервисов:

```bash
python -m unittest discover -s tests
python scripts/smoke_test.py --offline
```

`unittest` пропускает проверки compose/OpenAPI, пока соответствующие артефакты
не появились. `smoke_test.py --offline` предназначен для acceptance-gate и
завершается с ошибкой, если обязательный артефакт отсутствует.

Для работающего demo-стека:

```bash
docker compose --profile demo up --build
PULSE_BASE_URL=http://localhost:8080 \
  python scripts/smoke_test.py --log-file .tmp/pulse.jsonl
```

Полный stateful acceptance-контур (уникальный synthetic import, идемпотентный
повтор, PostgreSQL/Qdrant preview, operator decision, relation feedback,
analytics/drill-down, spike alerts + SSE, forecast, QueryIntent, PDF/XLSX,
RBAC, reindex и learning-cycle) запускается так:

```bash
PULSE_BASE_URL=http://localhost:8080 \
  python scripts/e2e_acceptance.py
```

Для дополнительной проверки сохранения решения после перезапуска Core:

```bash
PULSE_BASE_URL=http://localhost:8080 \
  python scripts/e2e_acceptance.py --restart-core
```

В обычном Compose worker использует версионированный Multinomial Naive Bayes
trainer для candidate artifact. Stateful acceptance строит candidate из
synthetic feedback, проходит evaluation и отклоняет его, проверяя, что
production pointer не изменился. При явном `PULSE_TEST_FAKE_TRAINER=true`
сценарий проверяет, что fake candidate помечен synthetic, остаётся
`INSUFFICIENT_EVIDENCE` и также не продвигается; его метрики не выдаются за
реальные ML-результаты.

Ролевые HTTP-пробы запускаются, если передан JSON с bearer-токенами. Значения
токенов не печатаются:

```bash
PULSE_ROLE_TOKENS='{"OPERATOR":"...","MANAGER":"...","ML_REVIEWER":"...","ADMIN":"..."}' \
  python scripts/smoke_test.py
```

Для локального demo-режима с контрактным `x-pulse-role` можно запускать те же
пробы без секретов:

```bash
PULSE_ROLE_HEADER_PROBE=1 python scripts/smoke_test.py
```

Проверка запрета автоматического promotion после feedback является opt-in и
требует тестовый `cycle_id`:

```bash
PULSE_ROLE_TOKENS='{"ML_REVIEWER":"..."}' \
  python scripts/smoke_test.py --learning-cycle-id <cycle-id>
```

В fixture явно указаны `dataset_version`, `seed`, `synthetic` и
`is_synthetic`; это не реальные продуктовые метрики.
