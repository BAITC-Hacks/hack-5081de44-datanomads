# Синтетический пилот Pulse 109 через NeMo Data Designer

## Вход и схема

[`pilot_scenarios.jsonl`](../data/sdg/pilot_scenarios.jsonl) содержит 25
**вымышленных** ситуаций по 14 темам. Каждая ситуация связана с одной
однозначной парой `category + service` из
[`almaty_2025_taxonomy_review.json`](../data/catalogs/almaty_2025_taxonomy_review.json).
Сырые обращения, комментарии и исполнители в seed не передаются.

| Поле | Источник | Назначение |
| --- | --- | --- |
| `scenario_id`, `topic_id`, `subtopic_id` | Наш seed | Неизменяемая ожидаемая разметка и группа split. |
| `source_category`, `source_service` | Проверенная пара каталога | Происхождение темы; в prompt не вставляется. |
| `facts_ru` | Придуманная ситуация | Только разрешённые факты для текста. |
| `language` | Sampler Data Designer | `RU` 45%, `KZ` 45%, `MIXED` 10% как целевая пропорция. |
| `style` | Sampler Data Designer | `short`, `conversational`, `neutral`. |
| `appeal_text` | Локальная LLM | Одно обращение без разметки и выдуманных действий службы. |

Data Designer читает локальный JSONL как seed, выбирает язык и стиль, затем
создаёт только текст. Повторные проходы по одному `scenario_id` дают варианты;
все варианты этой ситуации позже должны попадать в один train/validation/test
split. Для первого пилота не создаём `OTHER`, `UNKNOWN`, приоритеты и
исполнителей: у этих меток нет достаточно определённой основы в выбранных
ситуациях.

## Запуск в WSL

Нужны Python 3.10+, установленный `data-designer` и локальный сервер с
OpenAI-совместимым `/v1/chat/completions`. NeMo Data Designer поддерживает
локальный JSONL seed и собственный `ModelProvider` для такого endpoint:
[seed datasets](https://docs.nvidia.com/nemo/datadesigner/concepts/seed-datasets),
[custom model settings](https://docs.nvidia.com/nemo/datadesigner/concepts/models/custom-model-settings).

```bash
python3 -m venv .venv-sdg
source .venv-sdg/bin/activate
python -m pip install data-designer==0.9.3

python scripts/pulse_sdg.py --check-seeds
python scripts/pulse_sdg.py --model qwen3.5:9b --num-records 25
```

Если Ollama доступна по другому адресу, добавьте
`--endpoint http://<адрес>:11434/v1`. Скрипт ограничивает Data Designer одним
запросом к модели одновременно. Версию модели и пригодность 3060 нужно
проверить на фактическом запуске, а не предполагать по размеру файла модели.
Конфигурация проверена через `DataDesigner.validate()` с версией `0.9.3`.
Локальный Ollama принимает фиктивный API key `ollama`; внешний ключ не нужен.
Его OpenAI-совместимый endpoint и управление `reasoning_effort` описаны в
[документации Ollama](https://docs.ollama.com/api/openai-compatibility).

Результат появляется в игнорируемом Git каталоге `data/sdg/runs/<время>/`:
артефакты Data Designer, `candidates.jsonl` и `summary.json`. Каждый кандидат
имеет `synthetic=true`, `split_group=scenario_id` и
`review_status="PENDING"`. Записываются checksum source scenarios, версия
prompt, model ID и seed запроса к локальной LLM. `variant_id` вычисляется из
сценария, языка, стиля и текста, поэтому порядок строк не меняет ID.
`--generator-seed` по умолчанию равен `109`; [Ollama OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility)
поддерживает поле `seed` для chat completions. Seed задаётся для LLM запроса;
он не доказывает детерминизм sampler Data Designer и полного повторного запуска.
Это **не готовый обучающий корпус**.

## Проверки

[`pulse_sdg.py`](../scripts/pulse_sdg.py) до обращения к модели проверяет, что
каждый seed соответствует паре `CANDIDATE` с явным подтипом в каталоге. После
генерации он отклоняет строки со сломанной схемой, слишком коротким или длинным
текстом, точные повторы и известные форматы PII (телефон, ИИН, e-mail,
помеченные имя и адрес).
Отчёт содержит только счётчики отклонений.

Автоматическая проверка **не доказывает**, что модель сохранила смысл, правильно
написала казахский текст и не добавила новые факты. Такие варианты удаляются
при просмотре пилота; `PENDING` не допускается в trainer. После оценки первых
текстов можно подключить [NeMo Curator](https://docs.nvidia.com/nemo/curator/curate-text/synthetic)
для масштабной дедупликации и фильтрации.
Для 25–300 строк его GPU-конвейер не требуется: простой точный dedup уже есть,
а семантическая дедупликация сама по себе не проверяет правильность метки.
