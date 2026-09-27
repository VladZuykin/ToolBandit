# ToolBandit

ToolBandit — HTTP-сервис для семантического поиска инструментов, проверки их по бюджету и SLA, добавления контекстного **ucb_score** и обучения с отложенной оценкой результата моделью-оценщиком.

## О чем сервис

Учитывайте следующее:

- сервис **не вызывает инструменты**: он возвращает релевантный список, дополняя своим скором, а вызов выполняет агент, после чего ему нужно получить от него обратно актуальный latency и cost;
- порядок **tools** определяет semantic retriever; LinUCB только добавляет **ucb_score** и **не меняет порядок**;
- определения инструментов и cache embeddings сохраняются в PostgreSQL, но состояние LinUCB, request_id и задания Judge находятся в памяти и теряются после рестарта;
- **cost** обязателен при регистрации; необязательная **latency** автоматически обучается по реальным вызовам;
- используются модели для эмбеддинга от OpenAI (text-embedding-ada-002) и DeepSeek (по умолчанию deepseek-flash) в качестве оценщика;
- лицензия проекта — Apache-2.0.

## Быстрый старт

### Требования и установка

- Python 3.11;
- ключ OpenAI для эмбеддингов;
- ключ DeepSeek для модели-судьи.

Git Bash в Windows:

```bash
cd ~/Desktop/ToolBandit
python.exe -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Переменные окружения

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DATABASE_URL="postgresql://toolbandit:password@127.0.0.1:5432/toolbandit"
export TOOLBANDIT_HOST="127.0.0.1"
export TOOLBANDIT_PORT="8080"
export TOOLBANDIT_COVARIANCE="diagonal"
```

| Переменная | Обязательна | Назначение | По умолчанию |
|---|---:|---|---|
| `OPENAI_API_KEY` | для новых эмбеддингов (не из кеша) | Кодирование запросов и инструментов | отсутствует |
| `DEEPSEEK_API_KEY` | да | Для модели-судьи | отсутствует |
| `DEEPSEEK_JUDGE_MODEL` | нет | Модель-судья | `deepseek-flash` |
| `DATABASE_URL` | да | PostgreSQL для registry и cache embeddings | отсутствует |
| `TOOLBANDIT_HOST` | нет | Адрес Uvicorn | `127.0.0.1` |
| `TOOLBANDIT_PORT` | нет | Порт Uvicorn | `8080` |
| `TOOLBANDIT_COVARIANCE` | нет | `diagonal` или `full` | `diagonal` |
| `TOOL_COST_SCALE` | нет | Масштаб cost в UCB score | `0.012` |
| `TOOL_LATENCY_SCALE` | нет | Масштаб latency в UCB score | `3.0` |

### Docker Compose

```bash
cp .env.example .env
# перед этим нужно заполнить POSTGRES_PASSWORD, OPENAI_API_KEY и DEEPSEEK_API_KEY
docker compose up --build -d
docker compose ps
curl -s http://127.0.0.1:8080/health | python -m json.tool
```

Новый PostgreSQL volume и новый сервис стартуют с пустым каталогом. Инструменты добавляются через API после запуска. Остановка без удаления данных: `docker compose down`. Полное удаление БД: `docker compose down -v`.

### Проверка и запуск

```bash
python scripts/check_setup.py 
python scripts/check_api_connections.py
python run_service.py
```

```bash
curl -s http://127.0.0.1:8080/health | python -m json.tool
```

- Swagger UI: <http://127.0.0.1:8080/docs>
- OpenAPI JSON: <http://127.0.0.1:8080/openapi.json>

## Минимальный рабочий пример

### Добавить инструмент

```bash
curl -s -X POST http://127.0.0.1:8080/v1/tools \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "calculator",
    "description": "Evaluate a mathematical expression and return a numeric result",
    "input_schema": {
      "type": "object",
      "required": ["expression"],
      "properties": {"expression": {"type": "string"}}
    },
    "output_schema": {
      "type": "object",
      "required": ["result"],
      "properties": {"result": {"type": "number"}}
    },
    "cost": 0,
    "latency": 0.1,
    "metadata": {"provider": "internal"},
    "enabled": true
  }' | python -m json.tool
```

`input_schema` и `output_schema` используют JSON Schema. Сервис сохраняет схемы как метаданные и включает их в semantic document, но пока не валидирует реальный tool call по этим схемам.

### Retriever

```bash
curl -s -X POST http://127.0.0.1:8080/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate 125 divided by 5",
    "remaining_budget": 1,
    "latency_sla": 5,
    "retrieval_limit": 20,
    "result_limit": 5,
    "retrieval_threshold": 0
  }' | python -m json.tool
```

Ответ:

```json
{
  "request_id": "b66a2d1f-...",
  "retrieved_count": 11,
  "feasible_count": 2,
  "tools": [
    {
      "tool": "calculator",
      "retrieval_score": 0.82,
      "retrieval_rank": 1,
      "ucb_score": 1.73,
      "metadata": {}
    }
  ]
}
```

## Как это работает

```text
Запрос пользователя
        ↓
OpenAI embedding (1536 измерений)
        ↓
Semantic retrieval по описаниям и JSON Schema
        ↓
Фильтрация по remaining_budget, latency_sla и excluded_tools
        ↓
Список в порядке semantic relevance + диагностический ucb_score
        ↓
Внешний агент вызывает выбранный инструмент
        ↓
DeepSeek Judge асинхронно оценивает результат
        ↓
LinUCB получает бинарный reward (успешность использования) и обновляет модель инструмента
```

### Semantic retriever

Retriever кодирует запрос и описание каждого инструмента с помощью [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings), нормализует векторы и считает cosine similarity.

- retrieval_score — cosine similarity запроса и semantic document;
- retrieval_rank — исходная позиция retriever;
- порядок массива tools сохраняет semantic ranking, но добавляет score.

### LinUCB

[LinUCB](https://arxiv.org/abs/1003.0146) — contextual bandit: модель связывает embedding контекста с reward конкретной LLM. Используется disjoint-модель — для каждой LLM собственные параметры. Отсюда черпается идея, но мы работаем с инструментами.

В сервисе применяется diagonal approximation: вместо полной матрицы **1536 × 1536** для каждого инструмента хранится диагональ. Это уменьшает память и вычисления, но не моделирует взаимодействия между координатами embedding.

Режим выбирается перед запуском:

```bash
# O(d) памяти и вычислений на инструмент — рекомендуется для production
export TOOLBANDIT_COVARIANCE="diagonal"

# O(d²) памяти и вычислений на инструмент — для сравнительных экспериментов
export TOOLBANDIT_COVARIANCE="full"
```

При **d=1536** full-модель хранит примерно 18 MiB только для A⁻¹ каждого инструмента; 50 инструментов требуют около 900 MiB без учёта остальных данных. Текущий режим отображается в GET /health в поле **covariance**.

**ucb_score** дополнительно учитывает внутренние оценки cost и latency.

### Бюджет и latency SLA

Перед выдачей результата сервис проверяет:

```text
estimated_cost    <= remaining_budget
estimated_latency <= latency_sla
```

Источник оценки:

1. исходный cost и необязательная latency из registry (в начале);
2. фактические observed_cost и observed_latency реальных вызовов;
3. после каждого вызова средние обновляются автоматически, независимо от результата Judge.

Если изначально с добавлением инструмента **latency** не задана, первый вызов считается cold start и не отклоняется из-за неизвестной задержки. После этого сервис использует измеренную **observed_latency**.

### DeepSeek в качестве судьи и delayed feedback

Judge получает намерение, ожидаемый контракт и **tool_output**, затем возвращает:

- pass - требования выполнены, возвращает reward=1;
- partial - частичный результат, возвращает reward=0;
- fail - результат непригоден, возвращает reward=0;
- uncertain - недостаточно уверенности, LinUCB в таком случае не обновляется.

## REST API

### GET /health

```json
{
  "status": "ok",
  "judge": true,
  "registry": true,
  "active_tools": 11,
  "api_version": "1.1.0",
  "ranking_policy": "semantic_order_with_ucb_annotation"
}
```

### GET /v1/registry/stats

Возвращает total - всего инструментов, enabled - включенных и active_in_router.

### POST /v1/tools

Создаёт инструмент и обновляет semantic index.

| Поле | Тип | Обязательно | Описание |
|---|---|---:|---|
| `tool_name` | string | да | Уникальное имя |
| `description` | string | да | Описание инструмента |
| `input_schema` | object | да | JSON Schema аргументов |
| `output_schema` | object | да | JSON Schema результата |
| `cost` | number | да | Начальная стоимость вызова |
| `latency` | number | нет | Начальная оценка задержки в секундах |
| `metadata` | object | нет | Метаданные |
| `enabled` | boolean | нет | Флажок участия в поиске, по умолчанию true |

metadata не влияет на retrieval или UCB. Агент может хранить там provider (для Toolbench), HTTP method/path.

Ответ: 201 Created. Повторное имя: 409 Conflict.

### GET /v1/tools

Query parameters:

- **limit** — 1–1000, default 100;
- **offset** — смещение;
- **enabled=true|false** — необязательный фильтр.

```bash
curl -s "http://127.0.0.1:8080/v1/tools?limit=100&offset=0&enabled=true"
```

### GET /v1/tools/{tool_name}

Возвращает определение, version, created_at и updated_at. Неизвестное имя: 404.

### PUT /v1/tools/{tool_name}

Полностью заменяет определение. Имена в path и body должны совпадать. Версия увеличивается, semantic index обновляется, reward-обучение сохраняется.

### PATCH /v1/tools/{tool_name}/enabled

```json
{"enabled": false}
```

Выключенный инструмент остаётся в registry, но не участвует в поиске.

### DELETE /v1/tools/{tool_name}

Физически удаляет из текущего router. Успех: 204 No Content.

### POST /v1/tools/import

Bulk import:

```json
{
  "tools": [
    {
      "tool_name": "example",
      "description": "Example tool",
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"},
      "cost": 0,
      "enabled": true
    }
  ],
  "replace_existing": true
}
```

При replace_existing=false существующие имена пропускаются. После импорта активный semantic index обновляется.

### POST /v1/tools/search

| Поле | Обязательно | Смысл |
|---|---:|---|
| query | да | Непустой запрос |
| remaining_budget | да | Бюджет текущей попытки |
| latency_sla | нет | Предел latency |
| retrieval_limit | нет | Число semantic candidates, default 20 |
| result_limit | нет | Число результатов, default 5 (после отбора по SLA) |
| retrieval_threshold | нет | Минимальный cosine similarity, default 0 |
| excluded_tools | нет | Запрещённые инструменты |


Ответ:

- request_id — ID сохранённого embedding для delayed feedback;
- retrieved_count — кандидаты после semantic threshold/limit;
- feasible_count — кандидаты после budget/SLA/exclusion;
- tools — первые **result_limit** feasible-кандидатов в semantic order;
- retrieval_score, retrieval_rank=semantic retrieval;
- ucb_score — дополнительный контекстный score от нашей системы;
- metadata — метаданные.

### POST /v1/evaluations

Отправляет результат на асинхронную оценку. Ответ: 202 Accepted.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{
    "interaction_id": "agent-run-42-call-2",
    "request_id": "request-id-from-search",
    "tool_name": "calculator",
    "tool_intent": "Calculate 125 divided by 5",
    "expected_contract": "Return numeric result equal to 25",
    "tool_output": {"result": 25},
    "observed_cost": 0,
    "observed_latency": 0.03
  }' | python -m json.tool
```

| Поле | Описание |
|---|---|
| interaction_id | Ключ попытки |
| request_id | Предпочтительный ID предыдущего search |
| context_text | Fallback, если **request_id** отсутствует |
| tool_name | Реально вызванный инструмент |
| tool_intent | Что должен был выполнить вызов |
| expected_contract | Проверяемые требования |
| tool_output | Структурированный результат или ошибка |
| observed_cost | Фактическая стоимость |
| observed_latency | Фактическая длительность в секундах |

Нужно передать request_id или context_text. Повторный interaction_id не применяет reward дважды.

### GET /v1/evaluations/{evaluation_id}

Статусы: pending, completed, failed.

```json
{
  "evaluation_id": "agent-run-42-call-2",
  "status": "completed",
  "learned": true,
  "judgment": {
    "verdict": "pass",
    "passed": true,
    "confidence": 0.98,
    "reason_code": "ALL_REQUIREMENTS_MET",
    "explanation": "Result is numeric and equals 25.",
    "missing_requirements": [],
    "total_tokens": 320,
    "model": "deepseek-flash"
  }
}
```

## Tool adapter и примеры

ToolAdapter — абстрактный класс-обертка для работы с роутером:

```text
catalog_entry()  → описание для registry
estimate(args)   → estimated_cost и estimated_latency до вызова
invoke(args)     → output, observed_cost, observed_latency, status/error
```

В комплект входят десять примеров API.

```bash
python scripts/register_public_api_examples.py
python scripts/run_real_api_showcase.py --skip-judge
python scripts/run_real_api_showcase.py
```

## Структура проекта

```text
ToolBandit/
├── budget_tool_router/          production-код
│   ├── ada_embeddings.py       OpenAI encoder
│   ├── adapters.py             tool adapters
│   ├── judge.py                DeepSeek Judge
│   ├── registry.py             PostgreSQL registry
│   ├── retriever.py            semantic retrieval
│   ├── router.py               diagonal/full disjoint LinUCB
│   ├── runner.py               fallback attempts
│   └── service.py              FastAPI и application service
├── scripts/                     проверки и демонстрации
├── tests/                       pytest-тесты
├── run_service.py
├── requirements.txt
└── python_version.txt
```

## Тестирование

```bash
python -m pytest -q
python scripts/check_setup.py
python scripts/check_api_connections.py
```

## Доп. информация

- **Contextual bandit** — выбор действия с учётом контекста и reward: [LinUCB paper](https://arxiv.org/abs/1003.0146).
- **UCB** — upper confidence bound, оптимистичная оценка reward с exploration bonus.
- **SLA** — service-level agreement; здесь предел допустимой задержки.
- **Delayed feedback** — reward приходит после выполнения инструмента.
- **ToolBench** — набор данных и инфраструктура tool-use: [ToolLLM/ToolBench](https://arxiv.org/abs/2307.16789).

## Лицензия

Проект распространяется по лицензии Apache License 2.0. Полный юридический текст находится в файле `LICENSE`.
