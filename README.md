# ai-support — первый сквозной срез

Python 3.12, FastAPI, SQLAlchemy, Alembic, PostgreSQL + pgvector. Состояние,
попытки, обратная связь и создание заявки контролируются backend.
Внешний веб-поиск отсутствует. Nextcloud и почта работают **только как mock**.

## Быстрый запуск

Нужен Docker Desktop с Linux containers либо Docker Engine с Compose v2.
Существующие Open WebUI/Qwen для демонстрационного режима не нужны.

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose ps -a
```

В Linux/macOS вместо `Copy-Item` используйте `cp .env.example .env`.
Повторный запуск: `docker compose up -d`.

`migrate` должен завершиться с кодом 0; `api`, `worker`, `db` продолжают работу.
Данные сохраняются в named volume PostgreSQL. API опубликован только на
`127.0.0.1:8000`; БД наружу не публикуется.

- Swagger: http://localhost:8000/docs
- Readiness: http://localhost:8000/health/ready
- В Swagger нажмите **Authorize**, введите `local-demo-token` (без слова Bearer).
- При изменении `API_TOKEN` в `.env` используйте новое значение.
- Для каждого нового POST задавайте новый `Idempotency-Key`; при сетевом повторе — тот же.

## Сквозной тест за одну команду

```bash
docker compose exec api python scripts/demo.py
```

Скрипт отправляет «Принтер не печатает», ожидает worker, пять раз отвечает
«Не помогло», заполняет форму и проверяет `ESCALATED` + `SENT_MOCK`.
В stdout появятся ID обращения и названия пяти разных решений.
Он создаёт демонстрационную историю и влияет на статистику тестовой базы.
Перед запуском завершите ранее открытое обращение того же mock-пользователя.

Письмо **никуда не отправляется**. Полный снимок письма доступен через
`GET /api/incidents/{id}` → `escalation.mock_email`.

## Ручной сценарий через Swagger

1. `POST /api/chat`, `Idempotency-Key: demo-chat-1`:

   ```json
   {"message": "Принтер не печатает"}
   ```

   Ответ `202` содержит `incident_id`, `message_id`, `operation_id`.

2. `GET /api/incidents/{incident_id}`: подождите `WAITING_FEEDBACK`.
   Прочитайте `messages`; возьмите последний `attempts[].id`.

3. `POST /api/incidents/{incident_id}/feedback`, новый `Idempotency-Key`:

   ```json
   {"attempt_id": "UUID ИЗ ПРЕДЫДУЩЕГО ОТВЕТА", "helped": false}
   ```

   Worker предложит следующее решение. Повторите для пяти разных попыток.
   `helped: true` вместо этого завершит обращение как `RESOLVED`.

4. После пятой неудачи: `state=DIAGNOSING`,
   `pending_action=COLLECT_ESCALATION_DETAILS`; последнее сообщение содержит таблицу.

5. `POST /api/incidents/{incident_id}/escalation-details`, новый `Idempotency-Key`:

   ```json
   {
     "account_holder_name": "Иванов Иван Иванович",
     "account_login": "ivanov",
     "computer_name": "WS-001",
     "subject": "Принтер не печатает",
     "additional_info": "Не помогли пять предложенных решений"
   }
   ```

6. `GET /api/incidents/{id}`: `ESCALATED`, затем `escalation.status=SENT_MOCK`.
   Проверяйте ФИО, логин, компьютер, тему, время UTC, полную историю и summary в письме.
7. `GET /api/admin/solutions/stats` показывает попытки, успехи, неудачи и сглаженную оценку.

Дополнительные проверки:

- «Сломался космический телескоп» → `NO_KNOWLEDGE`, форма, **ноль попыток**.
- «Помогите» → `NEED_MORE_INFO`; уточнение с тем же `incident_id` не тратит попытку.
- «оператор» → досрочная форма эскалации.
- Без `incident_id` новое сообщение попадёт в активное обращение; новое обращение
  создаётся после завершения предыдущего.
- Текст «Помогло»/«Не помогло» через `/chat` пока отклоняется с подсказкой использовать
  `/feedback`: в этом срезе feedback передаётся структурированно с `attempt_id`.
- Пока ответ обрабатывается либо ожидается feedback, произвольные новые сообщения
  получают `409`. Это явное ограничение первого среза, а не потеря сообщения.

## Подключение реального Qwen через Open WebUI

В `.env`:

```dotenv
LLM_BACKEND=openwebui
OPENWEBUI_BASE_URL=http://host.docker.internal:3000
OPENWEBUI_API_KEY=<ключ сервисной учётной записи>
OPENWEBUI_MODEL=<точный model_id вашей установки>
LLM_TIMEOUT_SECONDS=60
```

```bash
docker compose up -d --force-recreate api worker
docker compose logs -f worker
```

Адрес должен быть доступен **из контейнера**. При отдельном сервере укажите его DNS/IP.
Адаптер вызывает `POST /api/chat/completions` с Bearer token, `stream=false`,
`response_format={"type":"json_object"}`. Нужна поддержка JSON mode на вашей связке
Open WebUI/сервер инференса. Профиль модели должен быть без подключённых инструментов,
внешнего поиска и выполняющих действия pipelines.

LLM анализирует сообщение, возвращает запрос и JSON-представление ответа,
а при эскалации — summary. В этом срезе свободная генерация технических инструкций
намеренно отключена: Qwen выбирает разрешённое вступление, **все шаги backend
копирует из выбранной утверждённой карточки**. Невалидный ответ или чужие ссылки
заменяются карточкой. Сбой анализа повторяется максимум три раза, затем пользователь
получает сообщение о недоступности; попытка решения не создаётся. Ошибка summary
не мешает сформировать письмо с полной историей.

Контракт реального адаптера проверяется HTTP-mock тестами. Совместимость с вашей
конкретной установкой Qwen требует отдельного интеграционного запуска.

## База знаний и поиск

При bootstrap загружается `sample_knowledge/printer.json`: пять **демонстрационных**
карточек. Это учебные примеры, не утверждённые корпоративной службой инструкции.
Флаг `approved=true` применяется к ним только для проверки workflow.
Не используйте эти карточки как реальную корпоративную базу без проверки владельцем.

`SEED_DEMO_KNOWLEDGE=false` отключает загрузку; повторный bootstrap также снимает
допуск с ранее загруженного демо-документа:

```bash
docker compose run --rm migrate
```

Заполнение демонстрации идемпотентно и не перезаписывает существующие версии карточек.
Не меняйте JSON в расчёте на автоматическое обновление уже созданной базы:
API импорта/версионирования будет отдельным этапом.

RAG сейчас использует локальный лексический baseline: нормализация терминов,
128-мерный hash-вектор, pgvector cosine ordering и проверка реального пересечения
терминов. Это **не семантическая embedding-модель**. Проверка пересечения отсеивает
совпадения из-за коллизий хеша. Для демонстрационного объёма просматриваются все
допущенные карточки. Семантическая модель, масштабирование и подбор порога будут
добавлены после получения корпоративного корпуса.

Сначала учитывается релевантность, затем среди одинаково релевантных решений —
`(successes + 1) / (successes + failures + 2)`. Нет feedback — нет оценки.
В уже открытом обращении решение не повторяется. При исчерпании базы форма
появляется раньше пяти попыток.

## Что включено

- Все десять основных сущностей из ТЗ и технические `Job`, `RequestReceipt`, `AuditEvent`.
- Миграция Alembic и идемпотентный bootstrap.
- Асинхронный `POST /api/chat` → очередь в PostgreSQL → RAG → mock/real LLM → попытка.
- Feedback, успех, следующее решение, лимит пять, ранняя эскалация.
- Обязательная форма и сохранение mock-письма с полной историей.
- Идемпотентность POST, ограничения БД, проверка версии обращения при записи worker.
- JSON-логи без текстов сообщений/секретов и аудит бизнес-действий в БД.
- Поиск и статистика для проверки результата.

Worker один: PostgreSQL advisory lock запрещает параллельный запуск второго.
После перезапуска возвращает `RUNNING` задания в очередь; обработчики проверяют
версию обращения и уже сохранённый результат. Во время запроса к LLM транзакция
БД не открыта. Очередь сохраняется после рестарта контейнеров. Один worker также
означает, что медленный Qwen задерживает остальные задания — для первого среза это допустимо.

Nextcloud mock сохраняет сообщения как `DELIVERED_MOCK` в той же транзакции.
Почтовый mock сохраняет письмо как `SENT_MOCK`. Семантика повторов реальных сетевых
адаптеров, включая неопределённый результат отправки, будет отдельным этапом.

Аутентификация сейчас — один локальный Bearer token, связанный с `mock-employee`.
Он же имеет права администратора демо. Многопользовательская аутентификация,
роли и ACL документов пока отсутствуют; наружу сервис не публиковать.

**Ещё не реализованы:** реальные Nextcloud/Outlook, загрузка/переиндексация документов,
парсеры/OCR, вложения, семантические embeddings, фильтры оборудования, production RBAC,
политики хранения/backup и проверка создания заявки самим Итилиумом.
Endpoints upload/reindex не объявлены до появления их реализации.

## Проверки

```bash
docker compose exec api python -m pytest -q
docker compose exec api ruff check app tests scripts migrations
docker compose exec -e TEST_POSTGRES=1 api python -m pytest tests/test_postgres.py -q
docker compose exec api python scripts/demo.py
```

Основные тесты используют отдельную SQLite БД в temp-каталоге; они проверяют
FastAPI и workflow, но не заменяют PostgreSQL-проверки блокировок и pgvector.
`test_postgres.py` проверяет реальную миграцию на расхождение с моделями, vector query
и advisory lock; запускать с включёнными demo-знаниями. `scripts/demo.py` проверяет
API и отдельный worker на настоящем PostgreSQL через HTTP.

Без Docker, для разработчика с Python 3.12:

```bash
python -m venv .venv
# Активируйте .venv средствами вашей оболочки.
python -m pip install -e ".[test]"
python -m pytest -q
```

Необходимые зависимости перечислены в `pyproject.toml`; `uv.lock` фиксирует набор
для локальных проверок через `uv sync --extra test`.

## Основные файлы

```text
app/
  api/                 REST и локальная аутентификация
  core/                настройки и JSON-логи
  models/              SQLAlchemy-сущности
  schemas/             строгие Pydantic-контракты
  services/
    chat/              состояния, лимиты, идемпотентность
    rag/               локальный retrieval и интерфейс embeddings
    llm/               mock и Open WebUI
    nextcloud/         mock сообщений
    feedback/          статистика
    ticketing/         форма и письмо
    mail/              mock доставки
  workers/             обработка устойчивой очереди
  bootstrap.py         миграции и демо-данные
migrations/versions/   неизменяемая начальная миграция
sample_knowledge/      демонстрационные карточки
scripts/demo.py        полный HTTP-сценарий
tests/                workflow, LLM-контракт, PostgreSQL
compose.yaml
Dockerfile
.env.example
```
