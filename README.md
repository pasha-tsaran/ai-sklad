# AI Support

Open WebUI хранит утверждённые инструкции и обращается к моделям на Mac Studio.
Debian VM запускает API, worker и PostgreSQL, управляет обращениями, обратной связью,
лимитом пяти решений и отправкой заявки в поддержку. Модели на VM не загружаются.
Nextcloud Talk и SMTP подготовлены, по умолчанию работают в `mock`.
Почта только отправляет заявки; чтения ответов из почты нет.

## Знания и модели

```dotenv
LLM_BACKEND=openwebui
OPENWEBUI_BASE_URL=https://ai.collant.ru
OPENWEBUI_API_KEY=<ключ сервисной учётной записи>
OPENWEBUI_MODEL=ai-support
KNOWLEDGE_BACKEND=openwebui
OPENWEBUI_KNOWLEDGE_IDS=["id-утвержденной-коллекции"]
SEED_DEMO_KNOWLEDGE=false
NEXTCLOUD_BACKEND=mock
MAIL_BACKEND=mock
```

Профиль `ai-support` в Open WebUI должен использовать модель на Mac Studio, без
инструментов, веб-поиска и выполняющих действия pipelines. Сервисной учётной записи
выдайте доступ только к необходимым коллекциям и файлам. Модель меняется в профиле
или через `OPENWEBUI_MODEL`. Embeddings/обработку документов настраивают в Open WebUI
с внутренним сервером моделей; приложение настройки Open WebUI не меняет.

В Knowledge загружайте **один законченный вариант решения на файл**, включая
условия выполнения и ограничения по оборудованию. Членство в разрешённой коллекции
означает допуск к выдаче ботом, поэтому добавляйте только проверенные инструкции.

Backend передаёт разрешённые коллекции через `files` в `/api/chat/completions`.
Модель выбирает ID файла либо сообщает об отсутствии решения. Backend повторно
проверяет членство файла в коллекциях, получает полный извлечённый текст через API
файлов и выводит его дословно. Произвольные технические шаги модели не выводятся.
Выбор релевантного файла зависит от качества модели и RAG — нужен тест на вашем корпусе.

Файлы длиннее `KNOWLEDGE_MAX_CHARS` (20000 по умолчанию) не обрезаются и не выдаются:
разделите их на законченные инструкции. Картинки и вложения backend не пересылает;
проверьте качество извлечённого текста в Open WebUI. Неизвестный ID/ошибка API не
заменяются локальными демо-инструкциями. Пустой список коллекций означает отсутствие
знаний. При отсутствии решения бот запрашивает форму без расходования попыток.

Один файл не повторяется в обращении даже после редактирования. PostgreSQL хранит
снимки источников и SHA-256 для истории, не второй поисковый индекс. Старые таблицы
pgvector сохранены для совместимости. Снимки `openwebui-reference` не участвуют
в локальном поиске и не требуют embedding-модели на VM.

## Обновление Debian VM

Сохраните `.env` и резервную копию БД. Не выполняйте `down -v`.
Оставьте прежние `API_TOKEN` и `POSTGRES_PASSWORD`, добавьте настройки знаний.
Не копируйте `.env.example` поверх действующего `.env`.

```bash
cd /opt/ai-support
git pull --ff-only
sudo docker compose stop api worker
sudo docker compose build api worker migrate
sudo docker compose run --rm migrate
sudo docker compose up -d
sudo docker compose ps -a
curl -fsS http://127.0.0.1:8000/health/ready
```

Миграция `0002` добавляет адресацию Talk и очередь исходящих сообщений, сохраняя
обращения. `migrate` должен завершиться с кодом 0. API слушает `127.0.0.1:8000`, БД
наружу не опубликована. Readiness проверяет БД и показывает выбранные режимы,
**не проверяет доступность Open WebUI/Talk/SMTP**. Нужны отдельные сквозные проверки.

## Подключение Nextcloud Talk позже

Требуется capability `bots-v1` (Talk 17.1 / Nextcloud 27.1 и новее).
Администратор Nextcloud регистрирует бота через `occ talk:bot:install` и добавляет
его в разрешённые комнаты; параметры смотрите через `occ talk:bot:install --help`.
Webhook: `https://<внутренний-адрес-бота>/api/integrations/nextcloud/talk`.
Внутренний HTTPS reverse proxy должен проксировать на VM только этот путь.
Остальные `/api/*` и `/docs` административные, сотрудникам не публикуются.
REST `API_TOKEN` даёт доступ к тестовому пользователю и статистике, не заменяет SSO.

```dotenv
NEXTCLOUD_BACKEND=talk
NEXTCLOUD_BASE_URL=https://<ваш-nextcloud>
NEXTCLOUD_BOT_SECRET=<общий-секрет-не-менее-32-символов>
NEXTCLOUD_ALLOWED_ROOMS=["token-комнаты"]
```

Используйте закрытую комнату для отдельного сотрудника и поддержки: бот отвечает
в ту же комнату, все её участники видят сообщения. Разделение обращений по пользователю
не делает общую комнату приватной. Webhook проверяет HMAC и backend URL, принимает
зарегистрированных пользователей из разрешённых комнат, игнорирует ботов и служебные
события. Повтор message ID не создаёт второе обращение или feedback.

Для оценки скопируйте команду из ответа: `Помогло <код попытки>` или
`Не помогло <код попытки>`. Код защищает следующую попытку от запоздавшего ответа.
Для обращения к человеку отправьте `оператор`. По запросу формы отправьте:

```text
ФИО: Иванов Иван Иванович
Логин: ivanov
Компьютер: WS-001
Тема: Принтер не печатает
Описание: Кабель проверен
```

Описание необязательно. Вложения не скачиваются. После изменения `.env`:
`sudo docker compose up -d --force-recreate api worker`.

## Подключение исходящей почты позже

```dotenv
MAIL_BACKEND=smtp
SUPPORT_EMAIL=<адрес-службы-поддержки>
SMTP_HOST=<внутренний-smtp>
SMTP_PORT=587
SMTP_SECURITY=starttls
SMTP_FROM=<адрес-бота>
SMTP_USERNAME=<логин-если-нужен>
SMTP_PASSWORD=<пароль-если-нужен>
```

Для implicit TLS используйте `SMTP_SECURITY=ssl` и порт сервера (обычно 465).
Сертификаты проверяются. Получатель задаётся конфигурацией, не моделью/пользователем.
Письмо содержит форму, полную историю, результаты попыток и summary LLM;
ошибка summary не блокирует отправку. `SENT` означает приём SMTP-сервером,
не создание тикета в сторонней системе.

## Неопределённая доставка

Talk/SMTP сохраняют `SENDING` до сетевого запроса. Тайм-аут или рестарт во время
отправки дают `UNKNOWN`: автоматического повтора нет, сообщение могло быть принято.
В Talk последующие сообщения этой комнаты ждут разрешения неопределённости;
остальные комнаты продолжают работу. Список без персональных данных:

```bash
sudo docker compose exec worker python -m app.deliveries mail
sudo docker compose exec worker python -m app.deliveries talk
```

Сверьте доставку на почтовом сервере/в Talk. Если подтверждена доставка:
`sudo docker compose exec worker python -m app.deliveries mail --id <ID> --action mark-sent`.
Если подтверждено отсутствие доставки, используйте `--action retry`.
Для Talk замените `mail` на `talk`. Действие записывается в аудит.
Повтор без проверки может создать дубль.

## Сеть без общего интернета

Для сборки нужны registry образов и источники Python-зависимостей. В рабочем режиме
VM нужны внутренние DNS/NTP, Open WebUI HTTPS, Nextcloud HTTPS и SMTP.
Nextcloud должен видеть webhook. Open WebUI должен видеть модели на Mac Studio.
Прямой доступ VM к Mac Studio при этой схеме не нужен. Закрытие общего интернета
не должно закрывать эти внутренние маршруты.

## Проверки

```bash
sudo docker compose exec api python -m pytest -q
sudo docker compose exec api ruff check app tests scripts migrations
sudo docker compose exec -e TEST_POSTGRES=1 api python -m pytest tests/test_postgres.py::test_migrations_match_models tests/test_postgres.py::test_worker_singleton_lock -q
```

Основные тесты используют временную SQLite и подменённые HTTP/SMTP-клиенты, реальные
сообщения не отправляют. PostgreSQL-проверки нужно отдельно запустить на VM.
Реальные версии Open WebUI/Talk и качество выбора инструкций проверяются после настройки.

Учебный сценарий отдельно: `LLM_BACKEND=mock`, `KNOWLEDGE_BACKEND=demo`,
`SEED_DEMO_KNOWLEDGE=true`, интеграции `mock`, затем `python scripts/demo.py`.
Он создаёт историю и меняет статистику. Демо-инструкции не корпоративные.
Контракты REST находятся в `/docs`; POST требует Bearer `API_TOKEN` и `Idempotency-Key`.

Протоколы: [Open WebUI API](https://docs.openwebui.com/reference/api-endpoints/),
[Nextcloud Talk bots](https://nextcloud-talk.readthedocs.io/en/latest/bots/).
