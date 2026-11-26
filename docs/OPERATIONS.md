# Эксплуатация CV-Agent

Практический runbook для локальной эксплуатации (дополняет README). Составлен
в рамках этапа 7 дорожной карты.

## Каталоги и данные

| Путь | Содержимое |
|---|---|
| `data/applications.db` | SQLite: аккаунты, вакансии, jobs, документы, artifacts, rate limits |
| `data/accounts/<uuid>/` | Документы (`cv/`), досье (`applications/`), временные файлы каждого профиля |
| `data/backups/` | Backup'ы миграций и экспорт-архивы удалённых профилей |
| `cv/<имя>/` | Legacy-каталог документов (после миграции — только исходники) |

Каталог данных можно перенести переменной `CV_AGENT_DATA_DIR` (этап 7.1);
по умолчанию используется каталог проекта.

## Backup

### Автоматические копии
- Перед каждой миграцией схемы Store создаёт консистентную копию DB:
  `data/backups/applications-pre-migration-<stamp>.db`.
- `migrate-storage --apply` дополнительно копирует legacy `cv/` и
  `applications/` в `data/backups/storage-<stamp>/` и пишет отчёт
  `migration-report.json`.

### Ручной backup (рекомендуемая периодичность — перед обновлениями)
```bash
sqlite3 data/applications.db ".backup 'data/backups/manual-$(date +%Y%m%dT%H%M%S).db'"
cp -a data/accounts "data/backups/accounts-$(date +%Y%m%dT%H%M%S)"
```

## Restore

1. Остановить приложение (`Ctrl+C` web / worker завершится сам).
2. Восстановить DB из копии:
   ```bash
   sqlite3 data/backups/manual-XXXX.db "PRAGMA integrity_check;"
   cp data/backups/manual-XXXX.db data/applications.db
   ```
3. При необходимости восстановить каталоги профилей из
   `data/backups/accounts-XXXX/` в `data/accounts/` (права 0700 сохраняются).
4. Запустить `.venv/bin/cv-agent status` — проверка чтения DB без сети.
5. Полная проверка пустого каталога данных (сценарий этапа 7): скопировать
   проект в новый каталог, задать `CV_AGENT_DATA_DIR`, убедиться, что CLI/web
   стартуют и создают схему заново.

## Retention

- Архивированные вакансии: `.venv/bin/cv-agent purge-archived --confirm
  [--older-than-days N]` (applied не удаляются никогда).
- Завершённые фоновые задания с их логами: `.venv/bin/cv-agent prune-jobs
  --confirm [--older-than-days N]`.

## Health check

`GET http://127.0.0.1:8000/health` — JSON без секретов: доступность DB,
версия схемы (фактическая и ожидаемая), число активных заданий, живость
Ollama. HTTP 503 при недоступной DB. Эндпоинт доступен без входа в профиль.

## Зависимости и аудит

- Проверенные версии зафиксированы в `constraints.txt`; `run.sh`
  переустанавливает venv автоматически, если изменились `pyproject.toml` или
  `constraints.txt`.
- Периодический security-аудит (вручную):
  ```bash
  .venv/bin/pip list --outdated
  .venv/bin/pip-audit -r constraints.txt   # после pip install pip-audit
  ```
  Обновлять по одной зависимости, прогоняя полный офлайн-набор тестов
  (`.venv/bin/pytest`) после каждой.

## WSGI / публикация в сеть

Приложение предназначено для loopback (ADR-001). Production WSGI-сервер и
proxy/TLS потребуются только при намеренном открытии доступа; до этого — см.
условия в ADR-001 и ROADMAP (пункт этапа 7 оставлен открытым сознательно).

## Диагностика типовых ситуаций

| Симптом | Действие |
|---|---|
| Jobs остались `running` после падения web | При старте web они помечаются `interrupted`; повторите прогон |
| Источник вернул 0 результатов | Посмотреть «Suchplan» и per-source summary в логе прогона; частичные ошибки источников видны как предупреждение шага |
| LLM недоступна | `/health` показывает состояние Ollama; настроенный fallback на внешний провайдер выполняется автоматически без дополнительного подтверждения |

## Runbook

### 1. Parser failure источника (StepStone/LinkedIn/get-in-it)

Симптом: источник стабильно отдаёт 0 результатов или растёт `fehler` в
per-source summary, тогда как BA работает.

1. Проверить, не изменилась ли внешняя разметка: открыть страницу источника
   руками, сравнить с парсером (`search.py`: `_parse_embedded`,
   `_parse_links`, `_parse`, `_parse_detail`).
2. Воспроизвести офлайн: сохранить HTML страницы в fixture и прогнать
   соответствующий тест (`tests/test_search_new.py`).
3. Починить селектор/JSON-путь; добавить fixture как golden-file.
4. До починки источник можно считать degraded — остальные продолжают
   работать независимо (изоляция ошибок per-query в `run_adapter`).

### 2. Stuck job

Симптом: job долго в `running`/`waiting_rate_limit`, прогресса нет.

1. `/health` — жив ли worker и DB; `data/logs/jobs.jsonl` — последние события
   по `run_id`.
2. Отменить прогон кнопкой на странице прогона (terminate -> kill группы ->
   reap, статус `cancelled`) и запустить заново.
3. Если web-процесс был убит жёстко и job остался активным — рестарт web
   помечает его `interrupted` автоматически.
4. Зависший CLI-потомок невозможен без отмены: wall-clock deadline делает
   terminate -> kill -> reap внутри шага.

### 3. Миграции схемы

1. Перед изменением схемы Store сам создаёт консистентный backup
   (`data/backups/applications-pre-migration-*.db`).
2. Применение автоматическое при первом открытии DB новой версией кода;
   миграции идемпотентны — повторный запуск ничего не дублирует.
3. Откат: остановить приложение, вернуть backup из `data/backups/`
   (см. «Restore»), запустить предыдущую версию кода.
4. Storage-layout (`migrate-storage --apply`) имеет собственный dry-run,
   отчёт и marker-файл; детали в `docs/STORAGE_MIGRATION.md`.

### 4. Rate limits внешних LLM (Gemini/Groq)

Симптом: статус прогона `waiting_rate_limit`, счётчики LLM-ожидания растут.

1. Это нормальное поведение: клиент уважает `Retry-After`/квоты из заголовков
   (лимиты показываются на странице профиля).
2. Коoldown хранится per profile/provider (`llm_rate_limits`) — другие
   профили и Ollama не блокируются.
3. Чтобы дождаться окончания — просто оставить прогон работать; чтобы
   прервать — cancel. Повторный запуск продолжит только неоценённые вакансии
   (`score --missing` semantics для новых входов).
4. Долгосрочно: снизить объём (`--max-keywords`, `--per-query-limit`),
   переключиться на Ollama либо другой ключ провайдера в профиле.

### 5. Backup/restore

См. разделы «Backup» и «Restore» выше; восстановление на пустом каталоге
данных покрыто автотестом `tests/test_operations.py`.
