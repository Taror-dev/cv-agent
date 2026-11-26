# Архитектура CV-Agent

Границы подсистем, поток данных и ключевые инварианты. Решения с обоснованием
— в `docs/adr/`; эксплуатация — в `docs/OPERATIONS.md`.

## Обзор

Локальное приложение (bind на `127.0.0.1`) для полуавтоматической подачи
заявок: поиск вакансий в четырёх источниках, LLM-оценка соответствия
кандидату, генерация Anschreiben и сборка досье для ручной отправки.
Мультипрофильность семейная: профили разделяют данные, но не защищают их друг
от друга паролями (ADR-001).

## Слои и модули

- Точки входа: `cli.py` (команды), `webapp.py` (Flask UI на русском),
  `job_worker.py` (отдельный процесс фоновых заданий).
- Pipeline: `search.py` (адаптеры источников, дедупликация), `score.py`
  (оценка с исключениями), `write.py` / `apply.py` (Anschreiben, досье).
- Хранилище: `store.py` — единственный шлюз к SQLite (schema v10):
  аккаунты, candidate_profiles, vacancies, jobs, documents, artifacts,
  llm_rate_limits; все дочерние таблицы каскадируют от `accounts`.
- Инфраструктура: `documents.py` (реестр экстракторов PDF/TXT/MD),
  `storage.py` (UUID-layout и миграция каталогов), `paths.py`
  (containment-check путей), `resource_limits.py` (flock-пулы лимитов),
  `security.py` (redaction), `config.py` (YAML, `CV_AGENT_DATA_DIR`).

## Поток данных

1. **Search.** Ключевые слова аккаунта нормализуются и разбиваются на запросы
   per keyword x локация (`--location` принимает список через запятую).
   Каждый источник реализует контракт `SearchAdapter`
   (`queries/search/detail/exists/refresh`) и изолируется в `run_adapter`:
   валидация вакансий, дедупликация refnr между источниками, локальный
   geo-filter для get-in-it, ограниченный retry/backoff, итог per-source.
   Upsert обновляет source-поля без потери score/application status.
2. **Description cache.** `ensure_description` скачивает описание до
   исключений/scoring во всех CLI/web-путях, сохраняет SHA-256 и время; TTL
   из `criteria.yaml`; недоступные/unsupported записи не скачиваются.
3. **Score.** Prefilter по исключениям (title/company/description, слова или
   regex) до вызова LLM; структурный ответ модели (score/reasons/missing) с
   одним repair-attempt; невалидный формат сохраняется как `score_error`, а
   не ноль. Повторный расчёт только при изменении входов (хэши кандидата,
   вакансии, описания, prompt version); явный `score --force`.
4. **Write/Apply.** Промпт письма собирается из документов текущего профиля
   (purpose-бюджеты символов, CV первым) и вакансии;
   `validate_letter_identity` проверяет имя профиля; досье собирается во
   временном каталоге и атомарно публикуется версиями (`_v2`, ...); footer
   date CV заменяется визуально и в text layer. Статус `prepared` после
   сборки; `applied` только по явному подтверждению (`confirm-applied`).

## Фоновые задания

- Очередь в SQLite (`jobs`): состояния `queued/running/waiting_rate_limit/
  succeeded/failed/cancelled/timed_out/interrupted`; шаги с кодами,
  ограниченный лог хвоста, метрики network/LLM запросов и ожидания.
- Каждый прогон выполняет отдельный процесс `python -m cv_agent.job_worker`
  в собственной группе процессов; web ведёт реестр воркеров и отменяет прогон
  через SIGTERM -> SIGKILL группы -> reap (статус `cancelled`).
- Startup recovery помечает зависшие active jobs как `interrupted`;
  аварийный выход воркера — `failed` (reaper).
- Структурный журнал событий заданий: `data/logs/jobs.jsonl`
  (`job_started`/`step_finished`/`job_finished` с run_id, account_id,
  длительностью); ошибки журналирования не влияют на выполнение.

## LLM-слой

- OpenAI-совместимый клиент; основной провайдер — локальная Ollama,
  внешние провайдеры используют per-account ключ из SQLite. При выборе
  внешней модели или автоматическом fallback дополнительного подтверждения
  передачи документов и вакансий нет.
- Endpoint allowlist на сервере: cookie UI хранит только option-id;
  base_url/api_key_env от клиента не принимаются; SSRF-проверки схем/хостов
  до создания клиента.
- Независимые счётчики обычных и 429-ретраев; учёт `Retry-After`;
  per-profile/provider cooldown в `llm_rate_limits`; межпроцессные пулы
  лимитов (`limits.*` в criteria.yaml) через flock.
- Версионированные промпты с untrusted-delimiters вокруг документов/вакансий;
  ключи никогда не попадают в subprocess env — CLI загружает ключ профиля
  прямо из SQLite.

## Границы безопасности

- Единственный resolver путей `account_path` (containment-check, запрет
  symlink); uploads атомарны, 0600/0700, квоты на файл/аккаунт.
- Сессия подписана, хранит только UUID аккаунта; cookie HttpOnly +
  SameSite=Strict (+Secure при HTTPS); logout через POST.
- Команды воркера проходят server-side allowlist (`_validated_job_steps`);
  request-specific переменные не наследуются CLI-потомком из os.environ.
- Подробности и threat model — в `docs/SECURITY.md`.
