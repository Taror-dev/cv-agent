# ADR-002: SQLite-backed очередь и отдельный worker

- Статус: Accepted
- Дата: 20.08.2026

## Контекст

Сейчас pipeline хранится в module-level словаре Flask и выполняется daemon
thread. Состояние исчезает при рестарте, timeout streaming subprocess не
ограничивает чтение stdout, а глобальный active run смешивает профили.

Проект локальный и не требует распределённой инфраструктуры Redis/Celery.

## Решение

- Хранить задания и их состояние в SQLite.
- Выполнять задания отдельным локальным worker-процессом.
- Job содержит `account_id`, тип команды, валидированные параметры, timestamps,
  progress, ограниченный лог и итоговую ошибку.
- Поддержать состояния `queued`, `running`, `waiting_rate_limit`, `succeeded`,
  `failed`, `cancelled`, `timed_out`, `interrupted`.
- На старте worker восстанавливает задания, оставшиеся в `running` после сбоя.
- Для первой версии запускать отдельный локальный worker-процесс на job и
  ограничивать фактический fan-out межпроцессными flock pools; не добавлять
  Redis, пока измерения не покажут необходимость.
- Общий HTTP pool и отдельные pools по LLM backend сохраняют в job количество
  запросов и время ожидания, чтобы concurrency limits можно было настраивать по
  фактическим данным.
- Subprocess получает wall-clock deadline, terminate/kill и обязательный reap.

## Последствия

- Job status переживает перезапуск web UI.
- Runs естественно разделяются по `account_id`.
- SQLite остаётся единственной операционной зависимостью.
- Потребуются schema migration, отдельная команда запуска worker и тесты
  восстановления/отмены.
