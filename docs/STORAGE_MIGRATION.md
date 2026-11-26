# Миграция документов и досье в UUID-каталоги

Новый layout хранит данные профилей в:

```text
data/accounts/<account_id>/
  cv/
  applications/
  tmp/
```

До применения миграции приложение продолжает работать с `cv/<имя>/` и общим
`applications/`. Переключение на новый layout происходит только после успешной
записи `data/storage-layout.json`.

## Порядок выполнения

1. Остановить web UI, чтобы старый процесс не создавал artifacts во время
   копирования.
2. Выполнить read-only проверку:

   ```bash
   .venv/bin/cv-agent migrate-storage
   ```

3. Проверить разделы `conflicts`, `unmapped_document_entries` и
   `legacy_application_entries` в JSON-отчёте. Первые два должны быть пустыми.
   Старые dossiers ожидаемо остаются неоднозначными: миграция их никому не
   приписывает.
4. Применить миграцию:

   ```bash
   .venv/bin/cv-agent migrate-storage --apply
   ```

Команда сначала сохраняет SQLite, весь `cv/` и `applications/` в
`data/backups/storage-<timestamp>/`, затем копирует документы, создаёт отдельные
каталоги profiles, записывает `migration-report.json` и только после этого
атомарно включает новый layout. Исходные `cv/<имя>/` намеренно сохраняются до
ручной проверки. Повторный `--apply` ничего не копирует и не создаёт новый
backup.

## Восстановление

1. Остановить web UI.
2. Переместить `data/storage-layout.json` в безопасное место. Без marker
   приложение снова читает legacy-каталоги, которые миграция не удаляла.
3. При необходимости восстановить `data/applications.db`, `cv/` и
   `applications/` из указанного в отчёте `backup_dir`.
4. Запустить dry-run ещё раз и проверить отчёт до повторного `--apply`.

Не удаляйте legacy-каталоги и backup, пока документы обоих профилей и создание
нового dossier не проверены вручную.
