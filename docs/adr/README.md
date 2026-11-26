# Architecture Decision Records

ADR фиксируют решения, на которых основана дорожная карта. Изменение принятого
решения оформляется новым ADR, который заменяет старый, а не молчаливой правкой
истории.

| ADR | Статус | Решение |
|---|---|---|
| [001](001-local-family-profiles.md) | Accepted | Локальные семейные профили без паролей |
| [002](002-sqlite-job-queue.md) | Accepted | SQLite-backed очередь и отдельный worker |
| [003](003-document-formats.md) | Accepted | Поэтапная поддержка безопасных форматов документов |
| [004](004-llm-privacy.md) | Accepted | Ollama по умолчанию, контролируемый внешний fallback |
| [005](005-account-storage.md) | Accepted | Неизменяемый account ID и account-scoped storage |
