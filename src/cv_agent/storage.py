"""UUID layout пользовательских документов и досье, включая safe migration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .paths import account_directory, account_path, ensure_account_path

STORAGE_LAYOUT_VERSION = 1
STORAGE_MARKER_NAME = "storage-layout.json"


def _marker_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir or config.DATA_DIR) / STORAGE_MARKER_NAME


def storage_layout_active(data_dir: Path | None = None) -> bool:
    marker = _marker_path(data_dir)
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return data.get("version") == STORAGE_LAYOUT_VERSION


def _account_id(user: str, store=None) -> str:
    if store is None:
        from .store import Store

        store = Store(user=user)
        close = True
    else:
        close = False
    try:
        value = store.account_id(user)
    finally:
        if close:
            store.close()
    if not value:
        raise ValueError(f"Профиль «{user}» не существует")
    return value


def documents_dir(
    user: str,
    legacy_root: Path,
    *,
    data_dir: Path | None = None,
    store=None,
    legacy_root_fallback: bool = True,
) -> Path:
    """Активный каталог CV: legacy до marker, UUID layout после marker."""
    actual_data_dir = Path(data_dir or config.DATA_DIR)
    if storage_layout_active(actual_data_dir):
        return ensure_account_path(
            actual_data_dir, _account_id(user, store), "cv"
        )
    legacy = account_directory(Path(legacy_root), user)
    if legacy.is_dir():
        return legacy
    if legacy_root_fallback and user == config.DEFAULT_USER:
        return Path(legacy_root)
    return legacy


def applications_dir(
    user: str,
    legacy_root: Path,
    *,
    data_dir: Path | None = None,
    store=None,
) -> Path:
    """Каталог новых artifacts текущего профиля."""
    actual_data_dir = Path(data_dir or config.DATA_DIR)
    if storage_layout_active(actual_data_dir):
        return ensure_account_path(
            actual_data_dir, _account_id(user, store), "applications"
        )
    return Path(legacy_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_storage_migration(
    store,
    *,
    data_dir: Path | None = None,
    legacy_cv_root: Path | None = None,
    legacy_applications_root: Path | None = None,
) -> dict:
    """Read-only план копирования legacy CV и список неоднозначных artifacts."""
    actual_data = Path(data_dir or config.DATA_DIR)
    cv_root = Path(legacy_cv_root or (config.ROOT / "cv"))
    applications_root = Path(
        legacy_applications_root or config.APPLICATIONS_DIR
    )
    records = store.list_account_records()
    known_names = {name for _account_id_value, name in records}
    report = {
        "version": STORAGE_LAYOUT_VERSION,
        "already_active": storage_layout_active(actual_data),
        "documents": [],
        "skipped_identical": [],
        "conflicts": [],
        "unmapped_document_entries": [],
        "legacy_application_entries": [],
        "accounts": [
            {"account_id": account_id_value, "name": name}
            for account_id_value, name in records
        ],
    }
    if report["already_active"]:
        return report

    for account_id_value, name in records:
        source_dir = account_directory(cv_root, name)
        target_dir = account_path(actual_data, account_id_value, "cv")
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.iterdir(), key=lambda item: item.name):
            if not source.is_file() or source.is_symlink():
                report["unmapped_document_entries"].append(str(source))
                continue
            target = target_dir / source.name
            item = {
                "account_id": account_id_value,
                "profile": name,
                "source": str(source),
                "target": str(target),
                "size": source.stat().st_size,
                "sha256": _sha256(source),
            }
            if target.exists():
                if (
                    target.is_file()
                    and not target.is_symlink()
                    and target.stat().st_size == source.stat().st_size
                    and _sha256(target) == item["sha256"]
                ):
                    report["skipped_identical"].append(item)
                else:
                    report["conflicts"].append(item)
            else:
                report["documents"].append(item)

    if cv_root.is_dir():
        for entry in sorted(cv_root.iterdir(), key=lambda item: item.name):
            if entry.name not in known_names:
                report["unmapped_document_entries"].append(str(entry))

    if applications_root.is_dir():
        report["legacy_application_entries"] = [
            str(entry)
            for entry in sorted(applications_root.iterdir(), key=lambda item: item.name)
        ]
    return report


def _copy_tree_if_present(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}-",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temp_name = stream.name
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_storage_migration(
    store,
    *,
    data_dir: Path | None = None,
    legacy_cv_root: Path | None = None,
    legacy_applications_root: Path | None = None,
) -> dict:
    """Backup → copy CV → report → marker. Исходники намеренно сохраняются."""
    actual_data = Path(data_dir or config.DATA_DIR)
    cv_root = Path(legacy_cv_root or (config.ROOT / "cv"))
    applications_root = Path(
        legacy_applications_root or config.APPLICATIONS_DIR
    )
    report = plan_storage_migration(
        store,
        data_dir=actual_data,
        legacy_cv_root=cv_root,
        legacy_applications_root=applications_root,
    )
    if report["already_active"]:
        return report
    if report["conflicts"] or report["unmapped_document_entries"]:
        raise RuntimeError(
            "Миграция storage остановлена: сначала разрешите конфликты/непривязанные документы"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = actual_data / "backups" / f"storage-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    database_backup = backup_dir / "applications.db"
    backup_conn = sqlite3.connect(database_backup)
    try:
        store.conn.backup(backup_conn)
    finally:
        backup_conn.close()
    database_backup.chmod(0o600)
    _copy_tree_if_present(cv_root, backup_dir / "cv")
    _copy_tree_if_present(applications_root, backup_dir / "applications")

    for item in report["documents"]:
        source = Path(item["source"])
        target = Path(item["target"])
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(source, target)
        target.chmod(0o600)
    for account in report["accounts"]:
        for kind in ("cv", "applications", "tmp"):
            ensure_account_path(actual_data, account["account_id"], kind)

    report.update(
        {
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "backup_dir": str(backup_dir),
        }
    )
    report_path = backup_dir / "migration-report.json"
    _atomic_json(report_path, report)
    _atomic_json(
        _marker_path(actual_data),
        {
            "version": STORAGE_LAYOUT_VERSION,
            "applied_at": report["applied_at"],
            "report": str(report_path),
        },
    )
    return report


def export_account_archive(
    store,
    user: str,
    *,
    data_dir: Path | None = None,
) -> dict:
    """Архив-экспорт профиля перед удалением: DB backup + копии документов.

    Политика удаления (этап 2.1 дорожной карты): archive → export → явное
    подтверждённое удаление. Этот шаг делает первые два; удаление выполняет
    вызывающий код после проверки отчёта. Копируются и UUID-каталоги, и
    legacy `cv/<имя>` (если существует).
    """
    actual_data = Path(data_dir or config.DATA_DIR)
    account_id = store.account_id(user)
    if not account_id:
        raise ValueError(f"Профиль «{user}» не существует")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    export_dir = actual_data / "backups" / f"profile-export-{stamp}"
    export_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    database_backup = export_dir / "applications.db"
    backup_conn = sqlite3.connect(database_backup)
    try:
        store.conn.backup(backup_conn)
    finally:
        backup_conn.close()
    database_backup.chmod(0o600)

    sources: list[tuple[str, Path]] = []
    for kind in ("cv", "applications", "tmp"):
        try:
            folder = account_path(actual_data, account_id, kind)
        except ValueError:
            continue
        if folder.is_dir():
            sources.append((kind, folder))
    try:
        legacy_cv = account_directory(Path(config.ROOT) / "cv", user)
    except ValueError:
        legacy_cv = None
    if legacy_cv is not None and legacy_cv.is_dir():
        sources.append(("legacy-cv", legacy_cv))

    documents: list[dict] = []
    for label, folder in sources:
        _copy_tree_if_present(folder, export_dir / label)
        for entry in sorted(folder.iterdir(), key=lambda item: item.name):
            if entry.is_file() and not entry.is_symlink():
                documents.append(
                    {
                        "kind": label,
                        "filename": entry.name,
                        "size_bytes": entry.stat().st_size,
                    }
                )

    counts = {}
    for table in ("vacancies", "jobs", "documents", "artifacts", "llm_rate_limits"):
        counts[table] = store.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE account_id=?", (account_id,)
        ).fetchone()[0]

    report = {
        "account_id": account_id,
        "profile": user,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir),
        "database_backup": str(database_backup),
        "counts": counts,
        "documents": documents,
    }
    _atomic_json(export_dir / "profile-export.json", report)
    return report


def remove_account_storage(
    account_id: str,
    user: str = "",
    *,
    data_dir: Path | None = None,
) -> bool:
    """Удаляет каталоги профиля с containment-проверкой. True, если что-то было.

    Убирает UUID-каталог `data/accounts/<account_id>/` и, если передано имя,
    legacy `cv/<имя>`. Возвращает True только при фактическом удалении.
    """
    actual_data = Path(data_dir or config.DATA_DIR)
    removed = False
    try:
        profile_root = account_path(actual_data, account_id, "cv").parent
    except ValueError:
        profile_root = None
    if (
        profile_root is not None
        and profile_root.is_dir()
        and not profile_root.is_symlink()
    ):
        shutil.rmtree(profile_root)
        removed = True
    if user:
        try:
            legacy_cv = account_directory(Path(config.ROOT) / "cv", user)
        except ValueError:
            legacy_cv = None
        if legacy_cv is not None and legacy_cv.is_dir() and not legacy_cv.is_symlink():
            shutil.rmtree(legacy_cv)
            removed = True
    return removed
