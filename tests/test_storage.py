from pathlib import Path
import stat

import pytest

from cv_agent.paths import account_path
from cv_agent.storage import (
    apply_storage_migration,
    applications_dir,
    documents_dir,
    plan_storage_migration,
    storage_layout_active,
)
from cv_agent.store import Store


def _fixture(tmp_path):
    data = tmp_path / "data"
    cv = tmp_path / "cv"
    applications = tmp_path / "applications"
    store = Store(data / "applications.db")
    store.create_account("Alice")
    for name, content in (("Илья", b"ilia"), ("Alice", b"alice")):
        folder = cv / name
        folder.mkdir(parents=True)
        (folder / "cv.txt").write_bytes(content)
    applications.mkdir()
    (applications / "Legacy_Dossier").mkdir()
    return store, data, cv, applications


def test_storage_migration_dry_run_is_read_only(tmp_path):
    store, data, cv, applications = _fixture(tmp_path)
    try:
        report = plan_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=cv,
            legacy_applications_root=applications,
        )
        assert len(report["documents"]) == 2
        assert report["conflicts"] == []
        assert report["unmapped_document_entries"] == []
        assert report["legacy_application_entries"] == [
            str(applications / "Legacy_Dossier")
        ]
        assert not storage_layout_active(data)
        assert not (data / "accounts").exists()
    finally:
        store.close()


def test_storage_migration_apply_backup_copy_and_idempotency(tmp_path):
    store, data, cv, applications = _fixture(tmp_path)
    try:
        report = apply_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=cv,
            legacy_applications_root=applications,
        )
        assert storage_layout_active(data)
        backup = Path(report["backup_dir"])
        assert (backup / "applications.db").is_file()
        assert (backup / "cv" / "Илья" / "cv.txt").is_file()
        assert (backup / "applications" / "Legacy_Dossier").is_dir()
        expected = {"Илья": b"ilia", "Alice": b"alice"}
        for account_id, name in store.list_account_records():
            target = account_path(data, account_id, "cv") / "cv.txt"
            assert target.read_bytes() == expected[name]
            assert account_path(data, account_id, "applications").is_dir()
            assert account_path(data, account_id, "tmp").is_dir()
            for folder in (
                data / "accounts",
                account_path(data, account_id, "cv").parent,
                account_path(data, account_id, "cv"),
                account_path(data, account_id, "applications"),
                account_path(data, account_id, "tmp"),
            ):
                assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert (cv / "Илья" / "cv.txt").is_file()
        assert (cv / "Alice" / "cv.txt").is_file()

        backup_count = len(list((data / "backups").glob("storage-*")))
        repeated = apply_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=cv,
            legacy_applications_root=applications,
        )
        assert repeated["already_active"] is True
        assert len(list((data / "backups").glob("storage-*"))) == backup_count
    finally:
        store.close()


def test_storage_migration_conflict_does_not_activate(tmp_path):
    store, data, cv, applications = _fixture(tmp_path)
    try:
        account_id = store.account_id("Alice")
        target = account_path(data, account_id, "cv") / "cv.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"different")
        report = plan_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=cv,
            legacy_applications_root=applications,
        )
        assert len(report["conflicts"]) == 1
        with pytest.raises(RuntimeError, match="конфликты"):
            apply_storage_migration(
                store,
                data_dir=data,
                legacy_cv_root=cv,
                legacy_applications_root=applications,
            )
        assert not storage_layout_active(data)
    finally:
        store.close()


def test_active_layout_routes_each_profile_separately(tmp_path):
    store, data, cv, applications = _fixture(tmp_path)
    try:
        apply_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=cv,
            legacy_applications_root=applications,
        )
        ilia_docs = documents_dir("Илья", cv, data_dir=data, store=store)
        alice_docs = documents_dir("Alice", cv, data_dir=data, store=store)
        assert ilia_docs != alice_docs
        assert ilia_docs.name == alice_docs.name == "cv"
        assert ilia_docs.is_dir() and alice_docs.is_dir()
        assert applications_dir(
            "Илья", applications, data_dir=data, store=store
        ) != applications_dir(
            "Alice", applications, data_dir=data, store=store
        )
    finally:
        store.close()
