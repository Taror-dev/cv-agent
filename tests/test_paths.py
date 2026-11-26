from pathlib import Path

import pytest

from cv_agent.paths import (
    account_directory,
    account_path,
    ensure_account_path,
    safe_filename,
    valid_account_name,
)


@pytest.mark.parametrize("name", [".", "..", "../Илья", "Илья/другой", "x\\y", ""])
def test_invalid_account_names(name):
    assert valid_account_name(name) is False


@pytest.mark.parametrize("name", ["Илья", "Катя", "Family Profile", "A-1_test"])
def test_valid_account_names(name):
    assert valid_account_name(name) is True


def test_account_directory_stays_inside_root(tmp_path):
    root = tmp_path / "cv"
    root.mkdir()
    assert account_directory(root, "Илья") == root / "Илья"
    with pytest.raises(ValueError):
        account_directory(root, "..")


def test_account_directory_rejects_symlink_escape(tmp_path):
    root = tmp_path / "cv"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "Илья").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        account_directory(root, "Илья")


@pytest.mark.parametrize("name", ["../cv.pdf", "/tmp/cv.pdf", "x\\cv.pdf", ".", ".."])
def test_safe_filename_rejects_paths(name):
    with pytest.raises(ValueError):
        safe_filename(name)


def test_safe_filename_accepts_basename():
    assert safe_filename("Lebenslauf 2026.pdf") == "Lebenslauf 2026.pdf"


def test_account_path_uses_uuid_and_kind(tmp_path):
    account_id = "11111111-1111-4111-8111-111111111111"
    assert account_path(tmp_path, account_id, "cv") == (
        tmp_path / "accounts" / account_id / "cv"
    ).resolve()
    with pytest.raises(ValueError):
        account_path(tmp_path, "not-a-uuid", "cv")
    with pytest.raises(ValueError):
        account_path(tmp_path, account_id, "unknown")


def test_account_path_rejects_symlink_profile_escape(tmp_path):
    account_id = "11111111-1111-4111-8111-111111111111"
    accounts = tmp_path / "accounts"
    outside = tmp_path / "outside"
    accounts.mkdir()
    outside.mkdir()
    (accounts / account_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        account_path(tmp_path, account_id, "cv")


def test_account_path_rejects_symlink_accounts_root_escape(tmp_path):
    account_id = "11111111-1111-4111-8111-111111111111"
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "accounts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="символической"):
        account_path(tmp_path, account_id, "cv")


def test_ensure_account_path_forces_private_parent_permissions(tmp_path):
    import stat

    account_id = "11111111-1111-4111-8111-111111111111"
    target = ensure_account_path(tmp_path, account_id, "cv")
    for folder in (target.parents[1], target.parent, target):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
