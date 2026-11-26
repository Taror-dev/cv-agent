import hashlib
import pytest

from cv_agent.models import Vacancy
from cv_agent.store import Store
from uuid import UUID


def test_add_and_top(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add(Vacancy("1", "AI Engineer", "ACME", "Wangen", "http://x", "ba", description="Python RAG"))
    s.update_score("1", 85)
    top = s.top(60)
    assert len(top) == 1
    assert top[0].score == 85
    s.close()


def test_list_all_includes_location_without_shifting_existing_columns(tmp_path):
    s = Store(tmp_path / "t.db")
    try:
        s.add(
            Vacancy(
                "1",
                "AI Engineer",
                "ACME",
                "Wangen im Allgäu, Bayern, Deutschland",
                "http://x",
                "ba",
            )
        )
        row = s.list_all()[0]
        assert row[:8] == (
            "1",
            "AI Engineer",
            "ACME",
            "ba",
            None,
            "new",
            "unscored",
            "",
        )
        assert row[8] == "Wangen im Allgäu, Bayern, Deutschland"
    finally:
        s.close()


def test_vacancies_for_dedup_is_account_scoped(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        ilia.add(Vacancy("same", "Python", "ACME", "Wangen", "http://x", "ba"))
        alice.add(Vacancy("same", "Java", "Beta", "Berlin", "http://y", "ba"))
        assert [vacancy.title for vacancy in ilia.vacancies_for_dedup()] == ["Python"]
        assert [vacancy.title for vacancy in alice.vacancies_for_dedup()] == ["Java"]
    finally:
        ilia.close()
        alice.close()


def test_score_error_is_distinct_and_artifact_metadata_is_account_scoped(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        ilia.add(Vacancy("1", "AI", "ACME", "Wangen", "u", "ba"))
        ilia.mark_score_error("1")
        error_details = ilia.score_details("1")
        assert error_details["score"] is None
        assert error_details["state"] == "score_error"
        assert error_details["reasons"] == []
        assert error_details["missing"] == []
        assert error_details["skip_reason"] == "format_error"
        ilia.update_score_details(
            "1",
            73.0,
            ["Python"],
            ["Kubernetes"],
            provider="ollama",
            model="qwen",
            prompt_version="scoring.v1",
            candidate_hash="a" * 64,
            vacancy_hash="b" * 64,
            description_hash="c" * 64,
        )
        score_details = ilia.score_details("1")
        assert score_details["score"] == 73.0
        assert score_details["state"] == "scored"
        assert score_details["reasons"] == ["Python"]
        assert score_details["missing"] == ["Kubernetes"]
        assert score_details["provider"] == "ollama"
        assert score_details["model"] == "qwen"
        assert score_details["description_hash"] == "c" * 64
        assert score_details["skip_reason"] == ""
        metadata = {
            "provider": "ollama",
            "model": "qwen",
            "prompt_version": "scoring.v1",
            "candidate_hash": "a" * 64,
            "vacancy_hash": "b" * 64,
            "description_hash": "c" * 64,
        }
        assert ilia.needs_scoring("1", **metadata) is False
        assert ilia.needs_scoring(
            "1", **{**metadata, "candidate_hash": "changed"}
        ) is True
        assert ilia.needs_scoring(
            "1", **{**metadata, "description_hash": "changed"}
        ) is True
        assert ilia.needs_scoring(
            "1", **{**metadata, "prompt_version": "scoring.v2"}
        ) is True
        assert ilia.needs_scoring("1", **metadata, force=True) is True
        artifact_id = ilia.record_artifact(
            "scoring",
            "succeeded",
            refnr="1",
            provider="ollama",
            model="qwen",
            prompt_version="scoring.v1",
            candidate_hash="a" * 64,
            vacancy_hash="b" * 64,
            details={"score": 73.0},
        )
        artifact = ilia.artifacts("scoring")[0]
        assert artifact["artifact_id"] == artifact_id
        assert artifact["provider"] == "ollama"
        assert artifact["prompt_version"] == "scoring.v1"
        assert artifact["details"] == {"score": 73.0}
        assert alice.artifacts() == []
    finally:
        ilia.close()
        alice.close()


def test_add_upserts_source_fields_and_preserves_score_and_application_status(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add(
        Vacancy(
            "1",
            "a",
            "b",
            "c",
            "u",
            "ba",
            published="2026-01-01",
            description="alte Beschreibung",
            score=81,
            source_metadata={"old": True},
        )
    )
    s.set_status("1", "applied")
    s.add(
        Vacancy(
            "1",
            "a2",
            "b2",
            "c2",
            "u2",
            "arbeitsagentur",
            published="2026-08-20",
            source_metadata={"search_keyword": "Python"},
        )
    )
    assert len(s.list_all()) == 1
    vacancy = s.by_refnr("1")
    assert (vacancy.title, vacancy.company, vacancy.location, vacancy.url) == (
        "a2",
        "b2",
        "c2",
        "u2",
    )
    assert vacancy.source == "arbeitsagentur"
    assert vacancy.published == "2026-08-20"
    assert vacancy.description == "alte Beschreibung"
    assert vacancy.score == 81
    assert s.list_all()[0][5] == "applied"
    assert vacancy.source_metadata == {"search_keyword": "Python"}
    s.close()


def test_description_cache_hash_ttl_and_terminal_source_status(tmp_path):
    s = Store(tmp_path / "t.db")
    try:
        s.add(Vacancy("1", "Job", "ACME", "Wangen", "https://x", "linkedin"))
        assert s.description_needs_refresh("1", 168) is True
        s.update_description("1", "Python und SQL")
        lifecycle = s.vacancy_lifecycle("1")
        assert lifecycle["description_hash"] == hashlib.sha256(
            "Python und SQL".encode()
        ).hexdigest()
        assert lifecycle["description_fetched_at"]
        assert s.description_needs_refresh("1", 168) is False

        s.conn.execute(
            "UPDATE vacancies SET description_fetched_at='2020-01-01 00:00:00'"
        )
        s.conn.commit()
        assert s.description_needs_refresh("1", 168) is True
        s.update_description("1", "")
        assert s.description_needs_refresh("1", 168) is False

        s.mark_unavailable("1")
        assert s.description_needs_refresh("1", 168) is False
        assert s.list_all() == []
        assert s.by_refnr("1").source_status == "unavailable"
    finally:
        s.close()


def test_archived_vacancy_is_reactivated_by_upsert_without_losing_applied(tmp_path):
    s = Store(tmp_path / "t.db")
    try:
        s.add(Vacancy("1", "Old", "ACME", "Wangen", "https://old", "linkedin"))
        s.set_status("1", "applied")
        s.mark_unavailable("1")
        assert s.list_all() == []

        s.add(Vacancy("1", "New", "ACME", "Wangen", "https://new", "linkedin"))
        assert s.by_refnr("1").source_status == "active"
        assert s.by_refnr("1").archived_at == ""
        assert s.by_refnr("1").title == "New"
        assert s.list_all()[0][5] == "applied"
    finally:
        s.close()


def test_purge_archived_is_aged_account_scoped_and_keeps_applied(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        for refnr in ("old", "applied", "recent"):
            ilia.add(Vacancy(refnr, "Job", "ACME", "x", "u", "linkedin"))
            ilia.mark_unavailable(refnr)
        ilia.set_status("applied", "applied")
        ilia.conn.execute(
            "UPDATE vacancies SET archived_at='2020-01-01 00:00:00' "
            "WHERE account_id=? AND refnr IN ('old', 'applied')",
            (ilia.account_id("Илья"),),
        )
        ilia.conn.commit()
        alice.add(Vacancy("old", "Alice", "Beta", "x", "u", "linkedin"))
        alice.mark_unavailable("old")
        alice.conn.execute(
            "UPDATE vacancies SET archived_at='2020-01-01 00:00:00' "
            "WHERE account_id=?",
            (alice.account_id("Alice"),),
        )
        alice.conn.commit()

        assert ilia.purge_archived(30) == 1
        assert {row[0] for row in ilia.list_all(include_archived=True)} == {
            "applied",
            "recent",
        }
        assert alice.by_refnr("old") is not None
    finally:
        ilia.close()
        alice.close()


def test_purge_archived_cli_requires_explicit_confirmation(monkeypatch):
    from types import SimpleNamespace
    import cv_agent.cli as cli

    monkeypatch.setattr(
        cli,
        "Store",
        lambda: pytest.fail("Store darf ohne --confirm nicht geöffnet werden"),
    )
    with pytest.raises(SystemExit, match="--confirm"):
        cli.cmd_purge_archived(
            SimpleNamespace(confirm=False, older_than_days=30)
        )


def test_set_status(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add(Vacancy("1", "a", "b", "c", "u", "ba"))
    s.set_status("1", "applied")
    assert s.by_refnr("1").score is None
    assert s.by_refnr("1").score_state == "unscored"
    assert s.list_all()[0][5] == "applied"
    s.close()


def test_default_account_exists_with_keywords(tmp_path):
    s = Store(tmp_path / "t.db")
    assert "Илья" in s.list_accounts()
    assert s.account_keywords("Илья") is not None  # импортированы из criteria.yaml
    assert s.account_keywords("Илья")
    s.close()


def test_create_account(tmp_path):
    s = Store(tmp_path / "t.db")
    assert s.create_account("Alice") is True
    assert "Alice" in s.list_accounts()
    assert s.create_account("Alice") is False  # дубликат
    assert s.account_keywords("Alice") == []
    account_id = s.account_id("Alice")
    assert str(UUID(account_id)) == account_id
    assert s.account_name(account_id) == "Alice"
    assert s.account_name("not-a-uuid") is None
    s.close()


def test_account_id_is_stable_across_reopen(tmp_path):
    path = tmp_path / "t.db"
    first = Store(path)
    account_id = first.account_id("Илья")
    first.close()
    second = Store(path)
    assert second.account_id("Илья") == account_id
    second.close()


def test_candidate_profiles_are_structured_and_isolated(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        alice.set_candidate_profile(
            {
                "full_name": "Alice Example",
                "address": "Example Street 1",
                "email": "alice@example.test",
                "phone": "+49 123",
                "languages": ["Deutsch C1", "English B2"],
                "preferred_letter_language": "en",
                "search_region": "Berlin",
            }
        )
        profile = alice.candidate_profile()
        assert profile["profile_name"] == "Alice"
        assert profile["full_name"] == "Alice Example"
        assert profile["languages"] == ["Deutsch C1", "English B2"]
        assert profile["preferred_letter_language"] == "en"
        assert profile["search_region"] == "Berlin"
        assert ilia.candidate_profile()["full_name"] == ""
    finally:
        ilia.close()
        alice.close()


def test_candidate_profile_rejects_unknown_language_and_account(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        with pytest.raises(ValueError, match="язык"):
            store.set_candidate_profile({"preferred_letter_language": "xx"})
        missing = Store(tmp_path / "t.db", user="Missing")
        try:
            assert missing.candidate_profile() is None
            with pytest.raises(ValueError, match="не существует"):
                missing.set_candidate_profile({})
        finally:
            missing.close()
    finally:
        store.close()


def test_document_metadata_is_isolated_and_manual_type_is_preserved(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    metadata = {
        "filename": "document.pdf",
        "document_type": "other",
        "type_source": "auto",
        "mime_type": "application/pdf",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "extracted_chars": 42,
        "extract_error": "",
    }
    try:
        alice.upsert_document_metadata(metadata)
        assert alice.set_document_type("document.pdf", "cv") is True
        alice.upsert_document_metadata(metadata)
        row = alice.document_metadata()[0]
        assert row["document_type"] == "cv"
        assert row["type_source"] == "manual"
        assert ilia.document_metadata() == []
        alice.delete_document_metadata("document.pdf")
        assert alice.document_metadata() == []
    finally:
        ilia.close()
        alice.close()


def test_document_metadata_validates_types(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        with pytest.raises(ValueError, match="тип документа"):
            store.upsert_document_metadata(
                {"filename": "x", "document_type": "executable"}
            )
        with pytest.raises(ValueError, match="тип документа"):
            store.set_document_type("x", "executable")
    finally:
        store.close()


def test_set_account_keywords(tmp_path):
    s = Store(tmp_path / "t.db")
    s.create_account("Alice")
    s.set_account_keywords(["Python", "AI Engineer"], "Alice")
    assert s.account_keywords("Alice") == ["Python", "AI Engineer"]
    s.set_account_keywords([], "Alice")
    assert s.account_keywords("Alice") == []
    s.close()


def test_account_keywords_unknown_returns_none(tmp_path):
    s = Store(tmp_path / "t.db")
    assert s.account_keywords("nobody") is None
    s.close()


def test_gemini_key_per_account(tmp_path):
    a = Store(tmp_path / "t.db", user="Alice")
    b = Store(tmp_path / "t.db", user="Bob")
    a.create_account("Alice")
    b.create_account("Bob")
    assert a.account_gemini_key("Alice") == ""
    a.set_account_gemini_key("AIza-ALICE", "Alice")
    assert a.account_gemini_key("Alice") == "AIza-ALICE"
    assert b.account_gemini_key("Bob") == ""  # у Bob своего ключа нет
    b.set_account_gemini_key("AIza-BOB", "Bob")
    assert b.account_gemini_key("Bob") == "AIza-BOB"
    a.close()
    b.close()


def test_gemini_key_clear(tmp_path):
    s = Store(tmp_path / "t.db")
    s.set_account_gemini_key("AIza-X", "Илья")
    s.set_account_gemini_key("", "Илья")
    assert s.account_gemini_key("Илья") == ""
    s.close()


def test_groq_key_per_account_and_generic_accessors(tmp_path):
    a = Store(tmp_path / "t.db", user="Alice")
    b = Store(tmp_path / "t.db", user="Bob")
    a.create_account("Alice")
    b.create_account("Bob")
    try:
        assert a.account_groq_key("Alice") == ""
        a.set_account_groq_key("gsk-ALICE", "Alice")
        assert a.account_groq_key("Alice") == "gsk-ALICE"
        assert b.account_groq_key("Bob") == ""
        # Обобщённый доступ по имени env-переменной.
        assert a.account_llm_key("GROQ_API_KEY", "Alice") == "gsk-ALICE"
        assert a.account_llm_key("GEMINI_API_KEY", "Alice") == ""
        assert a.account_llm_key("OLLAMA_API_KEY", "Alice") == ""
        b.set_account_llm_key("GROQ_API_KEY", "gsk-BOB", "Bob")
        assert b.account_groq_key("Bob") == "gsk-BOB"
        with pytest.raises(ValueError):
            a.set_account_llm_key("UNKNOWN_ENV", "x", "Alice")
    finally:
        a.close()
        b.close()


def test_preferred_llm_roundtrip_per_account(tmp_path):
    a = Store(tmp_path / "t.db", user="Alice")
    b = Store(tmp_path / "t.db", user="Bob")
    a.create_account("Alice")
    b.create_account("Bob")
    try:
        assert a.account_preferred_llm("Alice") == ""
        a.set_account_preferred_llm("groq:llama-3.3-70b-versatile", "Alice")
        assert a.account_preferred_llm("Alice") == "groq:llama-3.3-70b-versatile"
        assert b.account_preferred_llm("Bob") == ""
        a.set_account_preferred_llm("", "Alice")
        assert a.account_preferred_llm("Alice") == ""
    finally:
        a.close()
        b.close()


def test_search_params_roundtrip_per_account(tmp_path):
    """Поля формы поиска сохраняются в SQLite и изолированы по аккаунтам."""
    a = Store(tmp_path / "t.db", user="Alice")
    b = Store(tmp_path / "t.db", user="Bob")
    a.create_account("Alice")
    b.create_account("Bob")
    try:
        assert a.account_search_params("Alice") == {}
        params = {
            "limit": "42",
            "country": "Deutschland",
            "region": "Bayern",
            "city": "München",
        }
        a.set_account_search_params(params, "Alice")
        # Перезакрытие хранилища — данные живут в SQLite между сессиями.
        reopened = Store(tmp_path / "t.db", user="Alice")
        try:
            assert reopened.account_search_params("Alice") == params
        finally:
            reopened.close()
        assert b.account_search_params("Bob") == {}
        # Некорректный JSON и несловарные значения читаются как {}.
        raw = tmp_path / "t.db"
        import sqlite3 as sq

        conn = sq.connect(raw)
        try:
            conn.execute(
                "UPDATE accounts SET search_params=? WHERE name=?",
                ("not-json", "Bob"),
            )
            conn.commit()
        finally:
            conn.close()
        assert b.account_search_params("Bob") == {}
        a.set_account_search_params({}, "Alice")
        assert a.account_search_params("Alice") == {}
    finally:
        a.close()
        b.close()


def test_vacancies_isolated_per_user(tmp_path):
    a = Store(tmp_path / "t.db", user="Alice")
    b = Store(tmp_path / "t.db", user="Bob")
    a.create_account("Alice")
    b.create_account("Bob")
    a.add(Vacancy("1", "Alice Job", "b", "c", "u", "arbeitsagentur"))
    b.add(Vacancy("1", "Bob Job", "b", "c", "u", "arbeitsagentur"))
    assert [r[1] for r in a.list_all()] == ["Alice Job"]
    assert [r[1] for r in b.list_all()] == ["Bob Job"]
    assert a.by_refnr("1").title == "Alice Job"
    assert b.by_refnr("1").title == "Bob Job"
    a.delete("1")
    assert b.by_refnr("1") is not None
    assert a.by_refnr("1") is None
    assert a.ba_refnrs() == []
    assert b.ba_refnrs() == ["1"]
    a.close()
    b.close()


def test_vacancies_follow_stable_id_after_profile_rename(tmp_path):
    path = tmp_path / "t.db"
    original = Store(path, user="Alice")
    original.create_account("Alice")
    original.add(Vacancy("same-ref", "Stable Job", "ACME", "x", "u", "ba"))
    account_id = original.account_id("Alice")
    original.conn.execute(
        "UPDATE accounts SET name='Alicia' WHERE account_id=?", (account_id,)
    )
    original.conn.commit()
    original.close()

    renamed = Store(path, user="Alicia")
    try:
        assert renamed.account_id("Alicia") == account_id
        assert renamed.by_refnr("same-ref").title == "Stable Job"
        legacy_name = renamed.conn.execute(
            "SELECT user FROM vacancies WHERE account_id=?", (account_id,)
        ).fetchone()[0]
        assert legacy_name == "Alice"
    finally:
        renamed.close()


def test_vacancy_write_requires_existing_profile(tmp_path):
    store = Store(tmp_path / "t.db", user="Missing")
    try:
        with pytest.raises(ValueError, match="не существует"):
            store.add(Vacancy("1", "Job", "ACME", "x", "u", "ba"))
    finally:
        store.close()


def _job_steps(label="search"):
    return [{"cmd": label, "args": [label], "code": 99, "output": "ignored"}]


def test_jobs_are_isolated_and_one_active_job_is_allowed_per_profile(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        ilia_run, created = ilia.create_job("pipeline", _job_steps(), "local")
        assert created is True
        same_run, created = ilia.create_job("score_missing", _job_steps("score"))
        assert (same_run, created) == (ilia_run, False)

        alice_run, created = alice.create_job("pipeline", _job_steps())
        assert created is True
        assert alice_run != ilia_run
        assert ilia.job(alice_run) is None
        assert alice.job(ilia_run) is None
        assert ilia.active_job()["run_id"] == ilia_run
        assert alice.active_job()["run_id"] == alice_run
    finally:
        ilia.close()
        alice.close()


def test_job_progress_is_persisted_and_log_is_bounded(tmp_path):
    from cv_agent.store import JOB_LOG_LIMIT

    store = Store(tmp_path / "t.db")
    try:
        run_id, created = store.create_job("pipeline", _job_steps())
        assert created is True
        assert store.claim_job(run_id) is True
        assert store.claim_job(run_id) is False
        assert store.record_job_resource_usage(run_id, "network", 0.25)
        assert store.record_job_resource_usage(run_id, "llm", 0.5)
        store.append_job_output(run_id, 0, "x" * (JOB_LOG_LIMIT + 100))
        store.set_job_step_code(run_id, 0, 0)
        assert store.finish_job(run_id, "succeeded") is True

        job = store.job(run_id)
        assert job["done"] is True
        assert job["status"] == "succeeded"
        assert job["steps"][0]["code"] == 0
        assert len(job["steps"][0]["output"]) == JOB_LOG_LIMIT
        assert "обрезано" in job["steps"][0]["output"]
        assert job["metrics"] == {
            "network_requests": 1,
            "network_wait_seconds": 0.25,
            "llm_requests": 1,
            "llm_wait_seconds": 0.5,
            "embedding_requests": 0,
            "embedding_wait_seconds": 0.0,
        }
        assert store.active_job() is None
    finally:
        store.close()


def test_job_validates_kind_steps_and_terminal_state(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        with pytest.raises(ValueError, match="тип"):
            store.create_job("shell", _job_steps())
        with pytest.raises(ValueError, match="шаг"):
            store.create_job("pipeline", [])
        with pytest.raises(ValueError, match="allowlist"):
            store.create_job(
                "pipeline", [{"cmd": "web", "args": ["web"]}]
            )
        run_id, _ = store.create_job("pipeline", _job_steps())
        with pytest.raises(ValueError, match="конечное"):
            store.finish_job(run_id, "running")
    finally:
        store.close()


def test_startup_recovery_interrupts_active_jobs_and_unblocks_profiles(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        ilia_run, _ = ilia.create_job("pipeline", _job_steps())
        alice_run, _ = alice.create_job("pipeline", _job_steps())
        assert ilia.claim_job(ilia_run)
        assert ilia.interrupt_active_jobs() == 2
        assert ilia.job(ilia_run)["status"] == "interrupted"
        assert alice.job(alice_run)["status"] == "interrupted"
        assert "запустите" in ilia.job(ilia_run)["error"]
        replacement, created = ilia.create_job("pipeline", _job_steps())
        assert created is True
        assert replacement != ilia_run
    finally:
        ilia.close()
        alice.close()


def test_rate_limit_cooldowns_are_isolated_by_account_and_provider(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    try:
        assert ilia.defer_llm_rate_limit("gemini-a", 30, now=1000) == 1030
        assert ilia.llm_rate_limit_delay("gemini-a", now=1005) == 25
        assert ilia.llm_rate_limit_delay("gemini-b", now=1005) == 0
        assert alice.llm_rate_limit_delay("gemini-a", now=1005) == 0
        ilia.clear_llm_rate_limit("gemini-a")
        assert ilia.llm_rate_limit_delay("gemini-a", now=1005) == 0
    finally:
        ilia.close()
        alice.close()


def test_llm_quota_snapshot_roundtrip_and_isolation(tmp_path):
    path = tmp_path / "t.db"
    ilia = Store(path, user="Илья")
    ilia.create_account("Alice")
    alice = Store(path, user="Alice")
    headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "970",
    }
    try:
        assert ilia.llm_quota("groq-key") is None
        ilia.save_llm_quota("groq-key", headers)
        saved, stamp = ilia.llm_quota("groq-key")
        assert saved == headers
        assert stamp  # время снимка записано
        # Перезапись — всегда последний снимок.
        ilia.save_llm_quota("groq-key", {"x-ratelimit-limit-tokens": "5"})
        saved, _ = ilia.llm_quota("groq-key")
        assert saved == {"x-ratelimit-limit-tokens": "5"}
        # Квоты изолированы по аккаунту.
        assert alice.llm_quota("groq-key") is None
        alice.save_llm_quota("groq-key", {"x": "1"})
        assert ilia.llm_quota("groq-key")[0] != {"x": "1"}
    finally:
        ilia.close()
        alice.close()


def test_job_rate_limit_wait_state_is_reversible(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        run_id, _ = store.create_job("pipeline", _job_steps())
        assert store.claim_job(run_id)
        assert store.set_job_rate_limit_wait(run_id, True)
        assert store.job(run_id)["status"] == "waiting_rate_limit"
        assert store.set_job_rate_limit_wait(run_id, False)
        assert store.job(run_id)["status"] == "running"
    finally:
        store.close()


def test_manual_description_edit_survives_upsert_and_ttl_refresh(tmp_path):
    s = Store(tmp_path / "t.db")
    try:
        s.add(
            Vacancy(
                "1", "AI Engineer", "ACME", "Wangen", "http://x", "ba",
                description="Старое описание из источника",
            )
        )
        s.update_description("1", "Вручную исправленное описание", edited=True)
        assert s.by_refnr("1").description == "Вручную исправленное описание"
        assert s.vacancy_lifecycle("1")["description_hash"] != ""

        # Свежий результат поиска не затирает ручную правку и не теряет пометку.
        s.add(
            Vacancy(
                "1", "AI Engineer", "ACME", "Wangen", "http://x", "ba",
                description="Свежее описание из источника",
            )
        )
        assert s.by_refnr("1").description == "Вручную исправленное описание"
        assert s._description_edited("1") is True

        # TTL-обновление тоже не трогает отредактированное описание.
        s.conn.execute(
            "UPDATE vacancies SET description_fetched_at='2020-01-01 00:00:00' "
            "WHERE refnr='1'"
        )
        s.conn.commit()
        assert s.description_needs_refresh("1", ttl_hours=1) is False
    finally:
        s.close()


def test_update_description_without_edit_flag_stays_refreshable(tmp_path):
    s = Store(tmp_path / "t.db")
    try:
        s.add(Vacancy("1", "AI Engineer", "ACME", "Wangen", "http://x", "ba"))
        s.update_description("1", "Описание из источника")
        assert s.by_refnr("1").description == "Описание из источника"
        assert s._description_edited("1") is False
        s.conn.execute(
            "UPDATE vacancies SET description_fetched_at='2020-01-01 00:00:00' "
            "WHERE refnr='1'"
        )
        s.conn.commit()
        assert s.description_needs_refresh("1", ttl_hours=1) is True
    finally:
        s.close()
