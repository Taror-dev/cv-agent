from dataclasses import replace
from types import SimpleNamespace

import pytest

from cv_agent.algorithmic_score import evaluate_algorithmic, task_passages
from cv_agent.embeddings import CachedEmbeddings
from cv_agent.models import Vacancy
from cv_agent.scoring_config import EmbeddingConfig, ScoringConfig
from cv_agent.scoring_inputs import ScoringInputs
from cv_agent.scoring_rules import Document
from cv_agent.store import Store
from test_embeddings import FixedBackend


def make_inputs(text="Python\nDeutsch B2\nBerufserfahrung:\nDevelop software services", kind="cv"):
    return ScoringInputs((Document("cv", kind, text),), "candidate-hash", (), False)


def vacancy(description=None):
    return Vacancy("r1", "Developer", "Synthetic", "Wangen", "https://example.test/job", "ba", description=description or "Requirements:\nPython required.\nGerman B2 required.\nResponsibilities:\nDevelop software systems")


def test_formula_has_exact_and_semantic_components_without_confirming_skills(tmp_path):
    store = Store(tmp_path / "scores.db")
    try:
        result = evaluate_algorithmic(make_inputs(), vacancy(), ScoringConfig(), CachedEmbeddings(store, FixedBackend()))
        assert result["components"]["skills"]["value"] == 1
        assert result["components"]["tasks"]["value"] == 1
        assert result["components"]["qualifications"]["value"] == 1
        assert result["components"]["location"]["state"] == "unknown"
        assert result["score"] == 90
        assert result["preliminary"]
        assert result["task_matches"][0]["status"] == "semantic_candidate"
        assert {match["requirement"]["key"] for match in result["matches"]} == {"Python", "Deutsch"}
    finally:
        store.close()


def test_semantics_never_override_language_mismatch(tmp_path):
    store = Store(tmp_path / "scores.db")
    try:
        result = evaluate_algorithmic(make_inputs("Python\nDeutsch B1\nDevelop software services"), vacancy(), ScoringConfig(mandatory_mismatch_cap=30), CachedEmbeddings(store, FixedBackend()))
        assert result["score"] == 30
        assert result["cap_applied"]
        assert result["uncapped_score"] > result["score"]
        assert result["uncapped_score"] == sum(component["points"] for component in result["components"].values())
        assert any(match["status"] == "mismatch" for match in result["matches"])
    finally:
        store.close()


def test_course_and_offers_are_not_semantic_work_experience():
    assert task_passages(Document("course", "curriculum", "Develop software services"), candidate=True, max_chars=700) == []
    assert task_passages(Document("cv", "cv", "Education:\nDevelop software in a course"), candidate=True, max_chars=700) == []
    assert task_passages(Document("job", "other", "We offer:\nDevelop software knowledge in paid courses"), candidate=False, max_chars=700) == []


def test_limits_keep_score_preliminary_and_deduplicate_tasks(tmp_path):
    store = Store(tmp_path / "scores.db")
    try:
        job = vacancy("Responsibilities:\nDevelop software systems\nDevelop software systems\nMaintain software services")
        result = evaluate_algorithmic(make_inputs(), job, ScoringConfig(max_fragments=1), CachedEmbeddings(store, FixedBackend()))
        assert len(result["task_matches"]) == 1
        assert result["preliminary"]
        assert any("Verarbeitungslimit" in reason for reason in result["missing"])
    finally:
        store.close()


def prepare_cli(monkeypatch, tmp_path, isolated_project):
    import cv_agent.cli as cli
    from cv_agent.config import Criteria
    path = tmp_path / "cli.db"
    store = Store(path)
    store.add(vacancy())
    store.update_score_details("r1", 77, ["legacy"], [], provider="llm", model="old")
    store.close()
    monkeypatch.setattr(cli, "Store", lambda: Store(path))
    criteria = Criteria("Wangen", 50, ["Python"], [], 70)
    monkeypatch.setattr(cli, "load_criteria", lambda: criteria)
    monkeypatch.setattr(cli, "_description_for", lambda *args: None)
    monkeypatch.setattr(cli, "_llm_client_for", lambda *args: pytest.fail("Algorithmic mode must never create generative client"))
    monkeypatch.setattr("cv_agent.scoring_inputs.collect_scoring_inputs", lambda *args, **kwargs: make_inputs())
    monkeypatch.setattr("cv_agent.embeddings.OllamaEmbeddings", lambda cfg: FixedBackend())
    return cli, path, criteria


def test_cli_preview_preserves_score_then_persist_and_reuse(monkeypatch, tmp_path, isolated_project):
    cli, path, _ = prepare_cli(monkeypatch, tmp_path, isolated_project)
    report = tmp_path / "preview.json"
    args = SimpleNamespace(engine="algorithmic", preview=True, report=str(report), missing=True, force=False, verbose=False)
    cli.cmd_score(args)
    check = Store(path)
    assert check.score_details("r1")["score"] == 77
    assert check.score_details("r1")["engine"] == "llm"
    check.close()
    assert report.stat().st_mode & 0o777 == 0o600
    args.preview, args.report = False, None
    cli.cmd_score(args)
    check = Store(path)
    details = check.score_details("r1")
    assert details["engine"] == "algorithmic"
    assert details["score"] == 90
    assert details["prompt_version"] == ""
    assert details["algorithm_version"] == "algorithmic.v3"
    assert len(check.artifacts("scoring")) == 1
    check.close()
    cli.cmd_score(args)
    check = Store(path)
    assert len(check.artifacts("scoring")) == 1
    check.close()


def test_cli_errors_are_retryable_not_zero_and_do_not_delete(monkeypatch, tmp_path, isolated_project):
    from cv_agent.embeddings import EmbeddingError
    cli, path, _ = prepare_cli(monkeypatch, tmp_path, isolated_project)
    backend = FixedBackend()
    backend.embed = lambda *args, **kwargs: (_ for _ in ()).throw(EmbeddingError("Unavailable"))
    monkeypatch.setattr("cv_agent.embeddings.OllamaEmbeddings", lambda cfg: backend)
    with pytest.raises(SystemExit, match="retryable"):
        cli.cmd_score(SimpleNamespace(engine="algorithmic", preview=False, missing=False, force=False, verbose=False))
    check = Store(path)
    try:
        assert check.by_refnr("r1") is not None
        assert check.score_details("r1")["score"] is None
        assert check.score_details("r1")["state"] == "score_error"
    finally:
        check.close()


def test_cli_no_description_is_skip_not_physical_deletion(monkeypatch, tmp_path, isolated_project):
    cli, path, _ = prepare_cli(monkeypatch, tmp_path, isolated_project)
    store = Store(path)
    store.conn.execute("UPDATE vacancies SET description=''")
    store.conn.commit()
    store.close()
    cli.cmd_score(SimpleNamespace(engine="algorithmic", preview=False, missing=False, force=False, verbose=False))
    check = Store(path)
    try:
        assert check.by_refnr("r1") is not None
        assert check.score_details("r1")["skip_reason"] == "description_unavailable"
    finally:
        check.close()


def test_full_description_and_settings_changes_invalidate_score(monkeypatch, tmp_path, isolated_project):
    cli, path, criteria = prepare_cli(monkeypatch, tmp_path, isolated_project)
    args = SimpleNamespace(engine="algorithmic", preview=False, missing=True, force=False, verbose=False)
    cli.cmd_score(args)
    store = Store(path)
    store.conn.execute("UPDATE vacancies SET description=description || ? WHERE refnr='r1'", ("\n" + "Text " * 800 + "\nDocker erforderlich.",))
    store.conn.commit()
    store.close()
    cli.cmd_score(args)
    store = Store(path)
    assert len(store.artifacts("scoring")) == 2
    store.close()
    criteria.scoring = replace(criteria.scoring, mandatory_mismatch_cap=35)
    cli.cmd_score(args)
    store = Store(path)
    try:
        assert len(store.artifacts("scoring")) == 3
        assert any("Docker" in text for text in store.score_details("r1")["missing"])
    finally:
        store.close()
