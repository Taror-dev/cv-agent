import json

from cv_agent.config import Criteria
from cv_agent.scoring_config import ScoringConfig
from cv_agent.store import Store
from cv_agent.models import Vacancy
from cv_agent.webapp import app


def test_algorithmic_pipeline_skips_generative_provider_selection(monkeypatch):
    import cv_agent.webapp as web
    criteria = Criteria("Wangen", 50, ["Python"], [], 70, scoring=ScoringConfig(engine="algorithmic"))
    monkeypatch.setattr(web, "load_criteria", lambda user=None: criteria)
    monkeypatch.setattr(web, "_subprocess_llm_env", lambda: (_ for _ in ()).throw(AssertionError("Generative provider must not be selected")))
    calls = []
    monkeypatch.setattr(web, "_start_pipeline_job", lambda *args: calls.append(args) or 42)
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    response = client.post("/pipeline", data={"limit": "10"})
    assert response.status_code == 302
    assert [step["args"][0] for step in calls[0][2]] == ["search", "score"]
    assert all(not key.startswith("LLM_") for key in calls[0][4])


def test_algorithmic_evidence_is_escaped_and_account_scoped(monkeypatch):
    import cv_agent.webapp as web
    store = Store()
    store.add(Vacancy("r1", "Developer", "Synthetic", "Wangen", "https://example.test", "ba", description="Meaningful synthetic vacancy description."))
    report = {
        "score": 40, "version": "algorithmic.v1", "backend": {"key": "test-model"},
        "components": {"skills": {"value": 1, "state": "evaluated"}},
        "preliminary": True, "reasons": [], "missing": ["Unconfirmed language"],
        "matches": [{"explanation": "Python", "evidence": [{"text": "<script>alert('x')</script>"}]}],
        "task_matches": [],
    }
    store.update_algorithmic_score("r1", report, candidate_hash="candidate", vacancy_hash="vacancy", description_hash="description", settings_hash="settings")
    store.create_account("Other")
    store.close()
    monkeypatch.setattr(web, "ensure_description", lambda *args, **kwargs: None)
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    response = client.get("/vacancy/r1")
    html = response.data.decode()
    assert response.status_code == 200
    assert "Алгоритмическая оценка" in html
    assert "&lt;script&gt;" in html
    assert "<script>alert('x')</script>" not in html
    client.post("/logout")
    client.post("/login", data={"account": "Other"})
    response = client.get("/vacancy/r1")
    assert "&lt;script&gt;" not in response.data.decode()


def test_installed_embedding_model_is_not_a_letter_model_option(monkeypatch):
    import subprocess
    import cv_agent.webapp as web
    monkeypatch.setattr(web.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="NAME ID SIZE MODIFIED\nqwen3.8:latest x 1GB now\nqwen3-embedding:0.6b y 1GB now\nbge-m3:latest z 1GB now\n"))
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    response = client.get("/")
    assert response.status_code == 200
    assert "ollama:qwen3-embedding:0.6b" not in response.data.decode()
    assert "ollama:bge-m3:latest" not in response.data.decode()


def test_table_distinguishes_engines_preliminary_results_and_filters_without_losing_sorting():
    from urllib.parse import urlencode
    store = Store()
    for refnr in ("algorithm", "llm", "failed"):
        store.add(Vacancy(refnr, "Job " + refnr, "Synthetic", "Wangen", "url", "ba"))
    report = {
        "score": 75, "version": "algorithmic.v2", "backend": {"key": "test-model"},
        "components": {}, "preliminary": True, "reasons": [], "missing": [],
        "matches": [], "task_matches": [],
    }
    store.update_algorithmic_score("algorithm", report, candidate_hash="candidate", vacancy_hash="vacancy", description_hash="description", settings_hash="settings")
    store.update_score_details("llm", 85, [], [])
    store.mark_score_error("failed", "embedding_error", engine="algorithmic")
    store.create_account("Other")
    store.close()
    other = Store(user="Other")
    other.add(Vacancy("algorithm", "Private Other Job", "Synthetic", "Wangen", "url", "ba", score=99))
    other.close()
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    html = client.get("/results").data.decode()
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert body.index("Job llm") < body.index("Job algorithm") < body.index("Job failed")
    assert "Предварительная" in body
    assert "ошибка оценки" in body
    assert "ошибка формата" not in html
    assert "Private Other Job" not in html
    response = client.get("/results?" + urlencode({"f_engine": "Алгоритм", "f_score": ">70"}))
    body = response.data.decode().split("<tbody>")[1].split("</tbody>")[0]
    assert "Job algorithm" in body
    assert "Job llm" not in body
    assert "Job failed" not in body
    client.post("/logout")
    client.post("/login", data={"account": "Other"})
    html = client.get("/results").data.decode()
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "Private Other Job" in body
    assert "Алгоритм" not in body
    assert "Предварительная" not in body


def test_card_shows_location_proof_escaped_without_claiming_distance(monkeypatch):
    from cv_agent.scoring_location import evaluate_location
    import cv_agent.webapp as web
    store = Store()
    store.add(Vacancy("r1", "Developer", "Synthetic", "<b>Berlin</b>", "url", "ba"))
    report = {
        "score": 50, "version": "algorithmic.v2", "backend": {"key": "test-model"},
        "components": {"location": evaluate_location({"work_mode": "remote", "cities": ["Berlin"]}, "<b>Berlin</b>", "100% remote <script>evil()</script>")},
        "preliminary": True, "reasons": [], "missing": [], "matches": [], "task_matches": [],
    }
    store.update_algorithmic_score("r1", report, candidate_hash="candidate", vacancy_hash="vacancy", description_hash="description", settings_hash="settings")
    store.close()
    monkeypatch.setattr(web, "ensure_description", lambda *args, **kwargs: None)
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    response = client.get("/vacancy/r1")
    assert response.status_code == 200
    html = response.data.decode()
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html
    assert "<script>evil()</script>" not in html
    assert "подтверждено" in html
    assert "нужна проверка" in html
    assert "Расстояние, радиус" in html


def test_scoring_history_preserves_both_engines_and_is_account_scoped(monkeypatch):
    import cv_agent.webapp as web
    store = Store()
    store.add(Vacancy("r1", "Developer", "Synthetic", "Wangen", "url", "ba"))
    store.record_artifact("scoring", "succeeded", refnr="r1", provider="ollama", model="old-generative", prompt_version="scoring.v2", details={"score": 83})
    report = {"score": 71, "version": "algorithmic.v2", "backend": {"key": "test-model", "model": "local-embedding"}, "components": {}, "preliminary": True, "reasons": [], "missing": [], "matches": [], "task_matches": []}
    store.update_algorithmic_score("r1", report, candidate_hash="candidate", vacancy_hash="vacancy", description_hash="description", settings_hash="settings")
    store.record_artifact("scoring", "succeeded", refnr="r1", provider="local-embeddings", model="test-model", details=report)
    store.record_artifact("scoring", "failed", refnr="r1", provider="ollama", details={"score": 0})
    store.record_artifact("scoring", "succeeded", refnr="r1", details={"score": None})
    store.create_account("Other")
    assert [entry["score"] for entry in store.scoring_history("r1")] == [71, 83]
    assert len(store.scoring_history("r1", limit=2)) == 1
    store.close()
    other = Store(user="Other")
    other.add(Vacancy("r1", "Other Job", "Synthetic", "Berlin", "url", "ba"))
    other.record_artifact("scoring", "succeeded", refnr="r1", model="private-other-model", details={"score": 99})
    assert len(other.scoring_history("r1")) == 1
    other.close()
    monkeypatch.setattr(web, "ensure_description", lambda *args, **kwargs: None)
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    html = client.get("/vacancy/r1").data.decode()
    assert "История оценок" in html
    assert "old-generative" in html and "local-embedding" in html
    assert "private-other-model" not in html
    client.post("/logout")
    client.post("/login", data={"account": "Other"})
    html = client.get("/vacancy/r1").data.decode()
    assert "private-other-model" in html
    assert "old-generative" not in html
