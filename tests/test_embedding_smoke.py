import json

import pytest

from cv_agent import embedding_smoke
from cv_agent.embeddings import BackendIdentity, EmbeddingError


def test_smoke_reports_fixed_backend_results_without_database(monkeypatch, tmp_path):
    class Backend:
        identity = BackendIdentity("test:1", "revision", 3, "http://localhost:11434", "", "", 1024)

        def embed(self, texts, *, role):
            return [(1, 0, 0), (0, 1, 0), (0, 0, 1)] if role == "document" else [(1, 0, 0), (1, 0, 0), (0, 0, 1)]

    monkeypatch.setattr(embedding_smoke, "OllamaEmbeddings", lambda cfg: Backend())
    path = tmp_path / "smoke.json"
    assert embedding_smoke.main(["--output", str(path)]) == 0
    report = json.loads(path.read_text())
    assert report["passed"]
    assert [row["nearest"] for row in report["results"]] == [0, 0, 2]
    assert "not validation" in report["limitations"]


def test_smoke_missing_service_has_explicit_failure(monkeypatch):
    def unavailable(cfg):
        raise EmbeddingError("Service unavailable")
    monkeypatch.setattr(embedding_smoke, "OllamaEmbeddings", unavailable)
    with pytest.raises(SystemExit) as exc:
        embedding_smoke.main([])
    assert exc.value.code == 2
