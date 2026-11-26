import hashlib
from dataclasses import replace

import pytest
import requests

from cv_agent.embeddings import BackendIdentity, CachedEmbeddings, EmbeddingError, OllamaEmbeddings, cosine, unit_vector
from cv_agent.scoring_config import EmbeddingConfig, parse_scoring_config
from cv_agent.store import Store


class FixedBackend:
    def __init__(self, digest="revision-one"):
        self.identity = BackendIdentity("test:1", digest, 2, "http://localhost:11434", "query:", "doc:", 1024)
        self.calls = []

    def embed(self, texts, *, role):
        self.calls.append((texts, role))
        return [(1., 0.) if "software" in text.lower() else (0., 1.) for text in texts]


def response(payload, status=200):
    class Response:
        status_code = status

        def json(self):
            return payload
    return Response()


def local_inventory(monkeypatch, digest="test-digest"):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response({"models": [{"name": "qwen3-embedding:0.6b", "digest": digest}]}))


def test_native_client_batches_prefixes_normalizes_and_disables_truncation(monkeypatch):
    local_inventory(monkeypatch)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response({"embeddings": [[3, 4] for _ in kwargs["json"]["input"]]})

    monkeypatch.setattr(requests, "post", post)
    client = OllamaEmbeddings(EmbeddingConfig(dimensions=2, batch_size=2, query_prefix="query:"))
    assert client.embed(["one", "two", "three"], role="query") == [(0.6, 0.8)] * 3
    assert len(calls) == 2
    assert calls[0][0].endswith("/api/embed")
    assert calls[0][1]["json"]["input"] == ["query:one", "query:two"]
    assert calls[0][1]["json"]["truncate"] is False
    assert calls[0][1]["allow_redirects"] is False
    assert client.identity.digest == "test-digest"


@pytest.mark.parametrize("vectors", [[[1, 2, 3]], [[float('nan'), 1]], [[10 ** 1000, 1]], [[True, 1]], [[0, 0]], []])
def test_invalid_vectors_never_become_scores(monkeypatch, vectors):
    local_inventory(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response({"embeddings": vectors}))
    with pytest.raises(EmbeddingError):
        OllamaEmbeddings(EmbeddingConfig(dimensions=2)).embed(["text"], role="document")


def test_missing_model_and_wrong_revision_do_not_fallback(monkeypatch):
    local_inventory(monkeypatch)
    with pytest.raises(EmbeddingError, match="digest differs"):
        _ = OllamaEmbeddings(EmbeddingConfig(expected_digest="wrong")).identity
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response({"models": []}))
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: pytest.fail("No provider fallback"))
    with pytest.raises(EmbeddingError, match="not installed"):
        _ = OllamaEmbeddings(EmbeddingConfig()).identity


def test_transport_failure_is_safe_and_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("sensitive contact: secret@example.test")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(EmbeddingError) as failure:
        _ = OllamaEmbeddings(EmbeddingConfig()).identity
    assert "secret@example.test" not in str(failure.value)


@pytest.mark.parametrize("url", ["https://api.example.com", "http://169.254.169.254", "http://localhost:11434/api", "http://user:pass@localhost:11434"])
def test_embedding_endpoint_must_be_local_root(url):
    with pytest.raises(ValueError):
        OllamaEmbeddings(EmbeddingConfig(base_url=url))


def test_cache_survives_process_recreation_and_edits_replace_old_rows(tmp_path):
    path = tmp_path / "cache.db"
    backend = FixedBackend()
    with_store = Store(path)
    cache = CachedEmbeddings(with_store, backend)
    assert cache.embed_entity("document", "cv", ["software", "tools", "software"], role="document") == [(1, 0), (0, 1), (1, 0)]
    assert cache.misses == 2
    with_store.close()
    second = Store(path)
    backend2 = FixedBackend()
    try:
        cache2 = CachedEmbeddings(second, backend2)
        cache2.embed_entity("document", "cv", ["software", "tools"], role="document")
        assert backend2.calls == []
        assert cache2.hits == 2
        cache2.embed_entity("document", "cv", ["software edited"], role="document")
        rows = second.embedding_vectors("document", "cv", cache2.identity.key, "document")
        assert set(rows) == {hashlib.sha256(b"software edited").hexdigest()}
    finally:
        second.close()


def test_digest_role_and_profile_are_separate_cache_keys(tmp_path):
    path = tmp_path / "cache.db"
    one = Store(path)
    one.create_account("Other")
    other = Store(path, user="Other")
    try:
        first = CachedEmbeddings(one, FixedBackend())
        first.embed_entity("document", "same-id", ["software"], role="document")
        assert other.embedding_vectors("document", "same-id", first.identity.key, "document") == {}
        assert one.embedding_vectors("document", "same-id", first.identity.key, "query") == {}
        changed = CachedEmbeddings(one, FixedBackend("revision-two"))
        changed.embed_entity("document", "same-id", ["software"], role="document")
        assert changed.misses == 1
        assert one.embedding_vectors("document", "same-id", first.identity.key, "document") == {}
    finally:
        one.close()
        other.close()


def test_invalid_inference_does_not_destroy_existing_cache(tmp_path):
    store = Store(tmp_path / "cache.db")
    try:
        backend = FixedBackend()
        cache = CachedEmbeddings(store, backend)
        cache.embed_entity("document", "cv", ["software"], role="document")
        backend.embed = lambda *args, **kwargs: [(0, 0)]
        with pytest.raises(EmbeddingError):
            cache.embed_entity("document", "cv", ["new"], role="document")
        assert len(store.embedding_vectors("document", "cv", backend.identity.key, "document")) == 1
    finally:
        store.close()


def test_corrupt_vector_is_recomputed(tmp_path):
    store = Store(tmp_path / "cache.db")
    try:
        backend = FixedBackend()
        cache = CachedEmbeddings(store, backend)
        cache.embed_entity("document", "cv", ["software"], role="document")
        store.conn.execute("UPDATE embedding_vectors SET vector_json='[0,0]'")
        store.conn.commit()
        before = len(backend.calls)
        cache.embed_entity("document", "cv", ["software"], role="document")
        assert len(backend.calls) == before + 1
    finally:
        store.close()


def test_deleted_document_and_profile_remove_vectors(tmp_path):
    store = Store(tmp_path / "cache.db")
    try:
        document_id = hashlib.sha256(f"{store.account_id()}:cv.pdf".encode()).hexdigest()
        cache = CachedEmbeddings(store, FixedBackend())
        cache.embed_entity("document", document_id, ["software"], role="document")
        store.delete_document_metadata("cv.pdf")
        assert store.embedding_vectors("document", document_id, cache.identity.key, "document") == {}
        cache.embed_entity("document", "gone", ["software"], role="document")
        store.prune_document_vectors(set())
        assert store.conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.parametrize("config", [
    {"engine": "unknown"}, {"weights": {"skills": 1}},
    {"similarity_low": .9, "similarity_high": .5}, {"total_chars": True},
    {"embeddings": {"dimensions": 0}}, {"embeddings": {"timeout": float("nan")}},
    {"embeddings": {"base_url": "https://cloud.example.test"}},
])
def test_invalid_scoring_config_fails_early(config):
    with pytest.raises(ValueError):
        parse_scoring_config(config)


def test_backend_identity_changes_with_prefix_dimensions_and_revision():
    original = FixedBackend().identity
    assert len({original.key, replace(original, digest="new").key, replace(original, dimensions=3).key, replace(original, query_prefix="other:").key}) == 4
    assert cosine(unit_vector([3, 4], 2), unit_vector([3, 4], 2)) == 1


def test_model_revision_change_during_request_does_not_return_vectors(monkeypatch):
    local_inventory(monkeypatch)
    def post(*args, **kwargs):
        local_inventory(monkeypatch, digest="changed-revision")
        return response({"embeddings": [[1, 0]]})
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(EmbeddingError, match="changed during inference"):
        OllamaEmbeddings(EmbeddingConfig(dimensions=2)).embed(["text"], role="document")


def test_profile_deletion_cascades_vectors(tmp_path):
    store = Store(tmp_path / "cache.db")
    store.create_account("Other")
    other = Store(tmp_path / "cache.db", user="Other")
    try:
        cache = CachedEmbeddings(other, FixedBackend())
        cache.embed_entity("document", "cv", ["software"], role="document")
        assert store.delete_account("Other")
        assert store.conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 0
    finally:
        store.close()
        other.close()


def test_optional_resident_snapshot_matches_revision_and_validates_numbers(monkeypatch):
    local_inventory(monkeypatch)
    client = OllamaEmbeddings(EmbeddingConfig(dimensions=2))
    _ = client.identity
    payload = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "test-digest", "size": 1000, "size_vram": 0, "context_length": 4096}]}
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response(payload))
    assert client.resident_model() == {"size": 1000, "size_vram": 0, "context_length": 4096}
    payload["models"][0]["size_vram"] = True
    assert client.resident_model()["size_vram"] is None
    payload["models"][0]["digest"] = "wrong-revision"
    assert client.resident_model() is None
