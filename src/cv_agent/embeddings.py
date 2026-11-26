"""Local Ollama embeddings with immutable backend identity and strict vectors."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import requests

from .config import ollama_host
from .resource_limits import FileSlotPool, network_request_limit
from .scoring_config import EmbeddingConfig, local_embedding_url

PREPARATION_VERSION = "embeddings.v1"


class EmbeddingError(RuntimeError):
    pass


def unit_vector(values, dimensions: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != dimensions:
        raise EmbeddingError("Embedding dimension mismatch")
    if any(type(value) not in {int, float} for value in values):
        raise EmbeddingError("Embedding contains non-finite or invalid numbers")
    try:
        numbers = tuple(float(value) for value in values)
    except OverflowError:
        raise EmbeddingError("Embedding contains out-of-range numbers") from None
    if not all(math.isfinite(value) for value in numbers):
        raise EmbeddingError("Embedding contains non-finite or invalid numbers")
    norm = math.hypot(*numbers)
    if not math.isfinite(norm) or norm <= 0:
        raise EmbeddingError("Embedding has invalid norm")
    return tuple(value / norm for value in numbers)


def cosine(left, right) -> float:
    if len(left) != len(right) or not left:
        raise EmbeddingError("Cannot compare incompatible embeddings")
    value = math.fsum(a * b for a, b in zip(left, right))
    if not math.isfinite(value):
        raise EmbeddingError("Cannot compare non-finite embeddings")
    return max(-1., min(1., value))


@dataclass(frozen=True)
class BackendIdentity:
    model: str
    digest: str
    dimensions: int
    base_url: str
    query_prefix: str
    document_prefix: str
    num_ctx: int
    preparation_version: str = PREPARATION_VERSION

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class OllamaEmbeddings:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.base_url = local_embedding_url(cfg.base_url or ollama_host())
        self._identity = None
        self.requests = 0

    def _request(self, method, path, **kwargs):
        try:
            with network_request_limit():
                response = method(self.base_url + path, timeout=self.cfg.timeout, allow_redirects=False, **kwargs)
                self.requests += 1
            if response.status_code != 200:
                if response.status_code == 404:
                    raise EmbeddingError("Local embedding model unavailable; install it separately with ollama pull")
                raise EmbeddingError(f"Local embedding request failed (HTTP {response.status_code})")
            payload = response.json()
            if not isinstance(payload, dict):
                raise EmbeddingError("Invalid local embedding response")
            return payload
        except (requests.RequestException, ValueError):
            # Never echo server bodies, input text or contact details into logs.
            raise EmbeddingError("Local embedding service unavailable or returned invalid JSON") from None

    @property
    def identity(self) -> BackendIdentity:
        if self._identity is None:
            models = self._request(requests.get, "/api/tags").get("models")
            if not isinstance(models, list):
                raise EmbeddingError("Invalid local model inventory")
            requested = self.cfg.model if ":" in self.cfg.model else self.cfg.model + ":latest"
            model = next((model for model in models if isinstance(model, dict) and model.get("name") == requested), None)
            if model is None:
                raise EmbeddingError(f"Local embedding model {self.cfg.model} is not installed; run ollama pull separately")
            digest = model.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                raise EmbeddingError("Local model lacks an immutable digest")
            if self.cfg.expected_digest and digest != self.cfg.expected_digest:
                raise EmbeddingError("Local embedding model digest differs from configured revision")
            self._identity = BackendIdentity(requested, digest, self.cfg.dimensions, self.base_url, self.cfg.query_prefix, self.cfg.document_prefix, self.cfg.num_ctx)
        return self._identity

    def embed(self, texts: list[str], *, role: str) -> list[tuple[float, ...]]:
        if role not in {"query", "document"}:
            raise ValueError("Embedding role must be query or document")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must be nonempty text")
        identity = self.identity
        self._validate_revision(identity)
        prefix = identity.query_prefix if role == "query" else identity.document_prefix
        vectors = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = [prefix + text for text in texts[start:start + self.cfg.batch_size]]
            # One pool across local embedding backends limits peak memory/CPU competition.
            with FileSlotPool("embedding", "shared-local-embeddings", 1).lease():
                payload = self._request(requests.post, "/api/embed", json={
                    "model": identity.model, "input": batch, "truncate": False,
                    "dimensions": identity.dimensions, "keep_alive": "5m",
                    "options": {"num_ctx": identity.num_ctx, "num_thread": 2},
                })
            returned = payload.get("embeddings")
            if not isinstance(returned, list) or len(returned) != len(batch):
                raise EmbeddingError("Local embedding batch count mismatch")
            vectors.extend(unit_vector(vector, identity.dimensions) for vector in returned)
        self._validate_revision(identity)
        return vectors

    def _validate_revision(self, identity):
        models = self._request(requests.get, "/api/tags").get("models", [])
        if not isinstance(models, list) or not any(isinstance(model, dict) and model.get("name") == identity.model and model.get("digest") == identity.digest for model in models):
            raise EmbeddingError("Local embedding model changed during inference; rerun with a fixed revision")

    def resident_model(self) -> dict | None:
        """Optional benchmark snapshot, not peak RSS or total server memory."""
        identity = self.identity
        models = self._request(requests.get, "/api/ps").get("models", [])
        if not isinstance(models, list):
            return None
        for model in models:
            if isinstance(model, dict) and model.get("name") == identity.model and model.get("digest") == identity.digest:
                return {key: model.get(key) if type(model.get(key)) is int and model[key] >= 0 else None
                        for key in ("size", "size_vram", "context_length")}
        return None


class CachedEmbeddings:
    """Store-backed exact vectors; edits replace stale vectors for each entity."""
    def __init__(self, store, backend):
        self.store = store
        self.backend = backend
        self.hits = 0
        self.misses = 0

    @property
    def identity(self):
        return self.backend.identity

    def embed_entity(self, kind: str, entity_id: str, texts: list[str], *, role: str):
        identity = self.identity
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        cached = self.store.embedding_vectors(kind, entity_id, identity.key, role)
        vectors = {}
        for digest in dict.fromkeys(hashes):
            if digest in cached:
                try:
                    vectors[digest] = unit_vector(cached[digest], identity.dimensions)
                    self.hits += 1
                except EmbeddingError:
                    pass  # damaged cache entry is recomputed, never scored as zero
        missing = [digest for digest in dict.fromkeys(hashes) if digest not in vectors]
        first_text = dict(zip(hashes, texts))
        if missing:
            generated = self.backend.embed([first_text[digest] for digest in missing], role=role)
            if len(generated) != len(missing):
                raise EmbeddingError("Embedding backend count mismatch")
            vectors.update({digest: unit_vector(vector, identity.dimensions) for digest, vector in zip(missing, generated)})
            self.misses += len(missing)
        # Write only after all vectors validate; no transaction covers model inference.
        self.store.replace_embedding_vectors(kind, entity_id, identity.key, role, identity.dimensions, vectors)
        return [vectors[digest] for digest in hashes]
