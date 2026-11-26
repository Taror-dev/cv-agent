"""Validated server-side settings for experimental algorithmic scoring."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "qwen3-embedding:0.6b"
    base_url: str = ""  # empty -> OLLAMA_HOST, independent of generative LLM settings
    expected_digest: str = ""
    dimensions: int = 1024
    batch_size: int = 8
    timeout: float = 120.0
    num_ctx: int = 4096
    query_prefix: str = "Instruct: Given a job responsibility, retrieve candidate experience passages describing similar work.\nQuery: "
    document_prefix: str = ""


@dataclass(frozen=True)
class ScoringConfig:
    engine: str = "llm"
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    weights: dict[str, float] = field(default_factory=lambda: {"skills": .4, "tasks": .3, "qualifications": .2, "location": .1})
    similarity_low: float = .45
    similarity_high: float = .85
    mandatory_mismatch_cap: float = 49.0
    per_file: int = 60_000
    total_chars: int = 200_000
    vacancy_chars: int = 60_000
    fragment_chars: int = 700
    max_fragments: int = 256

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local_embedding_url(url: str) -> str:
    """Accept only explicitly configured local/LAN native Ollama endpoints."""
    try:
        parsed = urlsplit(url)
        _ = parsed.port
        host = parsed.hostname
        local = host == "localhost"
        if host and not local:
            address = ipaddress.ip_address(host)
            local = (address.is_private or address.is_loopback) and not (address.is_link_local or address.is_multicast or address.is_unspecified)
        if not local or parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError
    except ValueError:
        raise ValueError("Embedding URL must be a local/LAN Ollama root URL") from None
    return url.rstrip("/")


def parse_scoring_config(raw: object) -> ScoringConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(ScoringConfig.__dataclass_fields__):
        raise ValueError("Invalid criteria.scoring fields")
    values = dict(raw)
    embedding = values.pop("embeddings", {})
    if not isinstance(embedding, dict) or set(embedding) - set(EmbeddingConfig.__dataclass_fields__):
        raise ValueError("Invalid criteria.scoring.embeddings fields")
    defaults = EmbeddingConfig()
    for key in ("model", "base_url", "expected_digest", "query_prefix", "document_prefix"):
        value = embedding.get(key, getattr(defaults, key))
        if not isinstance(value, str) or len(value) > 2000 or (key == "model" and (not value.strip() or value.endswith(":cloud"))):
            raise ValueError(f"Invalid scoring.embeddings.{key}")
    if embedding.get("base_url"):
        local_embedding_url(embedding["base_url"])
    limits = {"dimensions": (1, 4096), "batch_size": (1, 32), "num_ctx": (512, 32768)}
    for key, (low, high) in limits.items():
        value = embedding.get(key, getattr(defaults, key))
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid scoring.embeddings.{key}")
    timeout = embedding.get("timeout", defaults.timeout)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError("Invalid scoring.embeddings.timeout")
    cfg = ScoringConfig(embeddings=EmbeddingConfig(**embedding), **values)
    if not isinstance(cfg.engine, str) or cfg.engine not in {"llm", "algorithmic"}:
        raise ValueError("scoring.engine must be llm or algorithmic")
    if not isinstance(cfg.weights, dict) or set(cfg.weights) != {"skills", "tasks", "qualifications", "location"}:
        raise ValueError("Invalid scoring.weights components")
    if any(type(weight) not in {int, float} or not math.isfinite(weight) or not 0 <= weight <= 1 for weight in cfg.weights.values()) or not math.isclose(sum(cfg.weights.values()), 1, abs_tol=1e-9):
        raise ValueError("Scoring weights must be finite, nonnegative and sum to 1")
    for key in ("similarity_low", "similarity_high", "mandatory_mismatch_cap"):
        value = getattr(cfg, key)
        maximum = 100 if key == "mandatory_mismatch_cap" else 1
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Invalid scoring.{key}")
    if cfg.similarity_low >= cfg.similarity_high:
        raise ValueError("Scoring similarity_low must be less than similarity_high")
    for key, bounds in {"per_file": (1, 200_000), "total_chars": (1, 1_000_000), "vacancy_chars": (100, 200_000), "fragment_chars": (80, 2000), "max_fragments": (1, 1024)}.items():
        if type(getattr(cfg, key)) is not int or not bounds[0] <= getattr(cfg, key) <= bounds[1]:
            raise ValueError(f"Invalid scoring.{key}")
    return cfg
