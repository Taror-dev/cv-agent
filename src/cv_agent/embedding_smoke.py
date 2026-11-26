"""Explicit local smoke check on synthetic multilingual task passages."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .embeddings import EmbeddingError, OllamaEmbeddings, cosine
from .scoring_config import parse_scoring_config

PASSAGES = [
    "Ich habe REST-Schnittstellen entwickelt und Geschäftsanwendungen integriert.",
    "Ich habe Rechnungen bearbeitet und Termine im Büro organisiert.",
    "I maintained Linux servers and deployed applications in Docker containers.",
]
QUERIES = [
    ("Entwicklung von REST APIs zur Integration von Geschäftssystemen.", 0),
    ("Разработка API для интеграции бизнес-приложений.", 0),
    ("Administration of Linux servers and deployment with Docker.", 2),
]


def smoke(backend) -> dict:
    started = time.perf_counter()
    documents = backend.embed(PASSAGES, role="document")
    queries = backend.embed([text for text, _ in QUERIES], role="query")
    results = []
    for (text, expected), vector in zip(QUERIES, queries):
        similarities = [cosine(vector, document) for document in documents]
        nearest = max(range(len(similarities)), key=similarities.__getitem__)
        results.append({"query": text, "expected": expected, "nearest": nearest, "similarities": similarities, "passed": nearest == expected})
    return {"backend": asdict(backend.identity), "passed": all(result["passed"] for result in results),
            "elapsed_seconds": round(time.perf_counter() - started, 3), "results": results,
            "limitations": "Synthetic transport/retrieval smoke only; not validation of vacancy score quality."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--document-prefix", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        options = {"model": args.model, "dimensions": args.dimensions}
        for name in ("query_prefix", "document_prefix"):
            if getattr(args, name) is not None:
                options[name] = getattr(args, name)
        report = smoke(OllamaEmbeddings(parse_scoring_config({"embeddings": options}).embeddings))
    except (EmbeddingError, ValueError) as exc:
        parser.exit(2, f"Embedding smoke unavailable: {exc}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
