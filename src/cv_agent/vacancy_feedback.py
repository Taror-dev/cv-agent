"""Local provenance for informal human feedback; never calls a model/source."""
from __future__ import annotations

from .documents import SUPPORTED_DOCUMENT_EXTENSIONS, _sha256, cv_dir, infer_document_type
from .scoring_benchmark import dataset_hash
from .scoring_inputs import collect_scoring_inputs

GRADE_LABELS = {2: "Подходит", 1: "Сомнительно", 0: "Не подходит"}


def candidate_revision(store, cfg) -> str:
    """Hash actual source files/types, without PDF extraction or LLM budgets."""
    types = {item["filename"]: item["document_type"] for item in store.document_metadata()}
    folder = cv_dir(store.user)
    files = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if not path.is_symlink() and path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
                files.append({"name": path.name, "kind": types.get(path.name, infer_document_type(path.name)), "hash": _sha256(path)})
    profile = store.candidate_profile() or {}
    return dataset_hash({"version": "feedback.v1", "account_id": store.account_id(), "files": files,
                         "languages": profile.get("languages", []), "preferences": profile.get("scoring_preferences", {}),
                         "per_file": cfg.per_file, "total_chars": cfg.total_chars})


def vacancy_revision(vacancy) -> str:
    return dataset_hash({key: getattr(vacancy, key) or "" for key in ("title", "company", "location", "description")})


def capture_inputs(store, vacancy, cfg, expected_candidate: str) -> dict:
    inputs = collect_scoring_inputs(store, per_file=cfg.per_file, total=cfg.total_chars)
    if candidate_revision(store, cfg) != expected_candidate:
        raise ValueError("Candidate documents changed during review capture")
    return {"version": 1,
            "profile": {"documents": [{"id": doc.document_id, "kind": doc.kind, "text": doc.text} for doc in inputs.documents],
                        "scoring_preferences": inputs.location_preferences},
            "vacancy": {**{key: getattr(vacancy, key) or "" for key in ("refnr", "title", "company", "location", "source", "url")},
                        "text": vacancy.description or ""},
            "document_limits": {"per_file": cfg.per_file, "total_chars": cfg.total_chars},
            "warnings": list(inputs.warnings), "truncated": inputs.truncated}
