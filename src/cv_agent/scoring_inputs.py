"""Account-scoped structured inputs for rule/vector scoring.

The explicit limits bypass LLM context budgets. No provider or model is probed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .documents import scan_documents
from .scoring_rules import Document


@dataclass(frozen=True)
class ScoringInputs:
    documents: tuple[Document, ...]
    candidate_hash: str
    warnings: tuple[str, ...]
    truncated: bool
    location_preferences: dict = field(default_factory=dict)


def collect_scoring_inputs(store, *, per_file: int = 60_000, total: int = 200_000) -> ScoringInputs:
    if type(per_file) is not int or not 1 <= per_file <= 200_000:
        raise ValueError("Scoring per-file budget must be between 1 and 200000")
    if type(total) is not int or not 1 <= total <= 1_000_000:
        raise ValueError("Scoring total budget must be between 1 and 1000000")
    account_id = store.account_id()
    if not account_id:
        raise ValueError("Scoring requires an existing account")
    extracted = scan_documents(
        user=store.user, store=store,
        max_chars_per_file=per_file, total_max=total, preserve_structure=True,
    )
    documents = []
    warnings = []
    manifest = []
    for result in extracted:
        # Opaque IDs stay stable across display-name changes; no file paths leak.
        document_id = hashlib.sha256(f"{account_id}:{result.filename}".encode()).hexdigest()
        manifest.append({"id": document_id, "file_hash": result.sha256, "kind": result.document_type, "text": result.text, "error": result.extract_error, "truncated": result.truncated})
        if result.text:
            documents.append(Document(document_id, result.document_type, result.text))
        if result.extract_error:
            warnings.append(f"{result.filename}: {result.extract_error}")
        if result.truncated:
            warnings.append(f"{result.filename}: Text durch Verarbeitungslimit begrenzt.")
    profile = store.candidate_profile() or {}
    languages = profile.get("languages") or []
    profile_truncated = False
    if languages:
        profile_text = "\n".join(languages)
        remaining = max(0, total - sum(len(document.text) for document in documents))
        profile_truncated = len(profile_text) > remaining
        if profile_text[:remaining]:
            documents.append(Document("structured-profile", "profile", profile_text[:remaining]))
        if profile_truncated:
            warnings.append("Strukturiertes Profil: Text durch Verarbeitungslimit begrenzt.")
    from .scoring_location import parse_location_preferences
    preferences = parse_location_preferences(profile.get("scoring_preferences"))
    # Search radius/resume footer are not implicit commute/relocation preferences.
    manifest.append({"profile_languages": languages, "search_region": profile.get("search_region", ""), "resume_city": profile.get("resume_city", ""), "scoring_preferences": preferences})
    encoded = json.dumps({"account_id": account_id, "per_file": per_file, "total": total, "manifest": manifest}, ensure_ascii=False, sort_keys=True)
    return ScoringInputs(
        tuple(documents), hashlib.sha256(encoded.encode()).hexdigest(),
        tuple(warnings), profile_truncated or any(result.truncated for result in extracted), preferences,
    )
