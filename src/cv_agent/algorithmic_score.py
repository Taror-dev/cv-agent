"""Experimental formula: exact requirements plus semantic task comparisons.

Semantic evidence never confirms a technology, CEFR level, degree or duration.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from .embeddings import cosine
from .scoring_config import ScoringConfig
from .scoring_rules import Document, extract_facts, extract_requirements, fragments, match_requirement, rules_fingerprint
from .scoring_location import LOCATION_VERSION, evaluate_location

ALGORITHM_VERSION = "algorithmic.v3"
TASK_VERBS = re.compile(r"\b(entwickel\w*|implement\w*|integrie\w*|analys\w*|automatis\w*|betreu\w*|verwalte\w*|develop\w*|maintain\w*|design\w*|integrat\w*|automat\w*|разработ\w*|интегр\w*|анализ\w*|автоматиз\w*)\b", re.I)
WORK_ARRANGEMENT = re.compile(r"\b(remote|homeoffice|hybrid\w*|arbeitsort|standort|on[ -]site\s+only|office[ -]only|(?:nur|ausschließlich)\s+vor\s+ort|удал[её]нн\w*|гибрид\w*|только\s+в\s+офисе)\b", re.I)


def task_passages(document: Document, *, candidate: bool, max_chars: int) -> list:
    if candidate and document.kind != "cv":
        return []
    section = "unknown"
    passages = []
    for evidence in fragments(document, max_chars=max_chars):
        heading = evidence.text.lower().strip(" :#*-•")
        first, _, _ = heading.partition(":")
        if first in {"berufserfahrung", "professional experience", "work experience", "опыт работы", "aufgaben", "ihre aufgaben", "deine aufgaben", "responsibilities", "your tasks", "задачи"}:
            section = "tasks"
            if ":" not in heading:
                continue
        elif first in {"ausbildung", "weiterbildung", "education", "образование", "kenntnisse", "skills", "sprachen", "languages", "anforderungen", "ihr profil", "dein profil", "requirements", "qualifications", "требования", "wir bieten", "we offer", "benefits", "мы предлагаем", "arbeitsort", "standort", "arbeitsmodell", "work location", "work arrangement", "формат работы", "место работы"}:
            section = "other"
            continue
        if section == "tasks" or (section == "unknown" and TASK_VERBS.search(evidence.text)):
            if WORK_ARRANGEMENT.search(evidence.text) and not TASK_VERBS.search(evidence.text):
                continue
            if not re.search(r"\b(kurs\w*|course\w*|учебн\w*|курс\w*|ziel\w*|target)\b", evidence.text, re.I):
                passages.append(evidence)
    # Repeated task text does not add weight to the formula.
    return list({evidence.text.casefold(): evidence for evidence in passages}.values())


def settings_hash(cfg: ScoringConfig) -> str:
    return hashlib.sha256((cfg.fingerprint() + rules_fingerprint() + ALGORITHM_VERSION + LOCATION_VERSION).encode()).hexdigest()


def vacancy_hash(vacancy) -> str:
    payload = {key: getattr(vacancy, key, "") for key in ("refnr", "title", "location", "description")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def task_contribution(similarity: float, cfg: ScoringConfig) -> float:
    return max(0., min(1., (similarity - cfg.similarity_low) / (cfg.similarity_high - cfg.similarity_low)))


def component_score(components: dict, cfg: ScoringConfig, *, scoreable: bool, mandatory_mismatch: bool) -> tuple:
    """Shared formula for production scoring and replay of benchmark vectors."""
    raw = round(100 * sum(cfg.weights[name] * component["value"] for name, component in components.items()), 2) if scoreable else None
    return (min(raw, cfg.mandatory_mismatch_cap) if mandatory_mismatch and raw is not None else raw), raw


def evaluate_algorithmic(inputs, vacancy, cfg: ScoringConfig, embeddings) -> dict:
    description = vacancy.description[:cfg.vacancy_chars]
    requirements = extract_requirements(description)
    facts = extract_facts(inputs.documents)
    matches = [match_requirement(requirement, facts) for requirement in requirements]
    candidate = []
    limited = len(vacancy.description) > cfg.vacancy_chars or inputs.truncated
    for document in inputs.documents:
        passages = task_passages(document, candidate=True, max_chars=cfg.fragment_chars)
        remaining = cfg.max_fragments - len(candidate)
        limited |= len(passages) > remaining
        candidate.extend(passages[:remaining])
    tasks = task_passages(Document("vacancy", "other", description), candidate=False, max_chars=cfg.fragment_chars)
    limited |= len(tasks) > cfg.max_fragments
    tasks = tasks[:cfg.max_fragments]
    task_matches = []
    if candidate and tasks and cfg.weights["tasks"] > 0:
        candidate_vectors = []
        for document in inputs.documents:
            subset = [evidence for evidence in candidate if evidence.document_id == document.document_id]
            if subset:
                candidate_vectors.extend(embeddings.embed_entity("document", document.document_id, [evidence.text for evidence in subset], role="document"))
        query_vectors = embeddings.embed_entity("vacancy", vacancy.refnr, [evidence.text for evidence in tasks], role="query")
        for task, vector in zip(tasks, query_vectors):
            comparisons = [cosine(vector, other) for other in candidate_vectors]
            index = max(range(len(comparisons)), key=lambda index: (comparisons[index], -index))
            similarity = comparisons[index]
            contribution = task_contribution(similarity, cfg)
            task_matches.append({
                "task": asdict(task), "evidence": asdict(candidate[index]),
                "similarity": round(similarity, 6), "similarity_raw": similarity, "contribution": round(contribution, 6),
                "status": "semantic_candidate" if contribution else "unknown",
                "explanation": "Ähnliche Aufgabenbeschreibung; kein Nachweis einzelner Qualifikationen.",
            })
    importance = {"mandatory": 3, "unspecified": 2, "desirable": 1}
    components = {}
    for name, categories in {"skills": {"skill"}, "qualifications": {"language", "education", "experience"}}.items():
        group = [match for match in matches if match.requirement.category in categories]
        total = sum(importance[match.requirement.importance] for match in group)
        components[name] = {
            "value": sum(importance[match.requirement.importance] * match.contribution for match in group) / total if total else 0,
            "coverage": sum(importance[match.requirement.importance] for match in group if match.status in {"confirmed", "mismatch"}) / total if total else 0,
            "state": "evaluated" if total else "unknown", "requirements": len(group),
        }
    components["tasks"] = {"value": sum(match["contribution"] for match in task_matches) / len(tasks) if tasks else 0,
                           "coverage": 0., "state": "semantic" if task_matches else "unknown", "requirements": len(tasks)}
    components["location"] = evaluate_location(inputs.location_preferences, vacancy.location, description)
    # Preference-only checks cannot turn an otherwise unreadable job/CV into a score.
    known = any(components[name]["requirements"] for name in ("skills", "tasks", "qualifications"))
    state = "scored" if known and (facts or candidate) else "requirements_unavailable" if not known else "candidate_facts_unavailable"
    if not any(document.text.strip() for document in inputs.documents):
        state = "candidate_unavailable"
    capped = any(match.requirement.importance == "mandatory" and match.status == "mismatch" for match in matches)
    score, uncapped_score = component_score(components, cfg, scoreable=state == "scored", mandatory_mismatch=capped)
    for name, component in components.items():
        component["weight"] = cfg.weights[name]
        component["points"] = round(100 * cfg.weights[name] * component["value"], 2)
    coverage = round(sum(cfg.weights[name] * component["coverage"] for name, component in components.items()), 4)
    reasons = [match.explanation for match in matches if match.status == "confirmed"]
    missing = [match.explanation for match in matches if match.status != "confirmed"]
    for check in components["location"]["checks"]:
        (reasons if check["status"] == "confirmed" else missing).append(check["explanation"])
    if capped:
        missing.append("Score wegen explizitem Widerspruch zu einer Muss-Anforderung begrenzt.")
    if components["location"]["state"] == "unknown" and cfg.weights["location"]:
        missing.append("Region und Arbeitsform: keine ausdrücklichen Bewertungspräferenzen gespeichert.")
    if limited:
        missing.append("Verarbeitungslimit erreicht; manuelle Prüfung erforderlich.")
    identity = embeddings.identity
    return {
        "score": score, "uncapped_score": uncapped_score,
        "cap_applied": score is not None and score != uncapped_score,
        "state": state, "coverage": coverage,
        "preliminary": limited or coverage < 1., "version": ALGORITHM_VERSION,
        "components": components, "matches": [asdict(match) for match in matches],
        "task_matches": task_matches, "reasons": reasons, "missing": missing,
        "warnings": list(inputs.warnings),
        "backend": {**asdict(identity), "key": identity.key},
        "cache": {"hits": embeddings.hits, "misses": embeddings.misses},
        "settings_hash": settings_hash(cfg),
    }
