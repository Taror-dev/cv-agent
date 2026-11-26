"""Conservative, offline checks of explicitly saved location preferences.

No geocoding, radius calculation or residence/work-permit inference. A city
match proves only that the advertised location is in the candidate's list.
"""
from __future__ import annotations

import re
import unicodedata

LOCATION_VERSION = "location.v1"
WORK_MODES = frozenset({"unspecified", "any", "remote", "hybrid", "onsite"})


def normalize_city(value: str) -> str:
    text = unicodedata.normalize("NFC", value).casefold().strip()
    return re.sub(r"\s+", " ", re.sub(r"^\d{4,6}\s+", "", text))


def parse_location_preferences(raw: object) -> dict:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - {"work_mode", "cities"}:
        raise ValueError("Некорректные предпочтения места работы.")
    mode = raw.get("work_mode", "unspecified")
    if not isinstance(mode, str) or mode not in WORK_MODES:
        raise ValueError("Некорректный формат работы.")
    cities = raw.get("cities", [])
    if not isinstance(cities, list) or len(cities) > 20:
        raise ValueError("Укажите не более 20 городов.")
    cleaned = {}
    for city in cities:
        if not isinstance(city, str) or not city.strip() or len(city) > 120 or any(ord(char) < 32 for char in city) or any(char in city for char in ",;|/"):
            raise ValueError("Некорректное название города (до 120 символов).")
        if not normalize_city(city) or not any(char.isalpha() for char in city):
            raise ValueError("Некорректное название города.")
        cleaned.setdefault(normalize_city(city), city.strip())
    return {"work_mode": mode, "cities": list(cleaned.values())}


# Bare 'remote' and 'Homeoffice möglich' do not promise fully remote work.
# Markers are deliberately specific; ambiguous/negated clauses stay unknown.
MODE_PATTERNS = {
    "remote": re.compile(r"\b(?:fully\s+remote|remote[ -]only|vollständig\s+(?:remote|im\s+homeoffice)|полностью\s+удал[её]нн\w*)\b|\b100\s*%\s*(?:remote|homeoffice|удал[её]нн\w*)", re.I),
    "hybrid": re.compile(r"\b(?:hybrid\w*\s+(?:work\w*|arbeit\w*|model\w*|schedule|format)|(?:work\w*|arbeit\w*)\s*:?\s*hybrid\w*|гибрид\w*\s+(?:формат|работ\w*|режим)|(?:or|oder|или)\s+hybrid(?!\s+cloud)\b)|^\s*(?:hybrid|гибридный)\s*$", re.I),
    "onsite": re.compile(r"\b(?:on[ -]site\s+only|office[ -]only|ausschließlich\s+vor\s+ort|nur\s+vor\s+ort|только\s+в\s+офисе)\b", re.I),
}
NEGATION = re.compile(r"\b(?:no|not|kein\w*|nicht|ohne|не|нет|без)\b", re.I)
UNCERTAIN = re.compile(r"\b(?:option\w*|possib\w*|potential\w*|möglich\w*|optional\w*|eventuell|возмож\w*|планиру\w*|up\s+to|bis\s+zu|до|after|nach|после|if|wenn|sofern|если)\b", re.I)


def observed_work_mode(text: str) -> tuple[str, list[str]]:
    found = set()
    evidence = []
    ambiguous = False
    # Keep negations and qualifiers in the same clause, including punctuation
    # after a percentage. Bounded excerpts are also used for explanations.
    for clause in re.split(r"[\n.!?;]+", text):
        for mode, pattern in MODE_PATTERNS.items():
            if pattern.search(clause):
                evidence.append(clause.strip()[:700])
                if NEGATION.search(clause) or UNCERTAIN.search(clause):
                    ambiguous = True
                else:
                    found.add(mode)
    if ambiguous or len(found) > 1:
        return "conflict" if len(found) > 1 else "unknown", evidence
    return next(iter(found), "unknown"), evidence


def evaluate_location(preferences: dict, location: str, description: str) -> dict:
    prefs = parse_location_preferences(preferences)
    mode, cities = prefs["work_mode"], prefs["cities"]
    if mode == "unspecified" and not cities:
        return {"value": 0., "coverage": 0., "state": "unknown", "requirements": 0, "checks": [], "version": LOCATION_VERSION}
    checks = []
    observed, evidence = observed_work_mode(description)
    if mode == "any":
        status, value = "confirmed", 1.
        explanation = "Alle Arbeitsformen ausdrücklich akzeptiert."
    elif mode == "unspecified":
        status, value = "unknown", 0.
        explanation = "Bevorzugte Arbeitsform nicht angegeben."
    elif observed in MODE_PATTERNS:
        status = "confirmed" if mode == observed else "mismatch"
        value = float(status == "confirmed")
        explanation = f"Arbeitsform: gewünscht {mode}, ausdrücklich angeboten {observed}."
    else:
        status, value = observed, 0.
        explanation = "Arbeitsform nicht eindeutig belegt; manuelle Prüfung erforderlich."
    checks.append({"key": "work_mode", "status": status, "value": value, "explanation": explanation, "evidence": evidence, "preference": mode})
    advertised = {normalize_city(part) for part in re.split(r"[,;|/\n]+", location) if part.strip()}
    matched = [city for city in cities if normalize_city(city) in advertised]
    if not cities and mode != "unspecified":
        status, value = "confirmed", 1.
        explanation = "Keine Einschränkung auf bestimmte Städte ausdrücklich gewählt."
    elif matched:
        status, value = "confirmed", 1.
        explanation = "Angegebener Arbeitsort entspricht einer gewählten Stadt: " + ", ".join(matched) + "."
    else:
        status, value = "unknown", 0.
        explanation = "Arbeitsort nicht eindeutig einer gewählten Stadt zugeordnet; Entfernung und Remote-Geltungsbereich nicht geprüft."
    checks.append({"key": "cities", "status": status, "value": value, "explanation": explanation, "evidence": [location] if location else [], "preference": cities})
    return {
        "value": sum(check["value"] for check in checks) / 2,
        "coverage": sum(check["status"] in {"confirmed", "mismatch"} for check in checks) / 2,
        "state": "evaluated", "requirements": len(checks), "checks": checks,
        "version": LOCATION_VERSION,
    }
