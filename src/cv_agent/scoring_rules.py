"""Pure, source-backed extraction and dictionary scoring baseline.

No database, network, generative model or user documents are read here. The
baseline is deliberately conservative; it is not the production score engine.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

RULES_VERSION = "rules.v1"
BASELINE_VERSION = "dictionary.v1"

# Canonical keys are stable; aliases describe exact equivalences, not inference.
SKILLS = {
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript", "java script"),
    "TypeScript": ("typescript",),
    "Node.js": ("node.js", "nodejs"),
    "SQL": ("sql",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Git": ("git",),
    "Linux": ("linux",),
    "C++": ("c++",),
    "C#": ("c#",),
    ".NET": (".net", "dotnet"),
    "React": ("react", "react.js", "reactjs"),
    "Angular": ("angular",),
    "Spring Boot": ("spring boot", "springboot"),
    "FastAPI": ("fastapi",),
    "Django": ("django",),
    "REST API": ("rest api", "rest-api", "restful api", "rest-schnittstellen"),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "Machine Learning": ("machine learning", "maschinelles lernen", "машинное обучение"),
    "SAP": ("sap",),
    "Excel": ("excel",),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure",),
    "Systemintegration": ("systemintegration", "system integration", "интеграция систем"),
    "Automatisierung": ("automatisierung", "automation", "автоматизация"),
}
LANGUAGES = {
    "Deutsch": ("deutsch", "german", "немецкий", "немецкого"),
    "Englisch": ("englisch", "english", "английский", "английского"),
    "Russisch": ("russisch", "russian", "русский", "русского"),
}
EDUCATION = {
    "Bachelor": ("bachelor", "b.sc.", "бакалавр"),
    "Master": ("master", "m.sc.", "магистр"),
    "Ausbildung": ("berufsausbildung", "ausbildung", "профессиональное образование"),
}
LEVELS = {level: i for i, level in enumerate(("A1", "A2", "B1", "B2", "C1", "C2"))}
MANDATORY = re.compile(r"\b(muss|zwingend|erforderlich|voraussetzung|required|must|mandatory|обязательно|требуется)\b", re.I)
OPTIONAL = re.compile(r"\b(wünschenswert|idealerweise|von vorteil|optional|preferred|nice.to.have|желательно)\b", re.I)
NEGATION = re.compile(r"\b(kein\w*|nicht|ohne|no|not|without|нет|без|не)\b", re.I)
NO_REQUIREMENT = re.compile(
    r"\b(nicht erforderlich|nicht notwendig|nicht vorausgesetzt|not required|not necessary|не требуется)\b", re.I
)
LINKS = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[\w-]+", re.I)
YEARS = re.compile(
    r"(?P<years>\d{1,2})(?:\s*[–-]\s*\d{1,2})?\s*\+?\s*(?:jahre[n]?\s*(?:berufs)?erfahrung|years?\s*(?:of\s*)?(?:professional\s*)?experience|лет\s*опыта|года?\s*опыта)", re.I
)


@dataclass(frozen=True)
class Document:
    document_id: str
    kind: str
    text: str

    def __post_init__(self):
        if self.kind not in {"cv", "curriculum", "certificate", "other", "profile"}:
            raise ValueError("Unsupported document kind")
        if not self.document_id.strip():
            raise ValueError("Document ID must not be empty")


@dataclass(frozen=True)
class Evidence:
    document_id: str
    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Fact:
    category: str
    key: str
    value: str | float
    basis: str  # stated / studied / mentioned / negated
    evidence: Evidence
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class Requirement:
    category: str
    key: str
    value: str | float
    importance: str  # mandatory / desirable / unspecified
    evidence: Evidence
    scope: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class Match:
    requirement: Requirement
    status: str  # confirmed / mismatch / unknown / conflict
    contribution: float
    evidence: tuple[Evidence, ...]
    explanation: str


@dataclass(frozen=True)
class BaselineResult:
    score: float | None
    state: str
    coverage: float
    preliminary: bool
    matches: tuple[Match, ...]
    reasons: tuple[str, ...]
    missing: tuple[str, ...]
    version: str = BASELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=256)
def _pattern(alias: str) -> re.Pattern:
    # Symbols are part of C++/C#/.NET; Java must not match JavaScript.
    return re.compile(r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])", re.I)


def _hits(text: str, aliases: Iterable[str]) -> list[re.Match]:
    text = LINKS.sub(lambda match: " " * len(match.group()), text)
    return sorted(
        (match for alias in aliases for match in _pattern(alias).finditer(text)),
        key=lambda match: (match.start(), match.end()),
    )


def fragments(document: Document, *, max_chars: int = 700) -> tuple[Evidence, ...]:
    """Bounded clauses with offsets into the *original* text, never fabricated pages."""
    if not 80 <= max_chars <= 4000:
        raise ValueError("Fragment limit must be between 80 and 4000")
    result = []
    # Semicolons delimit independently qualified requirements; decimal/tool dots do not.
    for span in re.finditer(r"[^\n;]+", document.text):
        for clause in re.finditer(r".+?(?:[.!?](?=\s+[A-ZА-ЯÄÖÜ])|$)", span.group()):
            start, end = span.start() + clause.start(), span.start() + clause.end()
            while start < end:
                stop = min(start + max_chars, end)
                if stop < end:
                    boundary = document.text.rfind(" ", start + max_chars // 2, stop)
                    if boundary > start:
                        stop = boundary
                left, right = start, stop
                while left < right and document.text[left].isspace():
                    left += 1
                while right > left and document.text[right - 1].isspace():
                    right -= 1
                if left < right:
                    result.append(Evidence(document.document_id, document.kind, left, right, document.text[left:right]))
                start = stop
    return tuple(result)


def _negated(text: str, position: int) -> bool:
    # Scope resets at contrast/conjunctions, to avoid "no Java, but Python".
    prefix = re.split(r"\b(?:aber|but|sondern|но)\b|[,;]", text[:position], flags=re.I)[-1]
    prefix = re.sub(r"\b(?:nicht nur|not only|не только)\b", "", prefix, flags=re.I)
    return bool(NEGATION.search(prefix[-70:]))


def _level(text: str, language_position: int, language_end: int) -> str | None:
    # A level must belong to this language, not the next one in the same line.
    tail = text[language_end:]
    next_language = [m.start() for aliases in LANGUAGES.values() for m in _hits(tail, aliases)]
    if next_language:
        tail = tail[:min(next_language)]
    levels = re.findall(r"\b[ABC][12]\b", tail[:55], re.I)
    if not levels:
        # Also accept "B2 Deutsch" without borrowing another language's level.
        before = text[max(0, language_position - 5):language_position]
        match = re.search(r"\b([ABC][12])\s*$", before, re.I)
        levels = [match[1]] if match else []
    # B2–C1 conservatively means at least B2, not verified C1.
    return min((level.upper() for level in levels), key=LEVELS.get) if levels else None


def _experience_scope(text: str, years: re.Match) -> tuple[str, ...]:
    """Only explicit 'years experience with Python', not an adjacent skill list."""
    tail = text[years.end():]
    if not re.match(r"\s*(?:mit|in|with|using|с|в)\s+", tail, re.I):
        return ()
    return tuple(key for key, aliases in SKILLS.items() if _hits(tail, aliases))


def extract_facts(documents: Iterable[Document]) -> tuple[Fact, ...]:
    facts = []
    for document in documents:
        for evidence in fragments(document):
            basis = "studied" if document.kind == "curriculum" else "mentioned" if document.kind == "other" else "stated"
            if re.search(r"\b((?:sprach)?kurs\w*|schulung\w*|lernprogramm|curriculum|course\w*|учебн\w*|курс\w*)\b", evidence.text, re.I):
                basis = "studied"
            for key, aliases in SKILLS.items():
                hits = _hits(evidence.text, aliases)
                if hits:
                    negated = all(_negated(evidence.text, hit.start()) for hit in hits)
                    facts.append(Fact("skill", key, key, "negated" if negated else basis, evidence))
            # A course syllabus is never evidence of the candidate's language/degree/stage.
            if document.kind in {"curriculum", "other"} or basis == "studied":
                continue
            for key, aliases in LANGUAGES.items():
                for hit in _hits(evidence.text, aliases)[:1]:
                    level = _level(evidence.text, hit.start(), hit.end())
                    aspirational = re.search(r"\b(ziel\w*|angestrebt|geplant|target|aim|планирую|цель)\b", evidence.text, re.I)
                    if level and not aspirational:
                        facts.append(Fact("language", key, level, "negated" if _negated(evidence.text, hit.start()) else "stated", evidence))
            for key, aliases in EDUCATION.items():
                hits = _hits(evidence.text, aliases)
                if hits:
                    incomplete = re.search(r"\b(studium|student|studying|laufend|in progress|abgebrochen|незаверш|учусь)\w*\b", evidence.text, re.I)
                    completed = re.search(r"\b(abgeschlossen|abschluss|graduated|completed|degree|окончил|диплом)\w*\b", evidence.text, re.I)
                    if not incomplete or completed:
                        facts.append(Fact("education", key, key, "negated" if _negated(evidence.text, hits[0].start()) else "stated", evidence))
            # Only an explicit amount; calendar dates do not establish relevant experience.
            years = YEARS.search(evidence.text)
            if years and document.kind in {"cv", "profile"}:
                facts.append(Fact("experience", "years", float(years["years"]), "negated" if _negated(evidence.text, years.start()) else "stated", evidence, _experience_scope(evidence.text, years)))
    return tuple(facts)


def extract_requirements(description: str) -> tuple[Requirement, ...]:
    result = []
    section = "unknown"
    for evidence in fragments(Document("vacancy", "other", description)):
        text = evidence.text
        heading = text.lower().strip(" :#*-•")
        # "Requirements: Python ..." must switch scope and retain the content.
        heading_part, separator, _ = heading.partition(":")
        scope_heading = heading_part.strip() if separator else heading
        # Headings carry scope across lines; employer offers must not become requirements.
        if scope_heading in {"wir bieten", "we offer", "benefits", "unser angebot", "мы предлагаем"}:
            section = "offer"
            continue
        if scope_heading in {"anforderungen", "ihr profil", "dein profil", "requirements", "qualifications", "требования"}:
            section = "requirements"
            if not separator:
                continue
        if scope_heading in {"aufgaben", "deine aufgaben", "ihre aufgaben", "responsibilities", "your tasks", "задачи"}:
            section = "tasks"
            continue
        if section == "offer" or re.match(r"(?:wir bieten|we offer|мы предлагаем)\b", text, re.I):
            continue
        if NO_REQUIREMENT.search(text):
            continue
        # Mixed importance in one clause is ambiguous, not automatically optional.
        importance = "unspecified" if OPTIONAL.search(text) and MANDATORY.search(text) else "desirable" if OPTIONAL.search(text) else "mandatory" if MANDATORY.search(text) else "unspecified"
        requirement_context = section == "requirements" or bool(re.search(
            r"\b(erfahrung|kenntnisse|beherrsch|mitbringen|experience|knowledge|proficien|опыт|знани|владени)\w*\b", text, re.I
        )) or importance != "unspecified"
        if not requirement_context:
            continue
        for key, aliases in SKILLS.items():
            hits = _hits(text, aliases)
            if hits and not all(_negated(text, hit.start()) for hit in hits):
                result.append(Requirement("skill", key, key, importance, evidence))
        for key, aliases in LANGUAGES.items():
            for hit in _hits(text, aliases)[:1]:
                if not _negated(text, hit.start()):
                    level = _level(text, hit.start(), hit.end())
                    result.append(Requirement("language", key, level or "unspecified", importance, evidence))
        degrees = tuple(key for key, aliases in EDUCATION.items() if any(not _negated(text, hit.start()) for hit in _hits(text, aliases)))
        if len(degrees) > 1 and re.search(r"\b(oder|or|либо|или)\b", text, re.I):
            label = " / ".join(degrees)
            result.append(Requirement("education", label, label, importance, evidence, alternatives=degrees))
        else:
            result.extend(Requirement("education", key, key, importance, evidence) for key in degrees)
        years = YEARS.search(text)
        if years and not _negated(text, years.start()):
            result.append(Requirement("experience", "years", float(years["years"]), importance, evidence, _experience_scope(text, years)))
    # Repeated wording must not increase a requirement's weight. Keep strictest requirement.
    dedup = {}
    importance_order = {"unspecified": 0, "desirable": 1, "mandatory": 2}
    for requirement in result:
        key = (requirement.category, requirement.key, requirement.scope)
        previous = dedup.get(key)
        if previous is None or (
            importance_order[requirement.importance],
            LEVELS.get(requirement.value, -1) if requirement.category == "language" else requirement.value if requirement.category == "experience" else 0,
        ) > (
            importance_order[previous.importance],
            LEVELS.get(previous.value, -1) if previous.category == "language" else previous.value if previous.category == "experience" else 0,
        ):
            dedup[key] = requirement
    return tuple(dedup.values())


def match_requirement(requirement: Requirement, facts: Iterable[Fact]) -> Match:
    candidates = tuple(
        fact for fact in facts
        if fact.category == requirement.category
        and fact.key in (requirement.alternatives or (requirement.key,))
        and (requirement.category != "experience" or not requirement.scope or set(requirement.scope).issubset(fact.scope))
    )
    evidence = tuple(dict.fromkeys(fact.evidence for fact in candidates))
    stated = tuple(fact for fact in candidates if fact.basis == "stated")
    negative = tuple(fact for fact in candidates if fact.basis == "negated")
    status, contribution = "unknown", 0.0
    if stated and negative:
        status = "conflict"
    elif negative:
        status = "mismatch"
    elif stated:
        if requirement.category == "language":
            values = {fact.value for fact in stated}
            if len(values) > 1:
                status = "conflict"
            elif requirement.value in LEVELS:
                status = "confirmed" if LEVELS[next(iter(values))] >= LEVELS[requirement.value] else "mismatch"
        elif requirement.category == "experience":
            values = {fact.value for fact in stated}
            if len(values) > 1:
                status = "conflict"
            else:
                status = "confirmed" if next(iter(values)) >= requirement.value else "mismatch"
        else:
            status = "confirmed"
        contribution = 1.0 if status == "confirmed" else 0.0
    elif any(fact.basis == "studied" for fact in candidates) and requirement.category == "skill":
        contribution = 0.25  # explicitly labelled study evidence, never professional experience
    label = requirement.key + (f" {requirement.value}" if requirement.category == "language" else "")
    if requirement.category == "experience":
        label = f"{requirement.value:g} Jahre Erfahrung" + (f" ({', '.join(requirement.scope)})" if requirement.scope else "")
    explanations = {
        "confirmed": f"{label}: durch Unterlagen bestätigt.",
        "mismatch": f"{label}: explizite Angabe erfüllt die Anforderung nicht.",
        "conflict": f"{label}: widersprüchliche Angaben; manuell prüfen.",
        "unknown": f"{label}: durch Unterlagen nicht bestätigt; manuell prüfen.",
    }
    explanation = explanations[status]
    if contribution == 0.25:
        explanation += " Nur im Lernprogramm erwähnt."
    return Match(requirement, status, contribution, evidence, explanation)


def dictionary_baseline(documents: Iterable[Document], description: str) -> BaselineResult:
    """Baseline weighted requirement coverage, *not* the planned final formula."""
    documents = tuple(documents)
    if not any(document.text.strip() for document in documents):
        return BaselineResult(None, "candidate_unavailable", 0, True, (), (), ())
    requirements = extract_requirements(description)
    if not requirements:
        return BaselineResult(None, "requirements_unavailable", 0, True, (), (), ())
    facts = extract_facts(documents)
    if not facts:
        return BaselineResult(None, "candidate_facts_unavailable", 0, True, (), (), ())
    matches = tuple(match_requirement(requirement, facts) for requirement in requirements)
    weights = {"mandatory": 3, "unspecified": 2, "desirable": 1}
    total = sum(weights[match.requirement.importance] for match in matches)
    score = round(100 * sum(weights[match.requirement.importance] * match.contribution for match in matches) / total, 2)
    coverage = sum(weights[match.requirement.importance] for match in matches if match.status in {"confirmed", "mismatch"}) / total
    return BaselineResult(
        score, "scored", round(coverage, 4), coverage < 1,
        matches,
        tuple(match.explanation for match in matches if match.status == "confirmed"),
        tuple(match.explanation for match in matches if match.status != "confirmed"),
    )


def rules_fingerprint() -> str:
    """Explicit extraction vocabulary/version for future cache invalidation."""
    payload = json.dumps({"version": RULES_VERSION, "skills": SKILLS, "languages": LANGUAGES, "education": EDUCATION}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
