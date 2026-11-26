import re

from .config import Criteria, ExclusionRules
from .llm import LLMClient
from .llm_contracts import (
    SCORING_PROMPT_VERSION,
    SCORING_SCHEMA,
    SCORING_SYSTEM,
    complete_json_contract,
    scoring_prompt,
    validate_score,
)
from .models import Vacancy

SYSTEM = SCORING_SYSTEM
PROMPT_VERSION = SCORING_PROMPT_VERSION


def _rules(criteria: Criteria) -> dict[str, ExclusionRules]:
    exclusions = getattr(criteria, "exclusions", None)
    if exclusions:
        return exclusions
    legacy = [
        value for value in getattr(criteria, "exclude_keywords", []) if value.strip()
    ]
    return {
        "title": ExclusionRules(words=legacy),
        "company": ExclusionRules(),
        "description": ExclusionRules(words=legacy),
    }


def _match_rules(text: str, rules: ExclusionRules) -> str:
    for word in rules.words:
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE):
            return f"excluded:word:{word}"[:200]
    for pattern in rules.regex:
        if re.search(pattern, text, re.IGNORECASE):
            return f"excluded:regex:{pattern}"[:200]
    return ""


def prefilter_reason_scopes(
    v: Vacancy, criteria: Criteria, scopes: tuple[str, ...]
) -> str:
    """Причина исключения по выбранным полям (без обращения к description)."""
    fields = {
        "title": v.title,
        "company": v.company,
        "description": v.description,
    }
    for scope in scopes:
        rules = _rules(criteria).get(scope)
        if rules is None:
            continue
        hit = _match_rules(str(fields.get(scope, "") or ""), rules)
        if hit:
            return f"{hit.split(':', 1)[0]}:{scope}:{hit.split(':', 1)[-1]}"[:200]
    return ""


def prefilter_reason(v: Vacancy, criteria: Criteria) -> str:
    fields = {
        "title": v.title,
        "company": v.company,
        "description": v.description,
    }
    for scope, rules in _rules(criteria).items():
        text = str(fields.get(scope, "") or "")
        hit = _match_rules(text, rules)
        if hit:
            return f"excluded:{scope}:{hit.split(':', 1)[-1]}"[:200]
    return ""


def prefilter(v: Vacancy, criteria: Criteria) -> bool:
    return not prefilter_reason(v, criteria)


def scoreability_reason(v: Vacancy) -> str:
    title = " ".join(str(v.title or "").split())
    if len(title) < 3:
        return "title_unavailable"
    description = " ".join(str(v.description or "").split())
    if len(description) < 20 or len(re.findall(r"\w+", description)) < 3:
        return "description_unavailable"
    return ""


def score_one(v: Vacancy, candidate: str, llm: LLMClient) -> tuple[float, list[str], list[str]]:
    return complete_json_contract(
        llm,
        SYSTEM,
        scoring_prompt(v, candidate),
        schema_name="vacancy_score",
        schema=SCORING_SCHEMA,
        validator=validate_score,
    )
