"""Independent acceptance examples for the dictionary baseline and extraction."""

import pytest

from cv_agent.scoring_rules import (
    Document,
    dictionary_baseline,
    extract_facts,
    extract_requirements,
    fragments,
    rules_fingerprint,
)


def cv(text):
    return Document("synthetic-cv", "cv", text)


def by_key(result):
    return {match.requirement.key: match for match in result.matches}


def test_java_is_not_javascript_and_git_is_not_github():
    result = dictionary_baseline([cv("JavaScript, GitHub, TypeScript")], "Java und Git Kenntnisse erforderlich.")
    assert {match.status for match in result.matches} == {"unknown"}
    assert result.score == 0
    assert "nicht bestätigt" in result.missing[0]


@pytest.mark.parametrize("text,key", [("C++", "C++"), ("C#", "C#"), (".NET", ".NET"), ("Postgres", "PostgreSQL"), ("NodeJS", "Node.js"), ("машинное обучение", "Machine Learning")])
def test_skill_symbols_and_exact_aliases(text, key):
    assert {fact.key for fact in extract_facts([cv(text)])} == {key}


def test_offsets_survive_whitespace_and_fragment_boundaries():
    document = cv("  Python Kenntnisse.\n\n  " + "Produktive SQL Anwendungen " * 25 + "; Deutsch B2.  ")
    chunks = fragments(document, max_chars=100)
    assert len(chunks) > 3
    for evidence in chunks:
        assert evidence.text == document.text[evidence.start:evidence.end]
        assert len(evidence.text) <= 100
        assert evidence.document_id == document.document_id
    assert "SQL" in " ".join(chunk.text for chunk in chunks)


def test_curriculum_does_not_prove_employment_language_or_degree():
    document = Document("synthetic-course", "curriculum", "Python\nDeutsch C1\nBachelor\n5 Jahre Berufserfahrung")
    result = dictionary_baseline([cv("Excel"), document], "Anforderungen:\nPython erforderlich.\nDeutsch C1 erforderlich.\nBachelor erforderlich.\n5 Jahre Berufserfahrung erforderlich.")
    matches = by_key(result)
    assert matches["Python"].status == "unknown"
    assert matches["Python"].contribution == 0.25
    assert all(matches[key].contribution == 0 for key in ("Deutsch", "Bachelor", "years"))
    assert result.preliminary


@pytest.mark.parametrize("text", ["Keine Kenntnisse in Java und Python.", "No experience with Java or Python.", "Нет опыта Java и Python."])
def test_negated_skills_are_not_matches(text):
    result = dictionary_baseline([cv(text)], "Java und Python Kenntnisse erforderlich.")
    assert {match.status for match in result.matches} == {"mismatch"}
    assert result.score == 0


def test_contrast_resets_negation():
    facts = extract_facts([cv("Kein Java, aber Python Kenntnisse.")])
    assert {fact.key: fact.basis for fact in facts} == {"Java": "negated", "Python": "stated"}


def test_requirement_not_required_does_not_become_mandatory():
    requirements = extract_requirements("Keine Java Kenntnisse erforderlich.\nDeutsch B2 erforderlich.")
    assert [requirement.key for requirement in requirements] == ["Deutsch"]


def test_employer_offers_do_not_become_candidate_requirements():
    requirements = extract_requirements("Wir bieten:\nPython Kurse\nEnglisch C1 Sprachkurse\nDein Profil:\nSQL Kenntnisse erforderlich.")
    assert [requirement.key for requirement in requirements] == ["SQL"]


def test_inline_section_heading_is_supported():
    assert [requirement.key for requirement in extract_requirements("Requirements: Python, SQL")] == ["Python", "SQL"]
    assert extract_requirements("Wir bieten: Python Kurse") == ()


def test_language_levels_are_bound_to_their_language():
    facts = extract_facts([cv("Deutsch B2 und Englisch C1")])
    assert {fact.key: fact.value for fact in facts} == {"Deutsch": "B2", "Englisch": "C1"}
    result = dictionary_baseline([cv("Deutsch B2 und Englisch C1")], "Deutsch C1 erforderlich.\nEnglisch B2 erforderlich.")
    assert by_key(result)["Deutsch"].status == "mismatch"
    assert by_key(result)["Englisch"].status == "confirmed"


def test_unquantified_language_is_not_inferred_from_adjacent_language():
    facts = extract_facts([cv("Englisch B2, Deutsch gut")])
    assert [fact.key for fact in facts] == ["Englisch"]
    result = dictionary_baseline([cv("Deutsch gut, Python")], "Deutsch C1 erforderlich.")
    assert by_key(result)["Deutsch"].status == "unknown"


def test_level_interval_is_conservative_and_reverse_level_supported():
    facts = extract_facts([cv("Deutsch B2–C1\nB1 Englisch")])
    assert {fact.key: fact.value for fact in facts} == {"Deutsch": "B2", "Englisch": "B1"}


def test_conflicting_languages_require_review_not_best_level():
    result = dictionary_baseline([cv("Deutsch B1"), Document("profile", "profile", "Deutsch C1")], "Deutsch B2 erforderlich.")
    assert by_key(result)["Deutsch"].status == "conflict"
    assert result.score == 0
    assert result.preliminary


def test_unknown_requirements_do_not_increase_score_or_trigger_rejection():
    known = dictionary_baseline([cv("Python")], "Python erforderlich.")
    unknown = dictionary_baseline([cv("Python")], "Python erforderlich.\nDocker erforderlich.")
    assert known.score == 100
    assert unknown.score == 50
    assert unknown.coverage == 0.5
    assert by_key(unknown)["Docker"].status == "unknown"
    assert unknown.state == "scored"


def test_optional_requirements_have_lower_weight():
    result = dictionary_baseline([cv("Python")], "Python erforderlich; Docker wünschenswert.")
    assert result.score == 75


def test_keyword_repetition_does_not_inflate_score():
    description = "Python erforderlich.\nSQL erforderlich."
    first = dictionary_baseline([cv("Python")], description)
    repeated = dictionary_baseline([cv("Python " * 20)], description + "\nPython erforderlich." * 20)
    assert first.score == repeated.score == 50
    assert len(repeated.matches) == 2


def test_experience_dates_do_not_prove_technology_specific_years():
    result = dictionary_baseline([cv("2010–2025 Softwareentwickler\nPython Kenntnisse\n10 Jahre Berufserfahrung")], "5 Jahre Erfahrung mit Python erforderlich.")
    assert by_key(result)["years"].status == "unknown"
    assert by_key(result)["years"].contribution == 0


def test_explicit_technology_experience_is_compared_without_calendar_inference():
    result = dictionary_baseline([cv("5 Jahre Erfahrung mit Python")], "3 Jahre Erfahrung mit Python erforderlich.")
    assert by_key(result)["years"].status == "confirmed"
    general = dictionary_baseline([cv("10 Jahre Berufserfahrung\nPython Kenntnisse")], "3 Jahre Erfahrung mit Python erforderlich.")
    assert by_key(general)["years"].status == "unknown"


def test_experience_does_not_transfer_between_technologies():
    result = dictionary_baseline([cv("5 Jahre Erfahrung mit Java\nPython Kenntnisse")], "3 Jahre Erfahrung mit Python erforderlich.")
    assert by_key(result)["years"].status == "unknown"


def test_degree_alternatives_are_not_treated_as_all_required():
    result = dictionary_baseline([cv("Bachelor abgeschlossen")], "Bachelor oder Master erforderlich.")
    assert len(result.matches) == 1
    assert result.matches[0].status == "confirmed"
    assert result.score == 100


def test_not_only_does_not_negate_a_skill():
    result = dictionary_baseline([cv("Nicht nur Python, sondern auch SQL Kenntnisse.")], "Python erforderlich.")
    assert result.score == 100


def test_experience_interval_uses_lower_bound():
    result = dictionary_baseline([cv("3–5 Jahre Erfahrung mit Python")], "5 Jahre Erfahrung mit Python erforderlich.")
    assert by_key(result)["years"].status == "mismatch"


def test_target_language_level_is_not_a_current_qualification():
    result = dictionary_baseline([cv("Python\nZielniveau Deutsch C1")], "Deutsch C1 erforderlich.")
    assert by_key(result)["Deutsch"].status == "unknown"


def test_links_and_email_addresses_do_not_prove_skills():
    facts = extract_facts([cv("Portfolio: https://example.test/python\nMail: java@example.test\nSQL")])
    assert [fact.key for fact in facts] == ["SQL"]


def test_course_in_cv_remains_study_evidence():
    result = dictionary_baseline([cv("Python Kurs abgeschlossen")], "Python erforderlich.")
    assert by_key(result)["Python"].status == "unknown"
    assert by_key(result)["Python"].contribution == 0.25


def test_incomplete_degree_does_not_prove_graduation():
    result = dictionary_baseline([cv("Bachelor Studium laufend\nPython")], "Bachelor erforderlich.")
    assert by_key(result)["Bachelor"].status == "unknown"


def test_other_document_mentions_do_not_prove_candidate_skills():
    result = dictionary_baseline([cv("Excel"), Document("misc", "other", "Java, Python")], "Python erforderlich.")
    assert by_key(result)["Python"].contribution == 0


@pytest.mark.parametrize("documents,description,state", [([], "Python erforderlich.", "candidate_unavailable"), ([cv(" \n")], "Python erforderlich.", "candidate_unavailable"), ([cv("Python")], "Wir bieten Obst und Kaffee.", "requirements_unavailable"), ([cv("Unlesbarer Inhalt")], "Python erforderlich.", "candidate_facts_unavailable")])
def test_missing_inputs_are_not_zero_scores(documents, description, state):
    result = dictionary_baseline(documents, description)
    assert result.score is None
    assert result.state == state


def test_deterministic_serializable_results_have_evidence():
    documents = [cv("Python SQL\nDeutsch B2")]
    first = dictionary_baseline(documents, "Python erforderlich. Deutsch B2 erforderlich.")
    second = dictionary_baseline(documents, "Python erforderlich. Deutsch B2 erforderlich.")
    assert first.to_dict() == second.to_dict()
    assert first.matches[0].evidence[0].document_id == "synthetic-cv"
    assert len(rules_fingerprint()) == 64
