from dataclasses import replace

import pytest

from cv_agent.algorithmic_score import evaluate_algorithmic
from cv_agent.embeddings import CachedEmbeddings
from cv_agent.scoring_config import ScoringConfig
from cv_agent.scoring_location import evaluate_location, observed_work_mode, parse_location_preferences
from cv_agent.store import Store
from test_algorithmic_score import make_inputs, vacancy
from test_embeddings import FixedBackend


@pytest.mark.parametrize("text,expected", [
    ("100% remote", "remote"), ("Fully remote work", "remote"),
    ("Vollständig im Homeoffice", "remote"), ("Полностью удалённая работа", "remote"),
    ("Hybrid working", "hybrid"), ("Hybrides Arbeiten", "hybrid"), ("Гибридный формат", "hybrid"),
    ("On-site only", "onsite"), ("Ausschließlich vor Ort", "onsite"), ("Только в офисе", "onsite"),
    ("Homeoffice möglich", "unknown"), ("Remote", "unknown"), ("Kein 100% remote", "unknown"),
    ("Not fully remote", "unknown"), ("Возможна полностью удалённая работа", "unknown"),
    ("Hybrid, not fully remote", "unknown"), ("Not hybrid", "unknown"),
    ("Fully remote or hybrid", "conflict"),
    ("Build hybrid cloud platforms", "unknown"),
    ("Up to 100% remote", "unknown"), ("Bis zu 100% Homeoffice", "unknown"),
    ("Fully remote after probation", "unknown"), ("100% remote if approved", "unknown"),
])
def test_explicit_multilingual_work_mode_and_ambiguity(text, expected):
    assert observed_work_mode(text)[0] == expected


def test_no_preferences_stay_unknown_and_do_not_infer_commute():
    assert evaluate_location({}, "Wangen", "100% remote")["state"] == "unknown"
    result = evaluate_location({"cities": ["Wangen"], "work_mode": "any"}, "Ravensburg", "100% remote")
    assert result["value"] == .5
    assert result["checks"][1]["status"] == "unknown"


def test_exact_city_matches_normalized_names_but_not_substrings_or_translations():
    prefs = {"cities": ["Wangen im Allgäu", "München"], "work_mode": "hybrid"}
    assert evaluate_location(prefs, "88239 WANGEN IM ALLGÄU, Deutschland", "Hybrid working")["value"] == 1
    assert evaluate_location(prefs, "München-West", "Hybrid working")["value"] == .5
    assert evaluate_location(prefs, "Munich", "Hybrid working")["value"] == .5
    assert evaluate_location(prefs, "", "Hybrid working")["value"] == .5
    assert evaluate_location(prefs, "München / Berlin", "Hybrid working")["value"] == 1


def test_explicit_mode_mismatch_does_not_impose_mandatory_cap():
    result = evaluate_location({"work_mode": "remote"}, "Berlin", "On-site only")
    assert result["coverage"] == 1
    assert result["value"] == .5
    assert result["checks"][0]["status"] == "mismatch"
    assert evaluate_location({"work_mode": "any"}, "", "")["value"] == 1


@pytest.mark.parametrize("raw", [
    {"work_mode": "cloud"}, {"work_mode": []}, {"cities": "Berlin"}, {"radius": 50},
    {"cities": ["Berlin\nMünchen"]}, {"cities": ["Berlin, München"]}, {"cities": ["a" * 121]},
    {"cities": ["12345"]}, {"cities": ["Berlin"] * 21},
])
def test_preferences_reject_malformed_or_unbounded_values(raw):
    with pytest.raises(ValueError):
        parse_location_preferences(raw)


def test_preferences_deduplicate_city_case_and_postcode():
    assert parse_location_preferences({"cities": ["Berlin", "10115 BERLIN"]})["cities"] == ["Berlin"]


def test_location_enters_formula_with_explicit_evidence_and_version(tmp_path):
    store = Store(tmp_path / "scores.db")
    try:
        inputs = replace(make_inputs(), location_preferences={"cities": ["Wangen"], "work_mode": "hybrid"})
        job = vacancy()
        job.description += "\nHybrid working"
        result = evaluate_algorithmic(inputs, job, ScoringConfig(), CachedEmbeddings(store, FixedBackend()))
        assert result["score"] == 100
        assert result["components"]["location"]["coverage"] == 1
        assert result["components"]["location"]["checks"][1]["evidence"] == ["Wangen"]
        assert result["version"] == "algorithmic.v3"
        assert result["preliminary"]  # Semantic tasks remain candidates for manual review.
        job.description += "\nNot fully remote"
        ambiguous = evaluate_algorithmic(inputs, job, ScoringConfig(), CachedEmbeddings(store, FixedBackend()))
        assert ambiguous["score"] == 95
    finally:
        store.close()


def test_location_alone_does_not_create_score_from_unreadable_cv_or_job(tmp_path):
    store = Store(tmp_path / "scores.db")
    try:
        inputs = replace(make_inputs("Document with no usable qualifications"), location_preferences={"work_mode": "any"})
        result = evaluate_algorithmic(inputs, vacancy("General company information."), ScoringConfig(), CachedEmbeddings(store, FixedBackend()))
        assert result["score"] is None
        assert result["state"] == "requirements_unavailable"
    finally:
        store.close()


def test_work_arrangements_are_not_added_to_semantic_tasks():
    from cv_agent.algorithmic_score import task_passages
    from cv_agent.scoring_rules import Document
    document = Document("job", "other", "Responsibilities:\nDevelop software\nHybrid working\nHomeoffice möglich\n100% remote\nWork location:\nBerlin")
    assert [item.text for item in task_passages(document, candidate=False, max_chars=700)] == ["Develop software"]


def test_office_tasks_are_not_mistaken_for_work_arrangements():
    from cv_agent.algorithmic_score import task_passages
    from cv_agent.scoring_rules import Document
    document = Document("job", "other", "Responsibilities:\nOffice appointment scheduling\nOn-site equipment maintenance\nЗадачи:\nОрганизация календаря офиса\nТолько в офисе")
    assert [item.text for item in task_passages(document, candidate=False, max_chars=700)] == ["Office appointment scheduling", "On-site equipment maintenance", "Организация календаря офиса"]
