from cv_agent.cli import _candidate_letter_settings, _candidate_text
from cv_agent.store import Store


def test_non_default_profile_never_falls_back_to_global_profile_yaml(monkeypatch):
    store = Store()
    store.create_account("Alice")
    store.set_candidate_profile(
        {
            "full_name": "Alice Example",
            "languages": ["Deutsch C1"],
            "preferred_letter_language": "en",
            "search_region": "Berlin",
        },
        "Alice",
    )
    store.close()
    monkeypatch.setenv("CV_AGENT_USER", "Alice")

    context = _candidate_text()
    assert "Alice Example" in context
    assert "Deutsch C1" in context
    assert "Ilia Rerich" not in context
    assert _candidate_letter_settings() == ("Alice Example", "en")
