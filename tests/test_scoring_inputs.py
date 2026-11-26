import pytest
from reportlab.pdfgen import canvas

from cv_agent.documents import extract_document_result
from cv_agent.scoring_inputs import collect_scoring_inputs
from cv_agent.scoring_rules import dictionary_baseline, extract_facts
from cv_agent.store import Store


def profile_documents(root, name, text, filename="Lebenslauf.txt"):
    folder = root / "cv" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text, encoding="utf-8")


def test_new_inputs_preserve_lines_and_bypass_llm_budget(isolated_project, monkeypatch):
    store = Store(user="Synthetic")
    store.create_account("Synthetic")
    profile_documents(isolated_project, "Synthetic", "Python\n" + "Projektarbeit\n" * 1000 + "Deutsch B2")
    monkeypatch.setattr("cv_agent.documents.document_limits_for", lambda: pytest.fail("New input budgets must not depend on LLM configuration"))
    try:
        inputs = collect_scoring_inputs(store)
        assert len(inputs.documents[0].text) > 6000
        assert "\n" in inputs.documents[0].text
        assert "Deutsch B2" in inputs.documents[0].text
        assert not inputs.truncated
        assert dictionary_baseline(inputs.documents, "Deutsch B2 erforderlich.").score == 100
    finally:
        store.close()


def test_profiles_do_not_share_documents_hashes_or_languages(isolated_project):
    first = Store(user="SyntheticOne")
    first.create_account("SyntheticOne")
    second = Store(user="SyntheticTwo")
    second.create_account("SyntheticTwo")
    profile_documents(isolated_project, "SyntheticOne", "Python")
    profile_documents(isolated_project, "SyntheticTwo", "Java")
    first.set_candidate_profile({"languages": ["Deutsch B2"], "email": "private@example.test"})
    try:
        a, b = collect_scoring_inputs(first), collect_scoring_inputs(second)
        assert a.candidate_hash != b.candidate_hash
        assert {fact.key for fact in extract_facts(a.documents)} == {"Python", "Deutsch"}
        assert {fact.key for fact in extract_facts(b.documents)} == {"Java"}
        assert all("private@example.test" not in document.text for document in a.documents)
    finally:
        first.close()
        second.close()


def test_manual_type_beats_filename_and_changes_hash(isolated_project):
    store = Store(user="Synthetic")
    store.create_account("Synthetic")
    profile_documents(isolated_project, "Synthetic", "Python", "Curriculum.txt")
    try:
        original = collect_scoring_inputs(store)
        assert original.documents[0].kind == "curriculum"
        assert store.set_document_type("Curriculum.txt", "cv")
        changed = collect_scoring_inputs(store)
        assert changed.documents[0].kind == "cv"
        assert changed.candidate_hash != original.candidate_hash
    finally:
        store.close()


def test_document_edits_profile_edits_and_deletion_change_hash(isolated_project):
    store = Store(user="Synthetic")
    store.create_account("Synthetic")
    profile_documents(isolated_project, "Synthetic", "Python")
    try:
        initial = collect_scoring_inputs(store)
        assert collect_scoring_inputs(store).candidate_hash == initial.candidate_hash
        profile_documents(isolated_project, "Synthetic", "Java")
        changed = collect_scoring_inputs(store)
        assert changed.candidate_hash != initial.candidate_hash
        store.set_candidate_profile({"languages": ["Deutsch B1"]})
        with_language = collect_scoring_inputs(store)
        assert with_language.candidate_hash != changed.candidate_hash
        (isolated_project / "cv" / "Synthetic" / "Lebenslauf.txt").unlink()
        assert collect_scoring_inputs(store).candidate_hash != with_language.candidate_hash
    finally:
        store.close()


def test_strict_independent_budget_records_truncation(isolated_project):
    store = Store(user="Synthetic")
    store.create_account("Synthetic")
    profile_documents(isolated_project, "Synthetic", "Python " * 100)
    profile_documents(isolated_project, "Synthetic", "SQL " * 100, "Curriculum.txt")
    try:
        inputs = collect_scoring_inputs(store, per_file=100, total=120)
        assert sum(len(document.text) for document in inputs.documents) == 120
        assert inputs.truncated
        assert len(inputs.warnings) == 2
    finally:
        store.close()


def test_structured_profile_also_respects_total_budget(isolated_project):
    store = Store(user="Synthetic")
    store.create_account("Synthetic")
    profile_documents(isolated_project, "Synthetic", "Python")
    store.set_candidate_profile({"languages": ["Deutsch B2", "Englisch C1"]})
    try:
        inputs = collect_scoring_inputs(store, per_file=100, total=10)
        assert sum(len(document.text) for document in inputs.documents) == 10
        assert inputs.truncated
        assert inputs.warnings
    finally:
        store.close()


def test_empty_profile_has_no_legacy_family_documents(isolated_project):
    store = Store(user="SyntheticEmpty")
    store.create_account("SyntheticEmpty")
    try:
        inputs = collect_scoring_inputs(store)
        assert inputs.documents == ()
        assert dictionary_baseline(inputs.documents, "Python erforderlich.").state == "candidate_unavailable"
    finally:
        store.close()


def test_structured_pdf_retains_line_boundaries(tmp_path):
    path = tmp_path / "synthetic.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 780, "Python")
    pdf.drawString(72, 760, "Deutsch B2")
    pdf.save()
    structured = extract_document_result(path, 1000, preserve_structure=True)
    assert "\n" in structured.text
    assert extract_document_result(path, 1000).text == "Python Deutsch B2"


@pytest.mark.parametrize("per_file,total", [(0, 100), (100, 0), (True, 100), (100, 1_000_001)])
def test_input_budgets_are_validated(per_file, total):
    with pytest.raises(ValueError):
        collect_scoring_inputs(None, per_file=per_file, total=total)
