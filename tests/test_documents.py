from cv_agent.cli import _parse_keywords
import hashlib

import pytest
from reportlab.pdfgen import canvas

from cv_agent.documents import (
    EXTRACTOR_REGISTRY,
    extract_document_result,
    load_candidate_text,
    load_document_texts,
    scan_documents,
)
from cv_agent.llm_contracts import LLMContractError


def test_document_texts_all_files_in_order():
    texts = load_document_texts()
    assert len(texts) == 2  # оба PDF в cv/, без привязки к именам
    assert "Rerich" in texts[0]  # Lebenslauf всегда приоритетнее curriculum
    assert "AI" in texts[1] or "RAG" in texts[1]


def test_candidate_text_orders_documents():
    text = load_candidate_text()
    assert "Dokument [cv]" in text
    assert "Dokument [curriculum]" in text
    assert text.index("Dokument [cv]") < text.index("Dokument [curriculum]")


def test_plain_text_document_is_extracted(isolated_project):
    path = isolated_project / "cv" / "Илья" / "Notizen.txt"
    path.write_text("Erfahrung mit Python und APIs", encoding="utf-8")
    assert any("Erfahrung mit Python" in text for text in load_document_texts())


def test_total_document_budget_is_strict(isolated_project):
    folder = isolated_project / "cv" / "Budget"
    folder.mkdir()
    for index in range(3):
        (folder / f"{index}.txt").write_text("x" * 10, encoding="utf-8")
    texts = load_document_texts(user="Budget", max_chars_per_file=10, total_max=15)
    assert sum(map(len, texts)) == 15


def test_new_profile_does_not_fall_back_to_root_documents(isolated_project):
    (isolated_project / "cv" / "legacy.txt").write_text("legacy", encoding="utf-8")
    assert load_document_texts(user="Катя") == []


def test_extractor_registry_matches_upload_allowlist():
    from cv_agent.webapp import ALLOWED_DOC_EXT

    assert set(EXTRACTOR_REGISTRY) == ALLOWED_DOC_EXT == {".pdf", ".txt", ".md"}


def test_blank_pdf_reports_that_ocr_is_required(tmp_path):
    path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()
    result = extract_document_result(path, 1000)
    assert result.text == ""
    assert "OCR" in result.error


def test_invalid_text_encoding_has_explicit_error(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes(b"\xff\xfe\xfa")
    result = extract_document_result(path, 1000)
    assert result.text == ""
    assert "UTF-8" in result.error


def test_cv_is_reserved_before_long_curriculum(isolated_project):
    folder = isolated_project / "cv" / "Priority"
    folder.mkdir()
    (folder / "A Curriculum.txt").write_text("C" * 100, encoding="utf-8")
    (folder / "Z Lebenslauf.txt").write_text("V" * 100, encoding="utf-8")
    results = scan_documents(
        "Priority", max_chars_per_file=10, total_max=15, persist=False
    )
    assert [result.document_type for result in results] == ["cv", "curriculum"]
    assert results[0].text == "V" * 10
    assert results[1].text == "C" * 5
    assert sum(result.included_chars for result in results) == 15


def test_default_limits_follow_configured_backend_class(monkeypatch, isolated_project):
    """Без явных лимитов бюджет берётся из criteria.document_limits."""
    from cv_agent.config import DocumentLimits

    folder = isolated_project / "cv" / "Limits"
    folder.mkdir()
    (folder / "Lebenslauf.txt").write_text("V" * 20000, encoding="utf-8")

    # Дефолт из criteria.yaml тестового проекта: класс ollama (provider=ollama).
    results = scan_documents("Limits", persist=False)
    assert results[0].included_chars == 6000

    monkeypatch.setattr(
        "cv_agent.documents.document_limits_for",
        lambda: DocumentLimits(
            max_chars_per_file=16000, keywords=24000, scoring=32000, letter=32000
        ),
    )
    results = scan_documents("Limits", persist=False)
    assert results[0].included_chars == 16000


def test_document_metadata_contains_hash_and_usage(isolated_project):
    path = isolated_project / "cv" / "Илья" / "Notizen.md"
    path.write_text("Python APIs", encoding="utf-8")
    result = next(
        item for item in scan_documents() if item.filename == "Notizen.md"
    )
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.mime_type == "text/markdown"
    assert result.extracted_chars == len("Python APIs")
    assert result.used is True


def test_parse_keywords_de_en():
    raw = '{"de": ["Python", "RAG"], "en": ["Machine Learning"]}'
    assert _parse_keywords(raw) == {
        "de": ["Python", "RAG"],
        "en": ["Machine Learning"],
    }


def test_parse_keywords_fenced():
    raw = '```json\n{"de": ["SQL"], "en": ["Database"]}\n```'
    assert _parse_keywords(raw) == {"de": ["SQL"], "en": ["Database"]}


def test_parse_keywords_rejects_plain_array():
    with pytest.raises(LLMContractError, match="JSON object"):
        _parse_keywords('["Python", "RAG"]')


@pytest.mark.parametrize("raw", ['noch etwas "A" "B" text', "12345"])
def test_parse_keywords_rejects_garbage(raw):
    with pytest.raises(LLMContractError):
        _parse_keywords(raw)
