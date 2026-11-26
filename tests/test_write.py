import cv_agent.write as write
import pytest
from cv_agent.llm_contracts import LLMContractError
from cv_agent.models import Vacancy


def test_save_letter(tmp_path, monkeypatch):
    monkeypatch.setattr(write, "APPLICATIONS_DIR", tmp_path)
    v = Vacancy("9", "AI Engineer", "ACME GmbH", "Wangen", "u", "ba")
    out = write.save_letter(v, "Sehr geehrte Damen und Herren,")
    assert out.exists()
    assert "ACME_GmbH" in out.name
    assert "9" in out.name


def test_save_letter_fallback_name(tmp_path, monkeypatch):
    monkeypatch.setattr(write, "APPLICATIONS_DIR", tmp_path)
    v = Vacancy("9", "AI Engineer", "", "Wangen", "u", "ba")
    out = write.save_letter(v, "text")
    assert "9" in out.name


def test_letter_uses_current_profile_name_without_hardcoded_person():
    captured = {}

    class FakeLLM:
        def complete(self, system, prompt):
            captured["system"] = system
            captured["prompt"] = prompt
            return "Anschreiben"

    v = Vacancy("9", "AI Engineer", "ACME", "Wangen", "u", "ba")
    assert write.letter(
        v, "Lebenslauf von Alice Example", FakeLLM(), candidate_name="Alice"
    ) == "Anschreiben"
    combined = captured["system"] + captured["prompt"]
    assert "Ilia Rerich" not in combined
    assert "Alice" in captured["prompt"]
    assert "Lebenslauf von Alice Example" in captured["prompt"]


def test_letter_respects_profile_language():
    captured = {}

    class FakeLLM:
        def complete(self, system, prompt):
            captured["system"] = system
            captured["prompt"] = prompt
            return "Letter"

    v = Vacancy("9", "AI Engineer", "ACME", "Wangen", "u", "ba")
    write.letter(
        v,
        "Alice CV",
        FakeLLM(),
        candidate_name="Alice Example",
        preferred_language="en",
    )
    assert "Englisch" in captured["prompt"]


def test_empty_letter_is_an_error_and_is_not_saved(tmp_path, monkeypatch):
    applications = tmp_path / "letters"
    monkeypatch.setattr(write, "APPLICATIONS_DIR", applications)

    class FakeLLM:
        def complete(self, system, prompt):
            return "   \n"

    v = Vacancy("9", "AI Engineer", "ACME", "Wangen", "u", "ba")
    with pytest.raises(LLMContractError, match="empty letter"):
        write.letter(v, "Alice CV", FakeLLM())
    with pytest.raises(LLMContractError, match="Empty letter"):
        write.save_letter(v, "\n")
    assert not applications.exists()


def test_safe_slug_limits_charset_and_length():
    from cv_agent.write import safe_slug

    assert safe_slug("ACME GmbH") == "ACME_GmbH"
    assert safe_slug('Очень длинная компания "/\\:*?<>|') == "stelle"
    assert safe_slug("ACME / GmbH") == "ACME_GmbH"
    assert safe_slug("a/b\\c:d") == "a_b_c_d"
    long = "x" * 500
    assert len(safe_slug(long)) == 60
    assert safe_slug("", 40) == "stelle"
    assert safe_slug("///", 40) == "stelle"
    refnr = "ss/" + "../evil"
    assert "/" not in safe_slug(refnr, 40)
    assert ".." not in safe_slug(refnr, 40)


def test_validate_letter_identity_blocks_wrong_names():
    from cv_agent.write import validate_letter_identity

    # своё имя присутствует, чужих нет — ок
    validate_letter_identity("Mit freundlichen Grüßen, Anna M.", "Anna M.", ["Ilia"])
    # чужое имя в письме — блокируется
    with pytest.raises(Exception, match="another profile's name"):
        validate_letter_identity(
            "Gruß, Ilia Rerich und Anna M.", "Anna M.", ["Ilia Rerich"]
        )
    # своего имени нет — блокируется
    with pytest.raises(Exception, match="does not contain the profile name"):
        validate_letter_identity("Sehr geehrte…", "Anna M.", [])
    # пустой текст — блокируется
    with pytest.raises(Exception, match="[Ee]mpty letter"):
        validate_letter_identity("  ", "Anna M.", [])


def test_validate_letter_identity_allows_substring_family_names():
    from cv_agent.write import validate_letter_identity

    # «Илья» — префикс собственного «Илья Рерих»: чужим не считается
    validate_letter_identity(
        "Gruß, Илья Рерих", "Илья Рерих", ["Илья", "Anna"]
    )
    # но настоящее чужое полное имя по-прежнему блокируется
    with pytest.raises(Exception, match="another profile's name"):
        validate_letter_identity(
            "Gruß, Илья Рерих и Anna M.", "Илья Рерих", ["Anna M."]
        )


def test_save_letter_writes_valid_pdf(tmp_path, monkeypatch):
    import io

    from pypdf import PdfReader

    monkeypatch.setattr(write, "APPLICATIONS_DIR", tmp_path)
    v = Vacancy("int-9", "Job", "ACME GmbH", "Wangen", "u", "ba")
    out = write.save_letter(
        v,
        "Betreff: Bewerbung als Python-Entwickler\n\n"
        "Sehr geehrte Damen und Herren,\n\n"
        "mit großem Interesse habe ich Ihre Stellung gesehen. **Kurz** zu mir.",
    )
    assert out.suffix == ".pdf"
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    text = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "Betreff: Bewerbung" in text
    assert "großem Interesse" in text  # умлауты не сломаны
    assert "Sehr geehrte" in text


def test_render_letter_pdf_keeps_line_breaks_in_block():
    import io

    from pypdf import PdfReader

    from cv_agent.write import render_letter_pdf

    data = render_letter_pdf("Zeile A\nZeile B\n\nAbsatz zwei")
    text = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "Zeile A" in text and "Zeile B" in text
    assert "Absatz zwei" in text


def test_render_letter_pdf_sanitizes_unencodable_characters():
    import io

    from pypdf import PdfReader

    from cv_agent.write import render_letter_pdf

    text = (
        "Zusammenarbeit\u2011nah \u202fbitte\u00a0danke "
        "\u201eZitate\u201c \u2013 Gedankenstrich"
    )
    data = render_letter_pdf(text)
    extracted = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "\u25a0" not in extracted  # никаких чёрных квадратов
    assert "Zusammenarbeit-nah" in extracted.replace("\n", "")
    assert "bitte danke" in extracted.replace("\n", "")
    assert "\u2013" not in extracted and "-" in extracted


def test_render_letter_pdf_drops_non_latin_scripts():
    import io

    from pypdf import PdfReader

    from cv_agent.write import render_letter_pdf

    data = render_letter_pdf("Sehr geehrte Damen und Herren привет ✓")
    text = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "\u25a0" not in text
    assert "Sehr geehrte" in text


def test_letter_prompt_v3_excludes_internal_refnr():
    """Внутренний refnr не попадает в промпт письма (риск «Kennziffer: ss-…»)."""
    from cv_agent.llm_contracts import LETTER_PROMPT_VERSION, letter_prompt

    v = Vacancy("ss-1a53ab13", "Python Dev", "ACME", "Wangen", "u", "stepstone")
    v.description = "Wir suchen."
    prompt = letter_prompt(v, "CV", "Ilia Rerich", "Deutsch")
    assert "ss-1a53ab13" not in prompt
    assert "refnr" not in prompt
    assert LETTER_PROMPT_VERSION == "letter.v4"


def test_letter_system_v2_covers_article_rules():
    from cv_agent.llm_contracts import LETTER_SYSTEM

    for fragment in (
        "Muss-Kriterien",
        "3-5 stärksten",
        "Hiermit bewerbe ich mich",
        "Ansprechperson",
        "170–230",
        "niemals erscheinen",
        "wörtlich",
    ):
        assert fragment in LETTER_SYSTEM


def test_letter_system_v4_forbids_hallucinations_and_limits_length():
    from cv_agent.llm_contracts import LETTER_SYSTEM

    for fragment in (
        "Anti-Halluzinations",
        "ausschließlich Tatsachen",
        "Outlook",
        "internationales Unternehmen",
        "1400 Zeichen",
        "halbe Seite",
    ):
        assert fragment in LETTER_SYSTEM
