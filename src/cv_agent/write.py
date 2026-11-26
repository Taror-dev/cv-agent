import re
from pathlib import Path

from .config import APPLICATIONS_DIR, current_user
from .llm import LLMClient
from .llm_contracts import (
    LETTER_PROMPT_VERSION,
    LETTER_SYSTEM,
    LLMContractError,
    letter_prompt,
)
from .models import Vacancy
from .storage import applications_dir

SYSTEM = LETTER_SYSTEM
PROMPT_VERSION = LETTER_PROMPT_VERSION

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_slug(raw: str, max_length: int = 60) -> str:
    """Безопасное имя файла/каталога: [A-Za-z0-9_-] ограниченной длины.

    Сырой refnr или название компании в путь без sanitization не попадают.
    """
    slug = _SLUG_RE.sub("_", str(raw))
    slug = re.sub(r"_{2,}", "_", slug).strip("_-")
    return slug[:max_length].rstrip("_-") or "stelle"


def validate_letter_identity(
    text: str, own_name: str, other_names: list[str]
) -> None:
    """Anschreiben должен содержать имя своего профиля и чужих — не contain.

    Имена-подстроки своего имени (например «Илья» при «Илья Рерих») чужими
    не считаются — семейные профили часто перекрываются (аудит M1).
    """
    if not str(text).strip():
        raise LLMContractError("Empty letter will not be saved")
    own = str(own_name or "").strip()
    if own and own not in text:
        raise LLMContractError(
            f"Letter does not contain the profile name ({own})"
        )
    for other in other_names:
        other = str(other or "").strip()
        if (
            not other
            or other == own
            or other in own
            or own in other
        ):
            continue
        if other in text:
            raise LLMContractError(
                f"Letter contains another profile's name ({other})"
            )


def letter(
    v: Vacancy,
    candidate: str,
    llm: LLMClient,
    candidate_name: str | None = None,
    preferred_language: str = "de",
) -> str:
    profile_name = (candidate_name or current_user()).strip()
    language = {"de": "Deutsch", "en": "Englisch", "ru": "Russisch"}.get(
        preferred_language, "Deutsch"
    )
    result = llm.complete(
        SYSTEM,
        letter_prompt(v, candidate, profile_name, language),
    ).strip()
    if not result:
        raise LLMContractError("LLM returned an empty letter")
    return result


def _folder_name(v: Vacancy) -> str:
    return safe_slug(v.company or v.refnr)


_PDF_CHAR_REPLACEMENTS = {
    "\u2011": "-", "\u2010": "-",
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2026": "...", "\u200b": "", "\ufeff": "",
}


def _pdf_safe_text(text: str) -> str:
    """Заменяет типографику вне WinAnsi и выкидывает некодируемое (без □)."""
    replaced = "".join(_PDF_CHAR_REPLACEMENTS.get(ch, ch) for ch in text)
    safe = []
    for ch in replaced:
        if ch == "\n":
            safe.append(ch)
            continue
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            continue
        safe.append(ch)
    return "".join(safe)


def render_letter_pdf(text: str) -> bytes:
    """Anschreiben-Text -> A4-PDF (Platypus, Helvetica, Absätze aus Leerzeilen).

    Zeilen innerhalb eines Blocks bleiben Umbrüche (<br/>), **fett** wird
    unterstützt; alles andere bleibt schlichter Fließtext.
    """
    import re
    from io import BytesIO
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    def inline(line: str) -> str:
        escaped = escape(line.strip())
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    style = ParagraphStyle(
        "Letter", fontName="Helvetica", fontSize=11, leading=15.5
    )
    story = []
    block: list[str] = []

    def flush() -> None:
        if block:
            story.append(Paragraph("<br/>".join(inline(l) for l in block), style))
            block.clear()

    for raw in _pdf_safe_text(text).replace("\r\n", "\n").split("\n"):
        if raw.strip():
            block.append(raw)
        else:
            flush()
            if story:
                story.append(Spacer(1, 8))
    flush()
    if not story:
        raise LLMContractError("Empty letter will not be saved")

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.25 * cm,
        rightMargin=2.25 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Anschreiben",
    )
    doc.build(story)
    return buf.getvalue()


def save_letter(v: Vacancy, text: str) -> Path:
    """Speichert das Anschreiben als PDF neben den Dossiers."""
    if not text.strip():
        raise LLMContractError("Empty letter will not be saved")
    root = applications_dir(current_user(), APPLICATIONS_DIR)
    out = root / f"Anschreiben_{_folder_name(v)}_{safe_slug(v.refnr, 40)}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_letter_pdf(text))
    return out
