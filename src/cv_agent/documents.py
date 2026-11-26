from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .config import ROOT, current_user, document_limits_for
from .storage import documents_dir

CV_DIR = ROOT / "cv"
MAX_CHARS_PER_FILE = 6000
TOTAL_MAX_CHARS = 12000
MAX_EXTRACT_SOURCE_BYTES = 10 * 1024 * 1024
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})
DOCUMENT_TYPES = frozenset({"cv", "curriculum", "certificate", "other"})
DOCUMENT_TYPE_PRIORITY = {"cv": 0, "curriculum": 1, "certificate": 2, "other": 3}
# Легаси-дефолты (до document_limits в criteria.yaml); остаются валидатором purpose.
CONTEXT_CHAR_BUDGETS = {"keywords": 8000, "scoring": 12000, "letter": 12000}


@dataclass
class Extraction:
    text: str = ""
    error: str = ""
    truncated: bool = False


@dataclass
class DocumentResult:
    filename: str
    document_type: str
    type_source: str
    mime_type: str
    size_bytes: int
    sha256: str
    extracted_at: str
    extracted_chars: int
    included_chars: int
    text: str
    extract_error: str
    truncated: bool = False

    @property
    def used(self) -> bool:
        return self.included_chars > 0


def cv_dir(
    user: str | None = None, *, legacy_root_fallback: bool = True
) -> Path:
    """Legacy cv/<name> до миграции; UUID-каталог после storage marker."""
    user = user or current_user()
    return documents_dir(user, CV_DIR, legacy_root_fallback=legacy_root_fallback)


def _normalized(text: str, max_chars: int, *, preserve_structure: bool = False) -> str:
    if preserve_structure:
        return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()[:max(0, max_chars)]
    return " ".join(text.split())[: max(0, max_chars)]


def _extract_pdf(path: Path, max_chars: int, *, preserve_structure: bool = False) -> Extraction:
    try:
        reader = PdfReader(str(path))
        raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return Extraction(error="PDF повреждён или не читается.")
    normalized = _normalized(raw, len(raw), preserve_structure=preserve_structure)
    text = normalized[:max(0, max_chars)]
    if not text:
        return Extraction(
            error="Требуется OCR: PDF не содержит извлекаемого текстового слоя."
        )
    return Extraction(text=text, truncated=len(normalized) > max_chars)


def _extract_utf8(path: Path, max_chars: int, *, preserve_structure: bool = False) -> Extraction:
    try:
        if path.stat().st_size > MAX_EXTRACT_SOURCE_BYTES:
            return Extraction(error="Файл превышает безопасный лимит извлечения 10 МБ.")
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeError:
        return Extraction(error="Текстовый файл должен быть в UTF-8.")
    except OSError:
        return Extraction(error="Файл не удалось прочитать.")
    normalized = _normalized(text, len(text), preserve_structure=preserve_structure)
    text = normalized[:max(0, max_chars)]
    if not text:
        return Extraction(error="Документ не содержит текста.")
    return Extraction(text=text, truncated=len(normalized) > max_chars)


Extractor = Callable[[Path, int], Extraction]
EXTRACTOR_REGISTRY: dict[str, Extractor] = {
    ".pdf": _extract_pdf,
    ".txt": _extract_utf8,
    ".md": _extract_utf8,
}


def extract_document_result(path: Path, max_chars: int, *, preserve_structure: bool = False) -> Extraction:
    if not path.is_file() or path.is_symlink():
        return Extraction(error="Документ не является обычным файлом.")
    extractor = EXTRACTOR_REGISTRY.get(path.suffix.lower())
    if extractor is None:
        return Extraction(error="Формат документа не поддерживается.")
    if preserve_structure and extractor in (_extract_pdf, _extract_utf8):
        return extractor(path, max_chars, preserve_structure=True)
    return extractor(path, max_chars)


def extract_document(path: Path, max_chars: int) -> str:
    """Совместимый API: только извлечённый текст без metadata."""
    return extract_document_result(path, max_chars).text


def infer_document_type(filename: str) -> str:
    name = filename.casefold()
    if any(word in name for word in ("lebenslauf", "resume", "résumé", "резюме")):
        return "cv"
    if any(word in name for word in ("curriculum", "syllabus", "lehrplan", "program")):
        return "curriculum"
    if any(
        word in name for word in ("zertifikat", "certificate", "zeugnis", "сертификат")
    ):
        return "certificate"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stored_document_types(user: str) -> tuple[dict[str, tuple[str, str]], object | None]:
    from .store import Store

    store = Store(user=user)
    if not store.account_id(user):
        store.close()
        return {}, None
    return {
        row["filename"]: (row["document_type"], row["type_source"])
        for row in store.document_metadata()
    }, store


def scan_documents(
    user: str | None = None,
    *,
    purpose: str = "scoring",
    max_chars_per_file: int | None = None,
    total_max: int | None = None,
    persist: bool = True,
    store=None,
    preserve_structure: bool = False,
) -> list[DocumentResult]:
    """Извлекает, приоритизирует и при необходимости сохраняет metadata.

    Без явных лимитов значения берутся из criteria.document_limits
    по классу текущего LLM-бэкенда (external/ollama).
    """
    user = user or current_user()
    if purpose not in CONTEXT_CHAR_BUDGETS:
        raise ValueError("Неизвестный контекст документов")
    if max_chars_per_file is None or total_max is None:
        limits = document_limits_for()
        if max_chars_per_file is None:
            max_chars_per_file = limits.max_chars_per_file
        if total_max is None:
            total_max = limits.budget(purpose)
    budget = max(0, int(total_max))
    folder = cv_dir(user)
    if not folder.is_dir():
        return []
    owned_store = None
    if persist and store is None:
        stored, owned_store = _stored_document_types(user)
        metadata_store = owned_store
    elif persist and store is not None and store.account_id(user):
        stored = {
            row["filename"]: (row["document_type"], row["type_source"])
            for row in store.document_metadata()
        }
        metadata_store = store
    else:
        stored, metadata_store = {}, None
    candidates = []
    try:
        for path in folder.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in EXTRACTOR_REGISTRY:
                continue
            stored_type, type_source = stored.get(
                path.name, (infer_document_type(path.name), "auto")
            )
            candidates.append((path, stored_type, type_source))
        candidates.sort(
            key=lambda item: (
                DOCUMENT_TYPE_PRIORITY.get(item[1], DOCUMENT_TYPE_PRIORITY["other"]),
                item[0].name.casefold(),
            )
        )
        results = []
        for path, document_type, type_source in candidates:
            if preserve_structure:
                extraction = extract_document_result(path, max_chars_per_file, preserve_structure=True)
            else:
                extraction = extract_document_result(path, max_chars_per_file)
            included = min(len(extraction.text), budget)
            budget -= included
            result = DocumentResult(
                filename=path.name,
                document_type=document_type,
                type_source=type_source,
                mime_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                extracted_at=datetime.now(timezone.utc).isoformat(),
                extracted_chars=len(extraction.text),
                included_chars=included,
                text=extraction.text[:included],
                extract_error=extraction.error,
                truncated=extraction.truncated or included < len(extraction.text),
            )
            results.append(result)
            if metadata_store is not None:
                metadata_store.upsert_document_metadata(
                    {
                        "filename": result.filename,
                        "document_type": result.document_type,
                        "type_source": result.type_source,
                        "mime_type": result.mime_type,
                        "size_bytes": result.size_bytes,
                        "sha256": result.sha256,
                        "extracted_chars": result.extracted_chars,
                        "extract_error": result.extract_error,
                    }
                )
        return results
    finally:
        if owned_store is not None:
            owned_store.close()


def load_document_texts(
    user: str | None = None,
    max_chars_per_file: int | None = None,
    total_max: int | None = None,
    *,
    purpose: str = "scoring",
) -> list[str]:
    return [
        result.text
        for result in scan_documents(
            user,
            purpose=purpose,
            max_chars_per_file=max_chars_per_file,
            total_max=total_max,
        )
        if result.text
    ]


def load_candidate_text(
    user: str | None = None, *, purpose: str = "scoring"
) -> str:
    parts = []
    for result in scan_documents(user, purpose=purpose):
        if result.text:
            parts.append(
                f"Dokument [{result.document_type}] {result.filename}:\n{result.text}"
            )
    return "\n\n".join(parts)
