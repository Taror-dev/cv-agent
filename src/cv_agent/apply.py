import io
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from .config import APPLICATIONS_DIR, DEFAULT_USER, ROOT, current_user
from .documents import cv_dir
from .models import Vacancy
from .storage import applications_dir
from .write import _folder_name, safe_slug

CV_PDF = ROOT / "cv" / "Lebenslauf-Rerich-Ilia.pdf"
_DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}")
_BOTTOM_BAND = 0.25  # нижняя четверть страницы считается footer-областью


def _cv_pdf(user: str | None = None) -> Path:
    """Первый подходящий PDF из активного каталога документов профиля."""
    user = user or current_user()
    folder = cv_dir(user)
    if folder.is_dir():
        candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        if candidates:
            preferred = [
                p for p in candidates
                if any(k in p.name.lower() for k in ("lebenslauf", "cv", "резюме"))
            ]
            return sorted(preferred or candidates)[0]
    if user == DEFAULT_USER and CV_PDF.is_file():
        return CV_PDF
    return folder


_FONT_ALIASES = {
    "arial": "Helvetica",
    "helvetica": "Helvetica",
    "liberationsans": "Helvetica",
    "calibri": "Helvetica",
    "carlito": "Helvetica",
    "dejavusans": "Helvetica",
    "times": "Times-Roman",
    "timesroman": "Times-Roman",
    "timesnewroman": "Times-Roman",
    "liberationserif": "Times-Roman",
    "georgia": "Times-Roman",
    "courier": "Courier",
    "couriernew": "Courier",
}


def _metric_font(font_dict) -> str:
    """Ближайший стандартный PDF-шрифт для расчёта ширины (аудит M4)."""
    try:
        raw = str(font_dict.get("/BaseFont", "") if font_dict else "")
    except Exception:
        return "Helvetica"
    name = raw.split("/")[-1].split("+")[-1]  # срезаем subset-префикс
    return _FONT_ALIASES.get(
        re.sub(r"[^A-Za-z]", "", name).lower(), "Helvetica"
    )


def _string_width(text: str, font_name: str, size: float) -> float:
    try:
        return stringWidth(text, font_name, size)
    except Exception:
        return stringWidth(text, "Helvetica", size)


def _embedded_text_width(font_dict, text: str, size: float) -> float | None:
    """Точная ширина строки по /Widths встроенного simple-шрифта.

    Подменные метрики (Helvetica за Calibri и т.п.) систематически врут,
    из-за чего белая плашка поверх старой даты смещалась. Ширины из
    самого PDF точны; при их отсутствии возвращается None (fallback).
    """
    try:
        if hasattr(font_dict, "get_object"):
            font_dict = font_dict.get_object()
        widths = font_dict.get("/Widths")
        if not widths:
            return None
        width_list = [float(w) for w in widths.get_object()]
        first = int(font_dict.get("/FirstChar", 0))
        missing = float(font_dict.get("/MissingWidth") or 0.0)
        total = 0.0
        for ch in text:
            idx = ord(ch) - first
            total += (
                width_list[idx]
                if 0 <= idx < len(width_list)
                else missing
            )
        return total * size / 1000.0
    except Exception:
        return None


def _find_date_pos(page) -> tuple[float, float, float, str, str, float] | None:
    """Самый нижний date-hit в проверяемой footer-области — или None.

    Возвращает (x, y, font_size, найденный_текст, метрический_шрифт,
    ширина_найденной_даты); ширина считается по /Widths встроенного
    шрифта (точно) или подстановными метриками (fallback).
    """
    hits: list[tuple[float, float, float, str, str, float]] = []

    def visitor(text, cm, tm, font_dict, font_size):
        m = _DATE_RE.search(text)
        if m:
            size = float(font_size or 10)
            fname = _metric_font(font_dict)
            prefix = text[: m.start()]
            prefix_w = _embedded_text_width(font_dict, prefix, size)
            if prefix_w is None:
                prefix_w = _string_width(prefix, fname, size)
            date_w = _embedded_text_width(font_dict, m.group(0), size)
            if date_w is None:
                date_w = _string_width(m.group(0), fname, size)
            x = tm[4] + prefix_w
            hits.append((x, float(tm[5]), size, m.group(0), fname, date_w))

    page.extract_text(visitor_text=visitor)
    if not hits:
        return None
    box = page.cropbox
    bottom_y = float(box.bottom)
    band = float(box.height) * _BOTTOM_BAND
    in_band = [h for h in hits if h[1] - bottom_y <= band]
    pool = in_band or hits
    return min(pool, key=lambda h: h[1])


def _overlay_position(
    x: float,
    y: float,
    cropbox: tuple[float, float, float, float],
    rotation: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Маппинг координат текста на систему оверлея с учётом /Rotate и cropbox.

    Возвращает ((draw_x, draw_y), (page_width, page_height)) для canvas.
    """
    cx0, cy0, cx1, cy1 = cropbox
    bw, bh = cx1 - cx0, cy1 - cy0
    rot = int(rotation) % 360
    dx, dy = x - cx0, y - cy0
    if rot == 90:
        # страница отображается повёрнутой по часовой: оси меняются местами
        return ((bh - dy), dx), (bh, bw)
    if rot == 180:
        return (bw - dx, bh - dy), (bw, bh)
    if rot == 270:
        return (dy, (bw - dx)), (bh, bw)
    return (dx, dy), (bw, bh)


def _strip_old_date_from_text_layer(
    writer: PdfWriter, page, old_text: str, new_text: str
) -> bool:
    """Удаляет старую дату из извлекаемого text layer заменой той же длины.

    Работает, когда дата записана в content stream одной литеральной строкой;
    иначе визуального перекрытия достаточно, а функция возвращает False.
    """
    try:
        data = page.get_contents().get_data()
        old_bytes = old_text.encode("latin-1", "replace")
        new_bytes = new_text.encode("latin-1", "replace")
        if len(old_bytes) != len(new_bytes) or old_bytes not in data:
            return False
        replacement = DecodedStreamObject()
        replacement.set_data(data.replace(old_bytes, new_bytes))
        page[NameObject("/Contents")] = writer._add_object(replacement)
        return True
    except Exception:
        return False


def update_cv_date(
    src: Path,
    dst: Path,
    date_str: str | None = None,
    city: str | None = None,
) -> None:
    """Копирует CV в dst, обновляя нижний footer-date на актуальный.

    Старая дата перекрывается белым прямоугольником и, по возможности,
    заменяется в самом text layer; новая рисуется тем же шрифтом и размером.
    Если даты в исходнике нет (рекомендуемый формат CV), рисуется строка
    «<город>, <дата>» (или просто дата) в правом нижнем углу.
    """
    date_str = date_str or date.today().strftime("%d.%m.%Y")
    footer = f"{city.strip()}, {date_str}" if city and city.strip() else date_str
    writer = PdfWriter(clone_from=str(src))
    last = writer.pages[-1]
    cropbox = (
        float(last.cropbox.left),
        float(last.cropbox.bottom),
        float(last.cropbox.right),
        float(last.cropbox.top),
    )
    hit = _find_date_pos(last)
    if hit:
        x, y, size, old_text, metric_font, old_date_width = hit
    else:
        # города/даты в исходнике нет: свежая строка ставится в правый
        # нижний угол поля контента — text layer остаётся чистым
        size = 10.0
        old_text, metric_font, old_date_width = "", "Helvetica", 0.0
        x = cropbox[2] - 56.8 - _string_width(footer, metric_font, size)
        y = 40.0
    try:
        rotation = int(last.get("/Rotate", 0) or 0)
    except Exception:
        rotation = 0
    (draw_x, draw_y), (page_w, page_h) = _overlay_position(
        x, y, cropbox, rotation
    )
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))
    new_width = _string_width(footer, metric_font, size)
    # плашка накрывает и старую (точная ширина из PDF), и новую строку
    cover_width = max(old_date_width, new_width)
    c.setFillColorRGB(1, 1, 1)
    c.rect(
        draw_x - 4,
        draw_y - 2,
        cover_width + 8,
        size + 5,
        stroke=0,
        fill=1,
    )
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", size)
    c.drawString(draw_x, draw_y, footer)
    c.save()
    buf.seek(0)
    overlay = PdfReader(buf).pages[0]
    last.merge_page(overlay)
    if old_text and old_text != date_str:
        _strip_old_date_from_text_layer(writer, last, old_text, date_str)
    with open(dst, "wb") as f:
        writer.write(f)


def _resume_city(user: str | None = None) -> str:
    """«Город для резюме» из профиля текущего аккаунта (пусто — не рисовать)."""
    from .store import Store

    store = Store(user=user or current_user())
    try:
        profile = store.candidate_profile(user or current_user()) or {}
    finally:
        store.close()
    return str(profile.get("resume_city") or "").strip()


def prepare_dossier(v: Vacancy, letter_path: Path) -> Path:
    """Собирает досье во временном каталоге и публикует его атомарно.

    Досье всегда живёт под одним именем (`<Компания>_<refnr>`): предыдущая
    версия заменяется, устаревшие копии `_vN` удаляются. При ошибке сборки
    временное содержимое удаляется, прежнее досье остаётся нетронутым.
    """
    letter_text = None
    if letter_path.suffix.lower() in {".md", ".txt"}:
        letter_text = letter_path.read_text(encoding="utf-8")
        if not letter_text.strip():
            raise ValueError("Empty letter: dossier will not be created")
    elif not letter_path.is_file() or letter_path.stat().st_size == 0:
        raise ValueError("Empty letter: dossier will not be created")
    root = applications_dir(current_user(), APPLICATIONS_DIR)
    base = f"{safe_slug(_folder_name(v))}_{safe_slug(v.refnr, 40)}"
    root.mkdir(parents=True, exist_ok=True)
    folder = root / base
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{folder.name}-", dir=str(folder.parent))
    )
    published = False
    replaced: Path | None = None
    try:
        shutil.copy2(letter_path, tmp_dir / f"Anschreiben{letter_path.suffix.lower()}")
        cv = _cv_pdf()
        if not cv.is_file():
            raise FileNotFoundError(f"CV file not found: {cv}")
        update_cv_date(cv, tmp_dir / "Lebenslauf.pdf", city=_resume_city())
        (tmp_dir / "stelle.txt").write_text(
            f"{v.title}\n{v.company}\n{v.location}\n{v.url}\n",
            encoding="utf-8",
        )
        if folder.exists():
            replaced = folder.parent / f".{folder.name}-replaced-{os.getpid()}"
            os.replace(folder, replaced)
        os.replace(tmp_dir, folder)
        published = True
        for legacy in sorted(root.glob(f"{folder.name}_v*")):
            if legacy.is_dir():
                shutil.rmtree(legacy, ignore_errors=True)
        return folder
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if replaced is not None and replaced.exists():
            if published:
                shutil.rmtree(replaced, ignore_errors=True)
            else:
                os.replace(replaced, folder)  # откат на прежнее досье
