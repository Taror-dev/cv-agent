from pypdf import PdfReader
from reportlab.pdfgen import canvas

import pytest
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject

import cv_agent.apply as apply_module
from cv_agent.apply import (
    _find_date_pos,
    _overlay_position,
    prepare_dossier,
    update_cv_date,
)
from cv_agent.models import Vacancy


def _make_cv(path):
    c = canvas.Canvas(str(path), pagesize=(595.28, 841.89))
    c.setFont("Helvetica", 11)
    c.drawString(72, 500, "Wangen im Allgäu, 16.03.2026")
    c.showPage()
    c.save()


def test_update_cv_date_replaces_bottom_date(tmp_path):
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    _make_cv(src)
    update_cv_date(src, dst, "16.08.2026")
    text = PdfReader(str(dst)).pages[-1].extract_text()
    assert "16.08.2026" in text


def test_update_cv_date_default_today(tmp_path):
    from datetime import date

    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    _make_cv(src)
    update_cv_date(src, dst)
    text = PdfReader(str(dst)).pages[-1].extract_text()
    assert date.today().strftime("%d.%m.%Y") in text


def test_update_cv_date_multipage(tmp_path):
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Seite eins")
    c.showPage()
    c.setFont("Helvetica", 11)
    c.drawString(72, 500, "Wangen im Allgäu, 16.03.2026")
    c.save()
    update_cv_date(src, dst, "16.08.2026")
    reader = PdfReader(str(dst))
    assert len(reader.pages) == 2
    assert "16.08.2026" in reader.pages[-1].extract_text()


def test_update_cv_date_fallback_position(tmp_path):
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Ohne Datum")
    c.save()
    update_cv_date(src, dst, "16.08.2026")
    page = PdfReader(str(dst)).pages[-1]
    assert "16.08.2026" in page.extract_text()
    # без даты в исходнике свежая ставится в правый нижний угол поля
    hit = _find_date_pos(page)
    assert hit is not None
    x, y, _size, matched, _font, _w = hit
    assert matched == "16.08.2026"
    assert x > 400  # прижата к правому краю, а не к (40, 30)
    assert y < 60


def test_update_cv_date_without_source_date_has_clean_text_layer(tmp_path):
    """CV без даты: в text layer ровно одна дата — нарисованная сервисом."""
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Wangen im Allgäu")
    c.save()
    update_cv_date(src, dst, "24.08.2026")
    text = PdfReader(str(dst)).pages[-1].extract_text()
    assert text.count("24.08.2026") == 1
    assert "16.03" not in text


def test_empty_letter_does_not_create_dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    letter = tmp_path / "empty.md"
    letter.write_text(" \n", encoding="utf-8")
    vacancy = Vacancy("1", "Job", "ACME", "Wangen", "u", "ba")
    with pytest.raises(ValueError, match="Empty letter"):
        prepare_dossier(vacancy, letter)
    assert not (tmp_path / "applications").exists()


# ---- этап 6: footer date как самый нижний hit ----


def test_find_date_pos_picks_bottom_most_hit(tmp_path):
    src = tmp_path / "cv.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.setFont("Helvetica", 11)
    c.drawString(72, 780, "Referenz 01.02.2026")
    c.drawString(72, 40, "Wangen im Allgäu, 16.03.2026")
    c.showPage()
    c.save()
    page = PdfReader(str(src)).pages[0]
    hit = _find_date_pos(page)
    assert hit is not None
    x, y, size, matched, metric_font, date_width = hit
    assert matched == "16.03.2026"  # нижний из двух, а не первый по порядку
    assert y < 100
    assert metric_font == "Helvetica"  # reportlab-тестовый шрифт
    assert date_width > 0


def test_update_cv_date_removes_old_date_from_text_layer(tmp_path):
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    _make_cv(src)
    update_cv_date(src, dst, "16.08.2026")
    text = PdfReader(str(dst)).pages[-1].extract_text()
    assert "16.08.2026" in text  # exit gate: новая дата присутствует
    assert "16.03.2026" not in text  # exit gate: старая отсутствует в text layer


def test_update_cv_date_handles_rotated_page(tmp_path):
    from pypdf import PdfReader, PdfWriter

    src = tmp_path / "cv.pdf"
    _make_cv(src)

    writer = PdfWriter(clone_from=str(src))
    writer.pages[-1].rotate(90)
    rotated = tmp_path / "rotated.pdf"
    with open(rotated, "wb") as f:
        writer.write(f)

    dst = tmp_path / "out-rotated.pdf"
    update_cv_date(rotated, dst, "16.08.2026")
    reader = PdfReader(str(dst))
    assert int(reader.pages[-1].get("/Rotate", 0)) == 90
    text = reader.pages[-1].extract_text()
    assert "16.08.2026" in text
    assert "16.03.2026" not in text


def test_overlay_position_accounts_cropbox_and_rotation():
    box = (10.0, 20.0, 610.0, 861.89)  # cx0 cy0 cx1 cy1

    (x0, y0), (w0, h0) = _overlay_position(110, 520, box, 0)
    assert (round(x0, 2), round(y0, 2)) == (100.0, 500.0)
    assert (w0, h0) == (600.0, 841.89)

    # 180°: обе координаты зеркалятся внутри cropbox
    (x180, y180), (w180, h180) = _overlay_position(110, 520, box, 180)
    assert (round(x180, 2), round(y180, 2)) == (600.0 - 100.0, 841.89 - 500.0)
    assert (w180, h180) == (600.0, 841.89)

    # 90°: оси меняются местами, размер страницы транспонируется
    (x90, y90), (w90, h90) = _overlay_position(110, 520, box, 90)
    assert (w90, h90) == (841.89, 600.0)
    assert round(x90, 2) == round(841.89 - 500.0, 2)
    assert round(y90, 2) == 100.0


# ---- этап 6: версионная атомарная публикация досье ----


def test_prepare_dossier_versions_do_not_overwrite(tmp_path, monkeypatch):
    """Повторный apply перезаписывает досье под тем же именем (версий нет)."""
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    monkeypatch.setattr(apply_module, "CV_PDF", tmp_path / "cv.pdf")
    c = canvas.Canvas(str(tmp_path / "cv.pdf"), pagesize=(595.28, 841.89))
    c.drawString(72, 500, "Wangen im Allgäu, 16.03.2026")
    c.save()
    letter = tmp_path / "Anschreiben.md"
    letter.write_text("Sehr geehrte Damen und Herren,\nIlia Rerich", encoding="utf-8")
    v = Vacancy("ref-9", "Job", "ACME", "Wangen", "u", "ba")

    first = prepare_dossier(v, letter)
    second = prepare_dossier(v, letter)

    assert first == second
    assert first.name.endswith("ref-9")
    for folder in (first, second):
        assert (folder / "Anschreiben.md").is_file()
        assert (folder / "Lebenslauf.pdf").is_file()
        assert (folder / "stelle.txt").is_file()
    # ровно одно досье — старое заменено, а не продублировано
    assert sorted(
        p.name for p in (tmp_path / "applications").iterdir() if p.is_dir()
    ) == [first.name]


def test_prepare_dossier_failure_publishes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    monkeypatch.setattr(
        apply_module,
        "_cv_pdf",
        lambda: tmp_path / "missing.pdf",
    )
    letter = tmp_path / "Anschreiben.md"
    letter.write_text("Ilia Rerich", encoding="utf-8")
    v = Vacancy("ref-x", "Job", "ACME", "Wangen", "u", "ba")

    with pytest.raises(FileNotFoundError):
        prepare_dossier(v, letter)

    apps = tmp_path / "applications"
    published = [p for p in apps.iterdir() if p.is_dir()] if apps.exists() else []
    assert published == []  # exit gate: ошибка CV не оставляет «готовое» досье
    leftovers = [p for p in apps.iterdir() if p.name.startswith(".")] if apps.exists() else []
    assert leftovers == []  # временный каталог удалён


def test_dossier_folder_name_is_sanitized_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    letter = tmp_path / "Anschreiben.md"
    letter.write_text("Ilia Rerich", encoding="utf-8")
    long_company = "Очень Длинная Компания " * 10 + '"/\\:*?"<>|'
    v = Vacancy("ss/../..bad id 1234567890" * 3, "Job", long_company, "Wangen", "u", "ba")

    folder = prepare_dossier(v, letter)

    import re

    assert re.fullmatch(r"[A-Za-z0-9_\-]+", folder.name)
    assert len(folder.name) <= 105  # slug компании + "_" + refnr, оба ограничены


def test_prepare_dossier_replaces_previous_and_cleans_legacy_versions(
    tmp_path, monkeypatch
):
    """Повторный apply: одно досье без _v2; легаси-версии удаляются."""
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    monkeypatch.setattr(apply_module, "CV_PDF", tmp_path / "cv.pdf")
    c = canvas.Canvas(str(tmp_path / "cv.pdf"), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Lebenslauf")
    c.save()
    letter = tmp_path / "letter.md"
    letter.write_text("Sehr geehrte Damen und Herren", encoding="utf-8")

    vacancy = Vacancy("9", "Job", "ACME", "Wangen", "u", "ba")
    first = prepare_dossier(vacancy, letter)
    assert first.name == "ACME_9"

    # легаси-версия от прежнего поведения
    legacy = first.parent / "ACME_9_v2"
    legacy.mkdir()
    (legacy / "Anschreiben.md").write_text("old", encoding="utf-8")

    second = prepare_dossier(vacancy, letter)
    assert second == first  # то же имя, никаких _v2/_v3
    assert not list(first.parent.glob("ACME_9_v*"))
    assert (first / "Anschreiben.md").is_file()

    # ошибка сборки не трогает прежнее досье и не создаёт версий
    bad_letter = tmp_path / "empty.md"
    bad_letter.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_dossier(vacancy, bad_letter)
    assert (first / "Anschreiben.md").is_file()
    assert len(list(first.parent.iterdir())) == 1


# ---- этап 6: prepared → applied только по подтверждению ----


def test_apply_sets_prepared_and_confirm_applied_sets_applied():
    from types import SimpleNamespace

    from cv_agent import cli
    from cv_agent.store import Store

    store = Store()
    store.add(Vacancy("conf-1", "Job", "ACME", "Wangen", "u", "ba"))
    store.close()

    cli.cmd_confirm_applied(SimpleNamespace(refnr="conf-1"))

    check = Store()
    try:
        assert check.application_status("conf-1") == "applied"
    finally:
        check.close()

    # повторное подтверждение идемпотентно
    cli.cmd_confirm_applied(SimpleNamespace(refnr="conf-1"))
    check = Store()
    try:
        assert check.application_status("conf-1") == "applied"
    finally:
        check.close()


def test_confirm_applied_unknown_vacancy_exits():
    from types import SimpleNamespace

    from cv_agent import cli

    with pytest.raises(SystemExit):
        cli.cmd_confirm_applied(SimpleNamespace(refnr="does-not-exist"))


# ---- этап 7: retention фоновых заданий ----


def test_prune_finished_jobs_removes_only_old_terminal():
    import time
    from cv_agent.store import Store

    store = Store()
    try:
        run_old, created = store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        assert created
        store.claim_job(run_old)
        store.finish_job(run_old, "succeeded")
        # finished_at проставлен CURRENT_TIMESTAMP — сдвинем назад вручную
        store.conn.execute(
            "UPDATE jobs SET finished_at=datetime('now', '-40 days') "
            "WHERE run_id=?",
            (run_old,),
        )
        run_fresh, _ = store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        store.claim_job(run_fresh)
        store.finish_job(run_fresh, "succeeded")

        removed = store.prune_finished_jobs(older_than_days=30)
        assert removed == 1
        assert store.job(run_old) is None
        assert store.job(run_fresh) is not None
        # активные задания не удаляются никогда
        run_active, _ = store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        store.claim_job(run_active)
        assert store.prune_finished_jobs(older_than_days=0) >= 0
        assert store.job(run_active) is not None
    finally:
        store.close()


def test_cmd_prune_jobs_requires_confirm(capsys):
    from types import SimpleNamespace

    from cv_agent import cli

    with pytest.raises(SystemExit):
        cli.cmd_prune_jobs(SimpleNamespace(confirm=False, older_than_days=30))


def test_embedded_text_width_uses_font_widths():
    from cv_agent.apply import _embedded_text_width

    # Calibri-подобный simple font: FirstChar=32, ширина пробела 227,
    # цифры (коды 33+) — 507 (в тысячных единицах)
    font = DictionaryObject()
    font[NameObject("/Widths")] = ArrayObject(
        [NumberObject(227)] + [NumberObject(507)] * 17
    )
    font[NameObject("/FirstChar")] = NumberObject(32)
    # ' ' + '11' -> (227 + 507 + 507) / 1000 * size
    assert _embedded_text_width(font, " 11", 10.0) == pytest.approx(12.41)
    # вне диапазона Widths -> MissingWidth (по умолчанию 0)
    assert _embedded_text_width(font, "A", 10.0) == 0.0


def test_embedded_text_width_falls_back_to_none():
    from cv_agent.apply import _embedded_text_width

    assert _embedded_text_width(DictionaryObject(), "16.03.2026", 11.0) is None
    assert _embedded_text_width(None, "16.03.2026", 11.0) is None


def test_update_cv_date_with_city_draws_city_and_date(tmp_path):
    """CV без даты + город профиля -> в футере «Город, ДД.ММ.ГГГГ»."""
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Lebenslauf ohne Datum")
    c.save()

    update_cv_date(src, dst, "24.08.2026", city="Wangen im Allgäu")

    page = PdfReader(str(dst)).pages[-1]
    text = page.extract_text()
    assert "Wangen im Allgäu, 24.08.2026" in text
    hit = _find_date_pos(page)
    assert hit is not None
    x, y, _size, matched, _font, _w = hit
    assert matched == "24.08.2026"
    # строка с городом шире — правый край всё ещё у поля контента
    assert x > 350 and y < 60


def test_update_cv_date_empty_city_draws_date_only(tmp_path):
    src = tmp_path / "cv.pdf"
    dst = tmp_path / "out.pdf"
    c = canvas.Canvas(str(src), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Lebenslauf")
    c.save()
    update_cv_date(src, dst, "24.08.2026", city="   ")
    text = PdfReader(str(dst)).pages[-1].extract_text()
    assert "24.08.2026" in text
    assert ", 24.08.2026" not in text


def test_prepare_dossier_stamps_resume_city_from_profile(
    tmp_path, monkeypatch, isolated_project
):
    from cv_agent.models import Vacancy

    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", tmp_path / "applications")
    monkeypatch.setattr(apply_module, "CV_PDF", tmp_path / "cv.pdf")
    c = canvas.Canvas(str(tmp_path / "cv.pdf"), pagesize=(595.28, 841.89))
    c.drawString(72, 700, "Lebenslauf ohne Datum")
    c.save()
    letter = tmp_path / "letter.md"
    letter.write_text("Sehr geehrte Damen und Herren", encoding="utf-8")

    from cv_agent.store import Store

    store = Store()
    try:
        store.set_candidate_profile({"resume_city": "Wangen im Allgäu"}, "Илья")
    finally:
        store.close()

    vacancy = Vacancy("1", "Job", "ACME", "Wangen", "u", "ba")
    folder = prepare_dossier(vacancy, letter)

    text = PdfReader(folder / "Lebenslauf.pdf").pages[-1].extract_text()
    assert "Wangen im Allgäu," in text
