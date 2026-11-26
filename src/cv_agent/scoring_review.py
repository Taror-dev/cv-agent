"""Export a private, blinded review packet from one account, without DB writes.

Uses cached descriptions only. Does not call sources, embeddings or LLMs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config
from .documents import DOCUMENT_TYPE_PRIORITY, DOCUMENT_TYPES, SUPPORTED_DOCUMENT_EXTENSIONS, extract_document_result, infer_document_type
from .paths import account_directory, account_path
from .scoring_config import parse_scoring_config
from .scoring_evaluation import prepare_review, write_private
from .scoring_benchmark import dataset_hash
from .scoring_location import parse_location_preferences
from .storage import storage_layout_active


def _normal(text):
    return " ".join((text or "").casefold().split())


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_packet(*, count=100, user=None, limits=None):
    """Read a SQLite snapshot through mode=ro; never instantiate Store."""
    if os.environ.get("CV_AGENT_RUN_ID"):
        raise ValueError("Export outside production jobs (unset CV_AGENT_RUN_ID)")
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Review count must be between 1 and 100")
    if limits is None:
        criteria = yaml.safe_load((config.ROOT / "criteria.yaml").read_text(encoding="utf-8"))
        if not isinstance(criteria, dict):
            raise ValueError("Expected criteria mapping")
        limits = parse_scoring_config(criteria.get("scoring"))
    user = user or config.current_user()
    uri = (config.DATA_DIR / "applications.db").resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only = ON")
        db.execute("BEGIN")
        account = db.execute("SELECT account_id FROM accounts WHERE name=?", (user,)).fetchone()
        if not account:
            raise ValueError("Current account does not exist")
        account_id = account["account_id"]
        columns = {row["name"] for row in db.execute("PRAGMA table_info(candidate_profiles)")}
        preference_column = "scoring_preferences" if "scoring_preferences" in columns else "'{}' AS scoring_preferences"
        profile_row = db.execute(f"SELECT languages,{preference_column} FROM candidate_profiles WHERE account_id=?", (account_id,)).fetchone()
        profile = dict(profile_row) if profile_row else {}
        types = {row["filename"]: row["document_type"] for row in db.execute(
            "SELECT filename, document_type FROM documents WHERE account_id=?", (account_id,))}
        rows = [dict(row) for row in db.execute(
            "SELECT refnr,title,company,location,url,source,description FROM vacancies "
            "WHERE account_id=? AND archived_at IS NULL AND length(description)>=80", (account_id,))]
    finally:
        db.close()

    # No ensure_account_path / documents_dir: those create directories or Store.
    folder = (account_path(config.DATA_DIR, account_id, "cv") if storage_layout_active()
              else account_directory(config.ROOT / "cv", user))
    candidates = []
    if folder.is_dir():
        for path in folder.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                continue
            kind = types.get(path.name, infer_document_type(path.name))
            if kind not in DOCUMENT_TYPES:
                raise ValueError("Invalid stored document type")
            candidates.append((DOCUMENT_TYPE_PRIORITY[kind], path.name, path, kind))
    documents, provenance, warnings = [], [], []
    remaining = limits.total_chars
    for _, _, path, kind in sorted(candidates):
        if remaining <= 0:
            warnings.append("Document budget exhausted; remaining files excluded")
            break
        item = extract_document_result(path, min(remaining, limits.per_file), preserve_structure=True)
        document_id = f"document-{len(provenance) + 1:03d}"
        provenance.append({"id": document_id, "filename": path.name, "kind": kind,
                           "included_chars": len(item.text), "included_text_hash": _hash(item.text),
                           "truncated": item.truncated, "error": item.error})
        if item.error:
            warnings.append(f"{document_id}: {item.error}")
        if item.truncated:
            warnings.append(f"{document_id}: text truncated by processing budget")
        if item.text:
            documents.append({"id": document_id, "kind": kind, "text": item.text})
            remaining -= len(item.text)
    languages = json.loads(profile.get("languages") or "[]")
    if not isinstance(languages, list) or any(not isinstance(item, str) for item in languages):
        raise ValueError("Invalid profile languages")
    language_text = "\n".join(languages)
    if language_text[:remaining]:
        documents.append({"id": "structured-profile", "kind": "profile", "text": language_text[:remaining]})
    if len(language_text) > remaining:
        warnings.append("Structured profile languages truncated")
    if not documents:
        raise ValueError("Current profile has no extractable scoring documents")
    preferences = parse_location_preferences(json.loads(profile.get("scoring_preferences") or "{}"))

    # Exact text duplicates removed; title/company families share one split.
    # Hash ordering and source round-robin are independent of saved scores.
    buckets, seen = defaultdict(list), set()
    for row in sorted(rows, key=lambda item: item["refnr"]):
        text_hash = _hash(_normal(row["description"]))
        if text_hash in seen or not _normal(row["title"]):
            continue
        seen.add(text_hash)
        row["text_hash"] = text_hash
        row["group"] = _hash(_normal(row["title"]) + "\n" + _normal(row["company"]))
        buckets[row["source"] or "unknown"].append(row)
    available = sum(len(items) for items in buckets.values())
    if available < count:
        raise ValueError(f"Only {available} unique cached vacancies available; requested {count}")
    for items in buckets.values():
        items.sort(key=lambda item: item["text_hash"])
    selected = []
    while len(selected) < count:
        for source in sorted(buckets):
            if buckets[source] and len(selected) < count:
                selected.append(buckets[source].pop(0))
    groups = sorted({row["group"] for row in selected})
    if len(groups) < 2:
        raise ValueError("Need at least two independent vacancy groups for splitting")
    holdout = set(groups[:max(1, round(len(groups) * .2))])
    vacancies, sources = {}, {}
    for index, row in enumerate(selected, 1):
        vacancy_id = f"vacancy-{index:03d}"
        vacancies[vacancy_id] = {"title": row["title"], "location": row["location"] or "",
                                 "text": row["description"], "group": row["group"],
                                 "split": "holdout" if row["group"] in holdout else "calibration"}
        sources[vacancy_id] = {key: row[key] for key in ("refnr", "source", "url", "company", "text_hash")}
    data = prepare_review({"version": 1, "label_source": "pending_human_review",
                           "profiles": {"candidate-1": {"documents": documents, "scoring_preferences": preferences}},
                           "vacancies": vacancies, "judgments": {"candidate-1": {key: None for key in vacancies}}})
    data["review_instructions"] = "Grade cached real vacancies: 2 suitable, 1 borderline, 0 unsuitable; record reasons and mandatory conditions. Replace label_source only after independent manual review."
    manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "account_id": account_id, "documents": provenance, "sources": sources, "warnings": warnings,
                "document_limits": {"per_file": limits.per_file, "total_chars": limits.total_chars},
                "selection": {"count": count, "available_unique": available,
                              "method": "source round-robin; SHA256 of normalized cached descriptions; saved scores never read",
                              "split": "20% of title/company groups by hash; exact text duplicates removed"},
                "limitations": ["Private source text may contain personal data; this is not an anonymized public dataset.",
                                 "One current profile; source-balanced sample is not a random estimate of all search results.",
                                 "Near duplicates with different titles/companies require manual group review before evaluation.",
                                 "No human grades, current scores, source refresh or quality claims."]}
    return data, manifest


REVIEW_GUIDE = """# Ручная проверка вакансий

Это закрытый пакет текущего профиля. Документы и объявления могут содержать
персональные данные; не публикуйте их без отдельного обезличивания.
Исходная база, документы и сохранённые баллы не изменены. Сеть не используется.

В dataset.yaml для каждой пары candidate-1 / vacancy-NNN заполните judgments:
0 — не подходит; 1 — пограничная; 2 — подходит для отклика.
В review_notes укажите причину, обязательные требования (mandatory_checks)
и подтверждающие или отсутствующие сведения в документах (evidence_checks).
Отсутствие информации не означает отсутствие навыка. Учебный материал не
подтверждает коммерческий опыт. Старые оценки ИИ в пакет не включены.

Сначала проверьте группы дубликатов: версии одной вакансии должны находиться
в одной части. Близкие тексты с разными названиями могут требовать объединения.
Оставьте holdout для окончательной проверки: его результаты нельзя использовать
для подбора порогов. После завершения всех оценок замените label_source на
human_reviewed_v1. До этого benchmark откажется считать качество.

sources.json содержит происхождение и ссылки на объявления, без старых баллов.
Выборка сбалансирована по источникам, включает один профиль и ещё не является
полным эталонным набором из нескольких профилей.
"""


def build_feedback_packet(*, user=None, limit=100):
    """Export saved labeled snapshots, never relabel changed current inputs."""
    if os.environ.get("CV_AGENT_RUN_ID"):
        raise ValueError("Export outside production jobs (unset CV_AGENT_RUN_ID)")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Feedback export limit must be between 1 and 100")
    uri = (config.DATA_DIR / "applications.db").resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    db.create_function("feedback_order", 1, _hash, deterministic=True)
    try:
        db.execute("PRAGMA query_only = ON")
        db.execute("BEGIN")
        account = db.execute("SELECT account_id FROM accounts WHERE name=?", (user or config.current_user(),)).fetchone()
        if not account:
            raise ValueError("Current account does not exist")
        rows = [dict(row) for row in db.execute(
            "SELECT refnr,grade,reason,candidate_hash,vacancy_hash,inputs_json,updated_at "
            "FROM vacancy_reviews WHERE account_id=? ORDER BY feedback_order(refnr),refnr LIMIT ?", (account["account_id"], limit))]
    finally:
        db.close()
    selected, skipped = [], []
    for row in rows:
        inputs = json.loads(row.pop("inputs_json"))
        if not isinstance(inputs, dict) or inputs.get("version") != 1 or not isinstance(inputs.get("profile"), dict) or not isinstance(inputs.get("vacancy"), dict):
            raise ValueError("Invalid saved feedback snapshot")
        vacancy = inputs["vacancy"]
        if not isinstance(vacancy.get("text"), str) or not vacancy["text"].strip():
            skipped.append(row["refnr"])
            continue
        row["inputs"] = inputs
        row["family"] = _hash(_normal(vacancy["title"]) + "\n" + _normal(vacancy["company"]))
        row["text_hash"] = _hash(_normal(vacancy["text"]))
        selected.append(row)
    # Sampling uses only stable source identity, never suitability grade.
    selected = sorted(selected, key=lambda row: _hash(row["refnr"]))[:limit]
    if not selected:
        raise ValueError("No saved feedback with vacancy text in the current account")
    # Union both same title/company families and exact duplicate descriptions.
    parent = {row["family"]: row["family"] for row in selected}
    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key
    texts = {}
    for row in selected:
        key = row["family"]
        if row["text_hash"] in texts:
            a, b = root(key), root(texts[row["text_hash"]])
            parent[max(a, b)] = min(a, b)
        texts[row["text_hash"]] = key
    groups = sorted({root(row["family"]) for row in selected})
    if len(groups) < 2:
        raise ValueError("Need feedback on at least two independent vacancy groups")
    holdout = set(groups[:max(1, round(len(groups) * .2))])
    data = {"version": 1, "label_source": "human_card_feedback_v1", "profiles": {}, "vacancies": {}, "judgments": {}, "review_notes": {}}
    manifest = {"version": 1, "account_id": account["account_id"], "sources": {}, "skipped_empty_descriptions": skipped,
                "limitations": ["Informal nonblinded user feedback: automatic scores may have been visible.",
                                 "Uses documents and vacancy text saved at feedback time, not current refreshed inputs.",
                                 "Private personal data; not an anonymized public benchmark.",
                                 "Related versions with different titles/company/text need manual duplicate review."]}
    profile_ids = {}
    for index, row in enumerate(selected, 1):
        profile_id = profile_ids.setdefault(row["candidate_hash"], f"candidate-{len(profile_ids) + 1:03d}")
        data["profiles"][profile_id] = row["inputs"]["profile"]
        vacancy_id = f"vacancy-{index:03d}"
        job = row["inputs"]["vacancy"]
        group = root(row["family"])
        data["vacancies"][vacancy_id] = {"title": job["title"], "location": job["location"], "text": job["text"],
                                         "group": group, "split": "holdout" if group in holdout else "calibration"}
        data["judgments"].setdefault(profile_id, {})[vacancy_id] = row["grade"]
        data["review_notes"].setdefault(profile_id, {})[vacancy_id] = {"reason": row["reason"], "mandatory_checks": [], "evidence_checks": []}
        manifest["sources"][vacancy_id] = {"refnr": row["refnr"], "source": job["source"], "url": job["url"],
                                           "company": job["company"], "updated_at": row["updated_at"], "candidate_hash": row["candidate_hash"],
                                           "vacancy_hash": row["vacancy_hash"], "warnings": row["inputs"].get("warnings", [])}
    manifest["dataset_hash"] = dataset_hash(data)
    return data, manifest


FEEDBACK_GUIDE = """# Отзывы из карточек вакансий

Оценки и пояснения уже заполнены вашими ответами в интерфейсе.
Для каждого ответа сохранены документы и текст вакансии на момент оценки.
Изменившиеся позже документы или объявления не подменяют эти исходные данные.
Разные версии документов представлены отдельными профилями candidate-NNN.

Это локальный пакет с персональными данными. Автоматический балл мог быть
виден при оценке, поэтому отзывы не являются независимой слепой проверкой.
Близкие дубликаты с разными названиями/компаниями требуют отдельной проверки.
Для подбора порогов в calibration нужны и подходящие, и неподходящие примеры;
маленькая выборка помогает найти ошибки, но не доказывает качество всей выдачи.

С запущенной локальной Ollama и установленными моделями:
PYTHONPATH=src .venv/bin/python -m cv_agent.scoring_evaluation \\
  --dataset PATH_TO_THIS_FOLDER/dataset.yaml \\
  --models docs/benchmarks/local_embedding_models.yaml \\
  --k 3 --output /tmp/new-feedback-evaluation.json

Проверка никогда не меняет рабочие настройки или текущие баллы.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=range(50, 101), default=100, metavar="50..100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-feedback", action="store_true", help="Export existing card feedback; up to --count labeled snapshots")
    args = parser.parse_args(argv)
    try:
        data, manifest = build_feedback_packet(limit=args.count) if args.from_feedback else build_packet(count=args.count)
        target = args.output_dir.absolute()
        if any(parent.is_symlink() for parent in target.parents):
            raise ValueError("Output parent must not be a symlink")
        # Existing directory/symlink is never reused. Parent must already exist.
        target.mkdir(mode=0o700)
        write_private(target / "dataset.yaml", yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        write_private(target / "sources.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        write_private(target / "README.md", FEEDBACK_GUIDE if args.from_feedback else REVIEW_GUIDE)
        print(f"Private review packet: {target} ({len(data['vacancies'])} vacancies)")
    except (ValueError, TypeError, OSError, sqlite3.Error, yaml.YAMLError) as exc:
        parser.exit(2, f"Review export error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
