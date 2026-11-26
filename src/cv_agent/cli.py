import argparse
import json
import os
import sys

from .apply import prepare_dossier
from .config import (
    DEFAULT_USER,
    ROOT,
    current_user,
    is_recruiting_agency,
    is_ollama_url,
    load_criteria,
    load_profile,
    pick_llm,
)
from .documents import load_candidate_text
from .llm import LLMClient, LLMError, LLMRateLimiter
from .llm_contracts import (
    KEYWORDS_PROMPT_VERSION,
    KEYWORDS_SCHEMA,
    KEYWORDS_SYSTEM,
    LETTER_PROMPT_VERSION,
    LLMContractError,
    SCORING_PROMPT_VERSION,
    complete_json_contract,
    keywords_prompt,
    parse_json_object,
    text_hash,
    vacancy_input,
    validate_keywords,
)
from .score import (
    prefilter_reason,
    prefilter_reason_scopes,
    score_one,
    scoreability_reason,
)
from .search import (
    BASearch,
    DEFAULT_MAX_KEYWORDS,
    DEFAULT_PER_QUERY_LIMIT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOTAL_LIMIT,
    GetInItSearch,
    LinkedInSearch,
    StepStoneSearch,
    VacancyUnavailable,
    dedup,
    ensure_description,
    fetch_description,
    normalize_keywords,
    run_adapter,
)
from .store import Store
from .security import redact_sensitive
from .storage import apply_storage_migration, plan_storage_migration
from .write import letter, save_letter, validate_letter_identity


def _llm_client_for(criteria) -> LLMClient:
    """LLM текущего профиля без записи account key в process environment."""
    cfg = pick_llm(criteria)
    api_key = None
    if cfg.api_key_env == "GEMINI_API_KEY":
        store = Store()
        try:
            api_key = store.account_gemini_key(current_user())
        finally:
            store.close()
        # Пустая строка запрещает LLMClient брать общий ключ из окружения.
        api_key = api_key or ""
    elif cfg.api_key_env == "GROQ_API_KEY":
        store = Store()
        try:
            api_key = store.account_groq_key(current_user())
        finally:
            store.close()
        api_key = api_key or ""
    client = LLMClient(cfg, api_key=api_key)
    if not is_ollama_url(cfg.base_url):
        raw_run_id = os.environ.get("CV_AGENT_RUN_ID", "")
        run_id = int(raw_run_id) if raw_run_id.isdigit() else None
        try:
            client.rate_limiter = LLMRateLimiter(cfg, current_user(), run_id)
        except (AttributeError, TypeError):
            pass  # schlanke Test-Doubles ohne beschreibbare Attribute
    return client


def cmd_search(args):
    c = load_criteria()
    max_keywords = getattr(args, "max_keywords", None)
    if max_keywords is None:
        max_keywords = getattr(c, "search_max_keywords", DEFAULT_MAX_KEYWORDS)
    per_query_limit = getattr(args, "per_query_limit", None)
    if per_query_limit is None:
        per_query_limit = getattr(
            c, "search_per_query_limit", DEFAULT_PER_QUERY_LIMIT
        )
    total_limit = getattr(args, "total_limit", None)
    if total_limit is None:
        total_limit = getattr(args, "limit", None)
    if total_limit is None:
        total_limit = getattr(c, "search_total_limit", DEFAULT_TOTAL_LIMIT)
    for label, value, maximum in (
        ("max_keywords", max_keywords, 50),
        ("per_query_limit", per_query_limit, 100),
        ("total_limit", total_limit, 500),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{label} must be between 1 and {maximum}")
    keyword_selection = normalize_keywords(c.keywords, max_keywords)
    if not keyword_selection.keywords:
        sys.exit("No German keywords in the current profile")

    location_override = getattr(args, "location", None)
    region_value = c.region if location_override is None else str(location_override).strip()
    locations: list[str] = []
    for part in region_value.split(","):
        loc = " ".join(part.split())
        if loc and loc not in locations:
            locations.append(loc)
    if not locations:
        locations = [""]
    radius_override = getattr(args, "radius", None)
    if radius_override is None:
        radius_km = int(c.radius_km)
    else:
        if isinstance(radius_override, bool) or not isinstance(radius_override, int) or not 1 <= radius_override <= 500:
            raise ValueError("radius must be between 1 and 500")
        radius_km = int(radius_override)

    ba = BASearch()
    configured_categories = getattr(c, "getinit_categories", []) or None
    getinit = (
        GetInItSearch(categories=configured_categories)
        if configured_categories
        else GetInItSearch()
    )
    adapters = [
        ba,
        StepStoneSearch(),
        LinkedInSearch(),
        getinit,
    ]
    planned_queries = sum(
        (
            # geo-filter источники не зависят от локации: один набор запросов
            len(adapter.queries(keyword_selection.keywords, locations[0], radius_km, per_query_limit))
            if getattr(adapter, "local_geo_filter", False)
            else len(adapter.queries(keyword_selection.keywords, "", radius_km, per_query_limit))
            * len(locations)
        )
        for adapter in adapters
    )
    store = Store()
    stored_ba_loader = getattr(store, "ba_refnrs", None)
    stored_ba_checks = (
        len(stored_ba_loader()) if callable(stored_ba_loader) else 0
    )
    print(
        "Search plan: "
        f"keywords={len(keyword_selection.keywords)}/{keyword_selection.input_count} "
        "(German/account profile; English not queried separately), "
        f"max_queries={planned_queries}, max_stored_checks={stored_ba_checks}, "
        f"per_query_limit={per_query_limit}, total_limit={total_limit}, "
        f"location={', '.join(locations) or '*'}, radius={radius_km}km"
    )
    if keyword_selection.duplicate_count or keyword_selection.omitted_count:
        print(
            "Keywords cleaned: "
            f"duplicates={keyword_selection.duplicate_count}, "
            f"omitted due to limit={keyword_selection.omitted_count}"
        )
    vacs: list = []
    seen: set[str] = set()
    total_skipped = 0
    total_errors = 0
    for adapter in adapters:
        result = run_adapter(
            adapter,
            keyword_selection.keywords,
            locations,
            radius_km,
            per_query_limit,
            seen,
            total_limit=max(0, total_limit - len(vacs)),
        )
        vacs.extend(result.vacancies)
        total_skipped += result.skipped
        total_errors += len(result.errors)
        print(result.summary())
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
    before_dedup = len(vacs)
    existing_loader = getattr(store, "vacancies_for_dedup", None)
    existing = existing_loader() if callable(existing_loader) else []
    vacs = dedup(
        vacs,
        similarity=getattr(
            c, "search_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
        ),
        existing=existing,
    )
    total_skipped += before_dedup - len(vacs)
    # Явно неоцениваемые вакансии не сохраняются (исключения по названию/
    # компании, рекрутинговые агентства); описание на этом этапе ещё не
    # загружено, поэтому правила для description применит cmd_score.
    kept = []
    dropped_by_filter = 0
    for v in vacs:
        reason = prefilter_reason_scopes(v, c, ("title", "company"))
        if not reason and is_recruiting_agency(v.company, c):
            reason = "recruiting_agency"
        if reason:
            dropped_by_filter += 1
            print(f"Not saved {v.refnr}: {reason}", file=sys.stderr)
            continue
        kept.append(v)
    vacs = kept
    for v in vacs:
        store.add(v)
    archived = _prune_stale(store, ba, {v.refnr for v in vacs})
    store.close()
    print(f"Found and stored: {len(vacs)}")
    if dropped_by_filter:
        print(f"Not saved (filters): {dropped_by_filter}")
    print(f"Skipped: {total_skipped}; query errors: {total_errors}")
    print(f"Freshness checked, archived: {archived}")


def _prune_stale(store: Store, ba: "BASearch", fresh_refnrs: set[str]) -> int:
    """Archiviert nicht verfügbare BA-Stellen; aktive Einträge werden gesehen."""
    archived = 0
    for refnr in store.ba_refnrs():
        if refnr in fresh_refnrs:
            continue
        try:
            if ba.exists(refnr):
                marker = getattr(store, "mark_seen", None)
                if callable(marker):
                    marker(refnr)
            else:
                store.mark_unavailable(refnr)
                archived += 1
                print(f"Archived (no longer available): {refnr}", file=sys.stderr)
        except Exception:
            marker = getattr(store, "mark_source_error", None)
            if callable(marker):
                marker(refnr)
    return archived


def _description_for(store, vacancy, criteria) -> str:
    return ensure_description(
        store,
        vacancy,
        ttl_hours=getattr(criteria, "description_ttl_hours", 168),
        fetcher=fetch_description,
    )


def _mark_detail_error(store, refnr: str) -> None:
    marker = getattr(store, "mark_source_error", None)
    if callable(marker):
        marker(refnr)


ACTUALIZE_ADAPTERS = {
    "arbeitsagentur": BASearch,
    "stepstone": StepStoneSearch,
    "linkedin": LinkedInSearch,
    "get-in-it": GetInItSearch,
}


def cmd_actualize(_args):
    """Prueft alle aktiven Stellen des Profils beim Quellbestand.

    Eindeutig entfernte Stellen werden archiviert (unavailable); Quellfehler
    bleiben wiederholbar und aendern den Status nur auf 'error'.
    """
    store = Store()
    try:
        refnrs = [refnr for refnr, *_ in store.list_all()]
        total = len(refnrs)
        active = archived = errors = skipped = 0
        for index, refnr in enumerate(refnrs, start=1):
            vacancy = store.by_refnr(refnr)
            if vacancy is None:
                continue
            adapter_factory = ACTUALIZE_ADAPTERS.get(vacancy.source or "")
            if adapter_factory is None:
                skipped += 1
                print(
                    f"[{index}/{total}] {refnr}: unsupported source, skipped",
                    file=sys.stderr,
                )
                continue
            try:
                alive = adapter_factory().exists(vacancy)
            except Exception as exc:
                errors += 1
                _mark_detail_error(store, refnr)
                print(
                    f"[{index}/{total}] {refnr}: source error ({exc})",
                    file=sys.stderr,
                )
                continue
            if alive:
                active += 1
                marker = getattr(store, "mark_seen", None)
                if callable(marker):
                    marker(refnr)
                print(f"[{index}/{total}] {refnr}: still available")
            else:
                archived += 1
                store.mark_unavailable(refnr)
                print(
                    f"[{index}/{total}] {refnr}: no longer available, archived",
                    file=sys.stderr,
                )
        print(
            f"Actualize summary: total {total}, active {active}, "
            f"archived {archived}, errors {errors}, skipped {skipped}"
        )
    finally:
        store.close()


def _candidate_text(purpose: str = "scoring") -> str:
    """Структурированный профиль + документы; global YAML только для legacy Ильи."""
    user = current_user()
    store = Store(user=user)
    try:
        profile = store.candidate_profile(user) or {"profile_name": user}
    finally:
        store.close()
    documents = load_candidate_text(user=user, purpose=purpose)
    parts = [
        "Strukturiertes Profil:\n"
        + json.dumps(profile, ensure_ascii=False, indent=2)
    ]
    if documents:
        parts.append(documents)
    elif user == DEFAULT_USER:
        parts.append(
            "Legacy-Profil (nur Standardkonto):\n"
            + json.dumps(load_profile(), ensure_ascii=False)
        )
    return "\n\n".join(parts)


def _candidate_letter_settings() -> tuple[str, str]:
    user = current_user()
    store = Store(user=user)
    try:
        profile = store.candidate_profile(user) or {}
    finally:
        store.close()
    return (
        profile.get("full_name") or user,
        profile.get("preferred_letter_language") or "de",
    )


def _record_llm_artifact(
    store,
    llm,
    kind: str,
    status: str,
    *,
    prompt_version: str,
    candidate_input: str,
    vacancy=None,
    output_path: str = "",
    error: str = "",
    details: dict | None = None,
) -> None:
    """Schreibt Audit-Metadaten; test doubles ohne Artifact-API bleiben kompatibel."""
    recorder = getattr(store, "record_artifact", None)
    if not callable(recorder):
        return
    cfg = getattr(llm, "cfg", None)
    recorder(
        kind,
        status,
        refnr=vacancy.refnr if vacancy is not None else None,
        provider=str(getattr(cfg, "provider", "")),
        model=str(getattr(cfg, "model", "")),
        prompt_version=prompt_version,
        candidate_hash=text_hash(candidate_input),
        vacancy_hash=text_hash(vacancy_input(vacancy)) if vacancy is not None else "",
        output_path=output_path,
        error=error,
        details=details,
    )


def cmd_score(args):
    c = load_criteria()
    engine = getattr(args, "engine", None) or getattr(getattr(c, "scoring", None), "engine", "llm")
    if engine == "algorithmic":
        return _cmd_algorithmic_score(args, c)
    if getattr(args, "preview", False) or getattr(args, "report", None):
        sys.exit("--preview and --report require --engine algorithmic")
    candidate = _candidate_text()
    store = Store()
    llm = _llm_client_for(c)
    llm_cfg = getattr(llm, "cfg", None)
    provider = str(getattr(llm_cfg, "provider", ""))
    model = str(getattr(llm_cfg, "model", ""))
    candidate_input = candidate
    candidate_hash = text_hash(candidate_input)
    force = bool(getattr(args, "force", False))
    scored = 0
    format_errors = 0
    unchanged = 0
    skipped = 0
    deleted = 0
    for refnr, *_ in store.list_all():
        v = store.by_refnr(refnr)
        if v is None:
            continue
        if args.missing and v.score is not None and not force:
            getter = getattr(store, "score_details", None)
            details = getter(refnr) if callable(getter) else None
            if not details or details.get("engine", "llm") == "llm":
                continue
        desc_ok = True
        try:
            _description_for(store, v, c)
        except VacancyUnavailable as e:
            skipped += 1
            print(
                f"Archived {v.refnr}: no longer available upstream ({e})",
                file=sys.stderr,
            )
            continue
        except Exception as e:
            desc_ok = False
            _mark_detail_error(store, v.refnr)
            print(f"Description for {v.refnr} unavailable: {e}", file=sys.stderr)
        skip_reason = scoreability_reason(v)
        if not skip_reason:
            skip_reason = prefilter_reason(v, c)
        if skip_reason:
            status_getter = getattr(store, "application_status", None)
            status = status_getter(v.refnr) if callable(status_getter) else ""
            if desc_ok and status not in {"applied", "prepared"}:
                deleter = getattr(store, "delete", None)
                if callable(deleter):
                    deleter(v.refnr)
                deleted += 1
                print(
                    f"Deleted without scoring {v.refnr}: {skip_reason}",
                    file=sys.stderr,
                )
            else:
                marker = getattr(store, "mark_score_skipped", None)
                if callable(marker):
                    marker(v.refnr, skip_reason)
                skipped += 1
                print(f"Skipped {v.refnr}: {skip_reason}", file=sys.stderr)
            continue
        vacancy_hash = text_hash(vacancy_input(v))
        description_hash = text_hash(v.description)
        checker = getattr(store, "needs_scoring", None)
        if callable(checker) and not checker(
            v.refnr,
            provider=provider,
            model=model,
            prompt_version=SCORING_PROMPT_VERSION,
            candidate_hash=candidate_hash,
            vacancy_hash=vacancy_hash,
            description_hash=description_hash,
            force=force,
        ):
            unchanged += 1
            continue
        if not callable(checker) and v.score is not None and not force:
            unchanged += 1
            continue
        try:
            s, reasons, missing = score_one(v, candidate, llm)
        except LLMContractError as e:
            marker = getattr(store, "mark_score_error", None)
            if callable(marker):
                marker(v.refnr)
            _record_llm_artifact(
                store,
                llm,
                "scoring",
                "failed",
                prompt_version=SCORING_PROMPT_VERSION,
                candidate_input=candidate,
                vacancy=v,
                error=str(e),
            )
            format_errors += 1
            print(f"Score format error for {refnr}: {e}", file=sys.stderr)
            continue
        except LLMError as e:
            _record_llm_artifact(
                store,
                llm,
                "scoring",
                "failed",
                prompt_version=SCORING_PROMPT_VERSION,
                candidate_input=candidate,
                vacancy=v,
                error=str(e),
            )
            store.close()
            sys.exit(f"LLM error for {refnr}: {e}")
        update_details = getattr(store, "update_score_details", None)
        if callable(update_details):
            try:
                update_details(
                    v.refnr,
                    s,
                    reasons,
                    missing,
                    provider=provider,
                    model=model,
                    prompt_version=SCORING_PROMPT_VERSION,
                    candidate_hash=candidate_hash,
                    vacancy_hash=vacancy_hash,
                    description_hash=description_hash,
                )
            except TypeError:
                update_details(v.refnr, s, reasons, missing)
        else:
            store.update_score(v.refnr, s)
        _record_llm_artifact(
            store,
            llm,
            "scoring",
            "succeeded",
            prompt_version=SCORING_PROMPT_VERSION,
            candidate_input=candidate,
            vacancy=v,
            details={"score": s, "reasons": reasons, "missing": missing},
        )
        scored += 1
        print(f"{s:5.0f}  {v.title}  ({v.company})")
        if args.verbose:
            print("  reasons:", ", ".join(reasons))
            print("  missing:", ", ".join(missing))
    store.close()
    print(f"Scored: {scored}")
    print(f"Unchanged: {unchanged}; skipped: {skipped}; deleted: {deleted}")
    if format_errors:
        print(f"Score format errors: {format_errors} (retryable)", file=sys.stderr)


def _cmd_algorithmic_score(args, criteria):
    from pathlib import Path
    from .algorithmic_score import ALGORITHM_VERSION, evaluate_algorithmic, settings_hash, vacancy_hash
    from .embeddings import CachedEmbeddings, EmbeddingError, OllamaEmbeddings
    from .scoring_inputs import collect_scoring_inputs

    cfg = criteria.scoring
    preview = bool(getattr(args, "preview", False))
    store = Store()
    reports = []
    errors = 0
    try:
        inputs = collect_scoring_inputs(store, per_file=cfg.per_file, total=cfg.total_chars)
        if not inputs.documents:
            sys.exit("Algorithmic scoring requires extractable current-profile documents or explicit profile facts")
        embeddings = CachedEmbeddings(store, OllamaEmbeddings(cfg.embeddings))
        try:
            identity = embeddings.identity
        except EmbeddingError as exc:
            sys.exit(f"Embedding error: {exc}")
        store.prune_document_vectors({document.document_id for document in inputs.documents})
        print(f"Scoring engine: {ALGORITHM_VERSION}; model: {identity.model}; digest: {identity.digest}; preview: {preview}")
        for refnr, *_ in store.list_all():
            vacancy = store.by_refnr(refnr)
            if vacancy is None:
                continue
            try:
                _description_for(store, vacancy, criteria)
            except VacancyUnavailable:
                continue
            except Exception as exc:
                _mark_detail_error(store, refnr)
                print(f"Description for {refnr} unavailable: {redact_sensitive(exc)}", file=sys.stderr)
            skip = scoreability_reason(vacancy) or prefilter_reason(vacancy, criteria)
            if skip:
                if not preview:
                    store.mark_score_skipped(refnr, skip)
                reports.append({"refnr": refnr, "state": "unscored", "skip_reason": skip})
                continue
            vhash = vacancy_hash(vacancy)
            if not preview and not store.needs_algorithmic_scoring(
                refnr, version=ALGORITHM_VERSION, settings_hash=settings_hash(cfg), model=identity.key,
                candidate_hash=inputs.candidate_hash, vacancy_hash=vhash, force=bool(getattr(args, "force", False)),
            ):
                print(f"Unchanged: {refnr}")
                continue
            # --missing includes scores whose engine/model/inputs are now stale.
            try:
                result = evaluate_algorithmic(inputs, vacancy, cfg, embeddings)
            except EmbeddingError as exc:
                errors += 1
                if not preview:
                    store.mark_score_error(refnr, "embedding_error", engine="algorithmic")
                reports.append({"refnr": refnr, "state": "score_error", "error": str(exc)})
                print(f"Embedding error for {refnr}: {exc}", file=sys.stderr)
                continue
            reports.append({"refnr": refnr, **result})
            if not preview:
                if result["score"] is None:
                    store.mark_score_skipped(refnr, result["state"])
                else:
                    store.update_algorithmic_score(refnr, result, candidate_hash=inputs.candidate_hash,
                        vacancy_hash=vhash, description_hash=text_hash(vacancy.description), settings_hash=settings_hash(cfg))
                store.record_artifact("scoring", "succeeded", refnr=refnr,
                    provider="local-embeddings", model=identity.key, candidate_hash=inputs.candidate_hash,
                    vacancy_hash=vhash, details=result)
            print(f"{result['score'] if result['score'] is not None else 'unscored'}  {vacancy.title} ({vacancy.company})")
            if getattr(args, "verbose", False):
                print("  components:", json.dumps(result["components"], ensure_ascii=False))
                print("  missing:", ", ".join(result["missing"]))
        if getattr(args, "report", None):
            path = Path(args.report)
            # Reports include candidate excerpts; use private, exclusive file creation.
            with path.open("x", encoding="utf-8") as handle:
                path.chmod(0o600)
                json.dump({"engine": ALGORITHM_VERSION, "preview": preview, "results": reports}, handle, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"Scoring report: {path}")
        print(f"Algorithmic results: {len(reports)}; errors: {errors}; cache hits: {embeddings.hits}; misses: {embeddings.misses}")
    finally:
        store.close()
    if errors:
        sys.exit("Algorithmic scoring completed with retryable embedding errors")


def cmd_write(args):
    criteria = load_criteria()
    store = Store()
    v = store.by_refnr(args.refnr)
    if v is None:
        sys.exit(f"Vacancy {args.refnr} not found")
    try:
        _description_for(store, v, criteria)
    except VacancyUnavailable as e:
        print(
            f"Vacancy {v.refnr} archived: no longer available upstream ({e})",
            file=sys.stderr,
        )
    except Exception as e:
        _mark_detail_error(store, v.refnr)
        print(f"Description for {v.refnr} unavailable: {e}", file=sys.stderr)
    llm = _llm_client_for(criteria)
    candidate_name, letter_language = _candidate_letter_settings()
    candidate = _candidate_text("letter")
    try:
        letter_text = letter(
            v,
            candidate,
            llm,
            candidate_name=candidate_name,
            preferred_language=letter_language,
        )
        path = save_letter(v, letter_text)
    except LLMError as e:
        _record_llm_artifact(
            store,
            llm,
            "letter",
            "failed",
            prompt_version=LETTER_PROMPT_VERSION,
            candidate_input=candidate,
            vacancy=v,
            error=str(e),
        )
        store.close()
        sys.exit(f"LLM error: {e}")
    _record_llm_artifact(
        store,
        llm,
        "letter",
        "succeeded",
        prompt_version=LETTER_PROMPT_VERSION,
        candidate_input=candidate,
        vacancy=v,
        output_path=str(path),
    )
    store.close()
    print(path)


def cmd_apply(args):
    criteria = load_criteria()
    store = Store()
    v = store.by_refnr(args.refnr)
    if v is None:
        sys.exit(f"Vacancy {args.refnr} not found")
    try:
        _description_for(store, v, criteria)
    except VacancyUnavailable as e:
        print(
            f"Vacancy {v.refnr} archived: no longer available upstream ({e})",
            file=sys.stderr,
        )
    except Exception as e:
        _mark_detail_error(store, v.refnr)
        print(f"Description for {v.refnr} unavailable: {e}", file=sys.stderr)
    llm = _llm_client_for(criteria)
    candidate_name, letter_language = _candidate_letter_settings()
    candidate = _candidate_text("letter")
    try:
        letter_text = letter(
            v,
            candidate,
            llm,
            candidate_name=candidate_name,
            preferred_language=letter_language,
        )
        letter_path = save_letter(v, letter_text)
    except LLMError as e:
        _record_llm_artifact(
            store,
            llm,
            "letter",
            "failed",
            prompt_version=LETTER_PROMPT_VERSION,
            candidate_input=candidate,
            vacancy=v,
            error=str(e),
        )
        store.close()
        sys.exit(f"LLM error: {e}")
    _record_llm_artifact(
        store,
        llm,
        "letter",
        "succeeded",
        prompt_version=LETTER_PROMPT_VERSION,
        candidate_input=candidate,
        vacancy=v,
        output_path=str(letter_path),
    )
    own_name = candidate_name
    other_names = [name for _account_id, name in store.list_account_records()]
    try:
        validate_letter_identity(letter_text, own_name, other_names)
    except LLMError as e:
        store.close()
        sys.exit(f"Letter validation failed: {e}")
    folder = prepare_dossier(v, letter_path)
    _record_llm_artifact(
        store,
        llm,
        "dossier",
        "succeeded",
        prompt_version=LETTER_PROMPT_VERSION,
        candidate_input=candidate,
        vacancy=v,
        output_path=str(folder),
    )
    # applied ставится только после отдельного подтверждения отправки (этап 6.1)
    store.set_status(v.refnr, "prepared")
    store.close()
    print(f"Dossier ready: {folder}")
    print("Review and confirm with: confirm-applied " + v.refnr)


def _parse_keywords(raw: str) -> dict[str, list[str]]:
    return validate_keywords(parse_json_object(raw))


def _update_criteria_keywords(keywords: list[str]):
    """Ключевые слова для текущей учётной записи хранятся в БД (не в criteria.yaml)."""
    store = Store()
    store.set_account_keywords(keywords)
    store.close()


def cmd_keywords(args):
    docs = load_candidate_text(purpose="keywords")
    if not docs:
        sys.exit("cv/ documents not found or empty")
    llm = _llm_client_for(load_criteria())
    store = Store()
    try:
        parsed = complete_json_contract(
            llm,
            KEYWORDS_SYSTEM,
            keywords_prompt(docs),
            schema_name="search_keywords",
            schema=KEYWORDS_SCHEMA,
            validator=validate_keywords,
        )
    except LLMError as e:
        _record_llm_artifact(
            store,
            llm,
            "keywords",
            "failed",
            prompt_version=KEYWORDS_PROMPT_VERSION,
            candidate_input=docs,
            error=str(e),
        )
        store.close()
        sys.exit(f"LLM error: {e}")
    de, en = parsed.get("de", []), parsed.get("en", [])
    if not de and not en:
        _record_llm_artifact(
            store,
            llm,
            "keywords",
            "failed",
            prompt_version=KEYWORDS_PROMPT_VERSION,
            candidate_input=docs,
            error="No search keywords extracted",
        )
        store.close()
        sys.exit("Unusable LLM response: no search keywords extracted")
    _record_llm_artifact(
        store,
        llm,
        "keywords",
        "succeeded",
        prompt_version=KEYWORDS_PROMPT_VERSION,
        candidate_input=docs,
        details={"de_count": len(de), "en_count": len(en)},
    )
    store.close()
    print("German:")
    for k in de:
        print(f"  {k}")
    print("English:")
    for k in en:
        print(f"  {k}")
    if args.apply:
        if not de:
            sys.exit("no German keywords — account not updated")
        _update_criteria_keywords(de)
        print(
            f"keywords updated for account '{current_user()}': "
            f"{len(de)} German keywords",
            file=sys.stderr,
        )


def cmd_confirm_applied(args):
    """Подтверждение ручной отправки: prepared → applied (этап 6.1)."""
    store = Store()
    try:
        v = store.by_refnr(args.refnr)
        if v is None:
            sys.exit(f"Vacancy {args.refnr} not found")
        status = store.application_status(args.refnr)
        if status == "applied":
            print(f"Vacancy {args.refnr} is already marked applied.")
            return
        store.set_status(args.refnr, "applied")
    finally:
        store.close()
    print(f"Vacancy {args.refnr} confirmed applied.")


def cmd_status(args):
    store = Store()
    rows = (
        store.list_archived()
        if getattr(args, "archived", False)
        else store.list_all()
    )
    for row in rows:
        print(" | ".join("" if x is None else str(x) for x in row))
    store.close()


def cmd_purge_archived(args):
    if not args.confirm:
        sys.exit("Aborted: --confirm is required to purge archived vacancies")
    store = Store()
    try:
        removed = store.purge_archived(args.older_than_days)
    finally:
        store.close()
    print(
        f"Archived vacancies permanently deleted: {removed} "
        f"(older than {args.older_than_days} days; applied rows excluded)"
    )


def cmd_prune_jobs(args):
    """Retention фоновых заданий: удаляет завершённые jobs старше N дней."""
    if not args.confirm:
        sys.exit("Aborted: --confirm is required to delete old jobs")
    if not 0 <= args.older_than_days <= 3650:
        sys.exit("older-than-days must be between 0 and 3650")
    store = Store()
    try:
        removed = store.prune_finished_jobs(args.older_than_days)
    finally:
        store.close()
    print(
        f"Abgeschlossene Jobs geloescht: {removed} "
        f"(aelter als {args.older_than_days} Tage)"
    )


def cmd_migrate_storage(args):
    """Dry-run по умолчанию; --apply включает UUID layout после backup/copy."""
    store = Store()
    try:
        report = (
            apply_storage_migration(store)
            if args.apply
            else plan_storage_migration(store)
        )
    finally:
        store.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def dotenv_overrides() -> dict[str, str]:
    """Значения .env без изменения os.environ (для env дочерних процессов)."""
    path = ROOT / ".env"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = value
    return values


def load_env():
    for key, value in dotenv_overrides().items():
        if key not in os.environ:
            os.environ[key] = value


def cmd_web(_args):
    from .webapp import main as web_main

    web_main()


def main():
    load_env()
    p = argparse.ArgumentParser(prog="cv-agent", description="Jobsuche und Anschreiben-Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="neue Stellen suchen und speichern")
    s.add_argument(
        "--total-limit",
        "--limit",
        dest="total_limit",
        type=int,
        default=None,
        help="maximale Zahl der Stellen im gesamten Profil-Lauf (Standard aus criteria.yaml)",
    )
    s.add_argument(
        "--per-query-limit",
        type=int,
        default=None,
        help="maximale Treffer pro Keyword/Kategorie (Standard aus criteria.yaml)",
    )
    s.add_argument(
        "--max-keywords",
        type=int,
        default=None,
        help="maximale Zahl deutscher Account-Keywords pro Lauf",
    )
    s.add_argument(
        "--location",
        dest="location",
        default=None,
        help=(
            "Ortsfilter fuer diesen Lauf (Stadt > Region > Land); "
            "mehrere Orte durch Komma getrennt, z. B. 'München, Hamburg'; "
            "leerer String bedeutet ohne Ortsbeschraenkung, "
            "ohne Flag gilt das Profil-/criteria-Gebiet"
        ),
    )
    s.add_argument(
        "--radius",
        dest="radius",
        type=int,
        default=None,
        help="Radius in km um den Ort (1-500, Standard aus criteria.yaml)",
    )
    s.set_defaults(func=cmd_search)

    sc = sub.add_parser("score", help="gespeicherte Stellen bewerten")
    sc.add_argument("--engine", choices=("llm", "algorithmic"), default=None)
    sc.add_argument("--preview", action="store_true", help="algorithmische Vorschau ohne Änderung gespeicherter Scores")
    sc.add_argument("--report", help="neue private JSON-Datei für algorithmische Ergebnisse")
    sc.add_argument("--verbose", action="store_true")
    sc.add_argument(
        "--missing",
        action="store_true",
        help="nur unbewertete Stellen bewerten (ohne Score)",
    )
    sc.add_argument(
        "--force",
        action="store_true",
        help="alle score-fähigen Stellen unabhängig von gespeicherten Input-Hashes neu bewerten",
    )
    sc.set_defaults(func=cmd_score)

    w = sub.add_parser("write", help="Anschreiben für eine Stelle generieren")
    w.add_argument("refnr")
    w.set_defaults(func=cmd_write)

    a = sub.add_parser("apply", help="Bewerbungsdossier vorbereiten (halbautomatisch)")
    a.add_argument("refnr")
    a.set_defaults(func=cmd_apply)

    ca = sub.add_parser(
        "confirm-applied",
        help="bestätigt manuell gesendete Bewerbung (Status 'applied')",
    )
    ca.add_argument("refnr")
    ca.set_defaults(func=cmd_confirm_applied)

    st = sub.add_parser("status", help="alle gespeicherten Stellen anzeigen")
    st.add_argument(
        "--archived",
        dest="archived",
        action="store_true",
        help="nur archivierte Stellen mit Source-Status anzeigen",
    )
    st.set_defaults(func=cmd_status)

    ac = sub.add_parser(
        "actualize",
        help="gespeicherte Stellen bei der Quelle pruefen; entfernte archivieren",
    )
    ac.set_defaults(func=cmd_actualize)

    purge = sub.add_parser(
        "purge-archived",
        help="archivierte Stellen nach expliziter Bestätigung endgültig löschen",
    )
    purge.add_argument("--confirm", action="store_true")
    purge.add_argument("--older-than-days", type=int, default=30)
    purge.set_defaults(func=cmd_purge_archived)

    pj = sub.add_parser(
        "prune-jobs",
        help="abgeschlossene Hintergrund-Jobs nach Bestätigung löschen (Retention)",
    )
    pj.add_argument("--confirm", action="store_true")
    pj.add_argument("--older-than-days", type=int, default=30)
    pj.set_defaults(func=cmd_prune_jobs)

    migration = sub.add_parser(
        "migrate-storage",
        help="CV/dossiers storage migration planen (dry-run) oder anwenden",
    )
    migration.add_argument(
        "--apply",
        action="store_true",
        help="nach Backup Dokumente kopieren und UUID layout aktivieren",
    )
    migration.set_defaults(func=cmd_migrate_storage)

    kw = sub.add_parser(
        "keywords", help="Suchbegriffe aus cv/-Dokumenten extrahieren (LLM)"
    )
    kw.add_argument("--apply", action="store_true", help="für Konto speichern (Bewerbungs-Keywords)")
    kw.set_defaults(func=cmd_keywords)

    w = sub.add_parser("web", help="Web-Interface starten (http://127.0.0.1:8000)")
    w.set_defaults(func=cmd_web)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
