"""Explicit-file comparison of dictionary and local-vector scoring.

No production database, CV folders, generative model or automatic installation.
Threshold selection consumes calibration rows only; holdout is evaluated after
selection. Reports contain evidence and must be treated as private documents.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from .algorithmic_score import ALGORITHM_VERSION, component_score, evaluate_algorithmic, task_contribution
from .embeddings import CachedEmbeddings, EmbeddingError, OllamaEmbeddings
from .models import Vacancy
from .scoring_benchmark import _metrics, dataset_hash, evaluate as evaluate_baseline, load_dataset
from .scoring_config import ScoringConfig, parse_scoring_config
from .scoring_inputs import ScoringInputs
from .scoring_location import parse_location_preferences
from .scoring_rules import Document

REPORT_VERSION = 1
DEFAULT_GRID = {"similarity_low": [.35, .45, .55], "similarity_high": [.75, .85, .95], "min_score": [50, 60, 70, 80]}


class MemoryVectors:
    """Benchmark-only store implementing the production vector-cache interface."""
    def __init__(self):
        self.vectors = {}

    def embedding_vectors(self, kind, entity, backend, role):
        return self.vectors.get((kind, entity, backend, role), {})

    def replace_embedding_vectors(self, kind, entity, backend, role, dimensions, vectors):
        self.vectors[(kind, entity, backend, role)] = dict(vectors)


def _finite(value, low=0., high=100.) -> bool:
    return type(value) in {int, float} and low <= value <= high and math.isfinite(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def threshold_metrics(rows: list[dict], min_score: float) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        selected = row["score"] is not None and row["score"] >= min_score
        positive = row["grade"] == 2
        tp += selected and positive
        fp += selected and not positive
        fn += not selected and positive
        tn += not selected and not positive
    return {"min_score": min_score, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
            "unscored": sum(row["score"] is None for row in rows),
            "suitable_unscored": sum(row["score"] is None and row["grade"] == 2 for row in rows)}


def summarize(rows: list[dict], *, k: int, min_score: float) -> dict:
    profiles = sorted({row["profile_id"] for row in rows})
    per_profile = {}
    for profile in profiles:
        subset = [row for row in rows if row["profile_id"] == profile]
        per_profile[profile] = {"ranking": _metrics(subset, k), "threshold": threshold_metrics(subset, min_score)}
    suitable = [row for row in rows if row["grade"] == 2]
    false_mismatches = [row["case_id"] for row in suitable if any(
        match["status"] == "mismatch" and match["requirement"]["importance"] == "mandatory"
        for match in row.get("matches", []))]
    return {"cases": len(rows), "unique_vacancies": len({row["vacancy_id"] for row in rows}),
            "per_profile": per_profile,
            "macro": {"precision_at_k": _mean(item["ranking"]["precision_at_k"] for item in per_profile.values()),
                      "ndcg_at_k": _mean(item["ranking"]["ndcg_at_k"] for item in per_profile.values()),
                      "f1": _mean(item["threshold"]["f1"] for item in per_profile.values())},
            "threshold": threshold_metrics(rows, min_score),
            "suitable_mandatory_mismatch_cases": false_mismatches,
            "suitable_mandatory_mismatch_rate": round(len(false_mismatches) / len(suitable), 4) if suitable else None}


def rescore(row: dict, cfg: ScoringConfig) -> dict:
    """Replay exact cached similarities through the shared production formula."""
    result = dict(row)
    if row["state"] != "scored":
        result["score"] = None
        return result
    components = copy.deepcopy(row["components"])
    tasks = components["tasks"]["requirements"]
    components["tasks"]["value"] = sum(round(task_contribution(match["similarity_raw"], cfg), 6) for match in row["task_matches"]) / tasks if tasks else 0.
    mismatch = any(match["status"] == "mismatch" and match["requirement"]["importance"] == "mandatory" for match in row["matches"])
    result["score"], result["uncapped_score"] = component_score(components, cfg, scoreable=True, mandatory_mismatch=mismatch)
    return result


def calibrate(rows: list[dict], cfg: ScoringConfig, *, k: int = 3, grid: dict | None = None) -> dict:
    """Select thresholds on calibration only, even if callers pass mixed rows."""
    training = [row for row in rows if row["split"] == "calibration"]
    if not any(row["grade"] == 2 for row in training) or not any(row["grade"] < 2 for row in training):
        raise ValueError("Calibration needs both suitable and other reviewed examples")
    grid = grid if grid is not None else DEFAULT_GRID
    if not isinstance(grid, dict) or set(grid) != set(DEFAULT_GRID):
        raise ValueError("Invalid calibration grid fields")
    for field, values in grid.items():
        maximum = 100 if field == "min_score" else 1
        if not isinstance(values, list) or not 1 <= len(values) <= 20 or any(not _finite(value, 0, maximum) for value in values):
            raise ValueError("Invalid calibration grid values")
    if math.prod(len(values) for values in grid.values()) > 1000:
        raise ValueError("Calibration grid is limited to 1000 combinations")
    trials = []
    for low, high, threshold in itertools.product(grid["similarity_low"], grid["similarity_high"], grid["min_score"]):
        if low >= high:
            continue
        proposal = replace(cfg, similarity_low=low, similarity_high=high)
        predicted = [rescore(row, proposal) for row in training]
        summary = summarize(predicted, k=k, min_score=threshold)
        trials.append({"similarity_low": low, "similarity_high": high, "min_score": threshold, "summary": summary})
    if not trials:
        raise ValueError("Calibration grid has no valid similarity interval")
    # Prespecified tie-break: F1, NDCG, fewer FP, then the original interval and
    # threshold 70. No holdout score/label participates in choosing a proposal.
    def objective(trial):
        summary = trial["summary"]
        return (summary["macro"]["f1"] or 0., summary["macro"]["ndcg_at_k"] or 0.,
                -summary["threshold"]["fp"],
                -abs(trial["similarity_low"] - cfg.similarity_low) - abs(trial["similarity_high"] - cfg.similarity_high),
                -abs(trial["min_score"] - 70), -trial["min_score"])
    selected = max(trials, key=objective)
    fingerprint = [{key: row[key] for key in ("case_id", "grade", "score", "components", "matches", "task_matches", "state")} for row in training]
    return {"selection_split": "calibration", "selection_hash": dataset_hash({"training": fingerprint, "grid": grid}),
            "case_ids": [row["case_id"] for row in training], "selected": selected,
            "trials": trials, "applied_to_production": False}


def _inputs(profile: dict, cfg: ScoringConfig, profile_id: str) -> ScoringInputs:
    documents = []
    remaining = cfg.total_chars
    truncated = False
    for item in profile["documents"]:
        text = item["text"][:min(cfg.per_file, remaining)]
        truncated |= len(text) < len(item["text"])
        remaining -= len(text)
        if text:
            documents.append(Document(item["id"], item["kind"], text))
    prefs = parse_location_preferences(profile.get("scoring_preferences"))
    return ScoringInputs(tuple(documents), dataset_hash({"profile": profile, "id": profile_id}), (), truncated, prefs)


def run_evaluation(data: dict, models: dict[str, ScoringConfig], *, k: int = 3, min_score: float = 70,
                   backend_factory=None, progress=None) -> dict:
    if os.environ.get("CV_AGENT_RUN_ID"):
        raise ValueError("Run benchmark outside production jobs (unset CV_AGENT_RUN_ID)")
    if type(k) is not int or k < 1 or not _finite(min_score) or not models:
        raise ValueError("Invalid evaluation k, min_score or models")
    factory = backend_factory or OllamaEmbeddings
    baseline = evaluate_baseline(data, split="all", k=k)
    report = {"version": REPORT_VERSION, "dataset_hash": dataset_hash(data), "label_source": data["label_source"],
              "algorithm_version": ALGORITHM_VERSION, "k": k, "min_score": min_score,
              "baseline": {"results": baseline["results"],
                           "splits": {split: summarize([row for row in baseline["results"] if row["split"] == split], k=k, min_score=min_score) for split in ("calibration", "holdout")}},
              "models": {}, "limitations": [
                  "Synthetic labels are not human validation of real vacancies.",
                  "Cold means initially empty vector cache; the model may already be resident.",
                  "Warm pass reuses an in-memory benchmark cache; production uses SQLite.",
                  "No production DB, real CV folders, LLM client or provider keys are used.",
                  "Latency includes rules and local requests, not PDF extraction or source HTTP.",
                  "Ollama peak RAM/CPU time are not measured by the Python client.",
                  "Threshold proposals never change criteria.yaml or enable a scoring engine.",
              ]}
    for name, cfg in models.items():
        if progress:
            progress(f"Model {name}: first pass")
        backend = factory(cfg.embeddings)
        tick = time.perf_counter()
        cold_requests = getattr(backend, "requests", 0)
        identity = backend.identity
        caches = {profile: CachedEmbeddings(MemoryVectors(), backend) for profile in data["judgments"]}
        results = []
        for profile_id, judgments in data["judgments"].items():
            inputs = _inputs(data["profiles"][profile_id], cfg, profile_id)
            cache = caches[profile_id]
            for vacancy_id, grade in judgments.items():
                item = data["vacancies"][vacancy_id]
                vacancy = Vacancy(vacancy_id, item.get("title", "Benchmark vacancy"), "Synthetic benchmark", item.get("location", ""), "", "benchmark", description=item["text"])
                started = time.perf_counter()
                result = evaluate_algorithmic(inputs, vacancy, cfg, cache)
                results.append({"case_id": f"{profile_id}/{vacancy_id}", "profile_id": profile_id, "vacancy_id": vacancy_id,
                                "split": item["split"], "grade": grade,
                                "elapsed_ms": round(1000 * (time.perf_counter() - started), 3), **result})
        cold_seconds = time.perf_counter() - tick
        cold_requests = getattr(backend, "requests", 0) - cold_requests
        misses = sum(cache.misses for cache in caches.values())
        if progress:
            progress(f"Model {name}: repeat pass")
        started = time.perf_counter()
        warm_requests = getattr(backend, "requests", 0)
        warm_scores = []
        for row in results:
            profile_id, vacancy_id = row["profile_id"], row["vacancy_id"]
            item = data["vacancies"][vacancy_id]
            vacancy = Vacancy(vacancy_id, item.get("title", "Benchmark vacancy"), "Synthetic benchmark", item.get("location", ""), "", "benchmark", description=item["text"])
            warm = evaluate_algorithmic(_inputs(data["profiles"][profile_id], cfg, profile_id), vacancy, cfg, caches[profile_id])
            warm_scores.append(warm["score"])
        warm_seconds = time.perf_counter() - started
        stable = warm_scores == [row["score"] for row in results]
        if not stable:
            raise ValueError("Repeated fixed-vector evaluation changed scores")
        repeat_requests = getattr(backend, "requests", 0) - warm_requests
        try:
            resident = backend.resident_model() if hasattr(backend, "resident_model") else None
        except EmbeddingError:
            resident = None  # Optional telemetry never changes scoring results.
        calibration = calibrate(results, cfg, k=k)
        chosen = calibration["selected"]
        proposed = replace(cfg, similarity_low=chosen["similarity_low"], similarity_high=chosen["similarity_high"])
        holdout = [row for row in results if row["split"] == "holdout"]
        report["models"][name] = {"backend": {**asdict(identity), "key": identity.key}, "config": asdict(cfg), "results": results,
                                  "resident_model_snapshot": resident,
                                  "timing": {"cold_seconds": round(cold_seconds, 4), "warm_seconds": round(warm_seconds, 4),
                                             "cold_requests": cold_requests, "warm_requests": repeat_requests,
                                             "vector_misses_first_pass": misses, "vector_misses_repeat_pass": sum(cache.misses for cache in caches.values()) - misses,
                                             "scores_identical": stable},
                                  "splits": {split: summarize([row for row in results if row["split"] == split], k=k, min_score=min_score) for split in ("calibration", "holdout")},
                                  "calibration": calibration,
                                  "proposed_holdout": summarize([rescore(row, proposed) for row in holdout], k=k, min_score=chosen["min_score"])}
        if progress:
            progress(f"Model {name}: {len(results)} cases; first {cold_seconds:.2f}s, repeat {warm_seconds:.2f}s")
    return report


def compare_llm(report: dict, saved: dict) -> dict:
    if not isinstance(saved, dict) or saved.get("version") != 1 or saved.get("dataset_hash") != report["dataset_hash"]:
        raise ValueError("Saved LLM scores must match evaluation version and dataset hash")
    for field in ("provider", "model", "prompt_version"):
        if not isinstance(saved.get(field), str) or not saved[field].strip():
            raise ValueError("Saved LLM scores require provider, model and prompt_version provenance")
    scores = saved.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Saved LLM scores must contain a nonempty scores mapping")
    known = {row["case_id"]: row for row in report["baseline"]["results"]}
    if set(scores) - set(known) or any(score is not None and not _finite(score) for score in scores.values()):
        raise ValueError("Invalid saved LLM case ID or score")
    rows = [{**known[case_id], "score": score, "matches": []} for case_id, score in scores.items()]
    def summaries(subset):
        return {split: summarize([row for row in subset if row["split"] == split], k=report["k"], min_score=report["min_score"]) for split in ("calibration", "holdout")}
    return {"provenance": {field: saved[field] for field in ("provider", "model", "prompt_version")},
            "cases_compared": len(rows), "cases_available": len(known), "saved_llm": summaries(rows),
            "baseline": summaries([known[case_id] for case_id in scores]),
            "models": {name: summaries([row for row in model["results"] if row["case_id"] in scores]) for name, model in report["models"].items()}}


def write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def prepare_review(data: dict) -> dict:
    draft = copy.deepcopy(data)
    draft["label_source"] = "pending_human_review"
    draft["review_instructions"] = "Fill each grade: 2 suitable, 1 borderline, 0 unsuitable; record reasons and mandatory conditions. Replace label_source only after review. Synthetic texts remain synthetic."
    draft["review_notes"] = {}
    for profile, judgments in draft["judgments"].items():
        draft["review_notes"][profile] = {}
        for vacancy in judgments:
            judgments[vacancy] = None
            draft["review_notes"][profile][vacancy] = {"reason": "", "mandatory_checks": [], "evidence_checks": []}
    return draft


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-template", action="store_true")
    parser.add_argument("--llm-scores", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=70)
    args = parser.parse_args(argv)
    try:
        data = load_dataset(args.dataset, allow_unreviewed=args.review_template)
        if args.review_template:
            if args.models or args.llm_scores:
                raise ValueError("Review template does not use models or saved scores")
            write_private(args.output, yaml.safe_dump(prepare_review(data), allow_unicode=True, sort_keys=False))
        else:
            if not args.models:
                raise ValueError("Explicit --models file is required for vector evaluation")
            model_file = yaml.safe_load(args.models.read_text(encoding="utf-8"))
            if not isinstance(model_file, dict) or type(model_file.get("version")) is not int or model_file["version"] != 1 or set(model_file) != {"version", "models"}:
                raise ValueError("Expected version 1 model configuration")
            models = model_file["models"]
            if not isinstance(models, dict) or not 1 <= len(models) <= 5:
                raise ValueError("Configure one to five explicit local models")
            if any(not isinstance(name, str) or not name.strip() or len(name) > 80 for name in models):
                raise ValueError("Model labels must be nonempty strings up to 80 characters")
            configs = {name: parse_scoring_config({"engine": "algorithmic", "embeddings": options}) for name, options in models.items()}
            report = run_evaluation(data, configs, k=args.k, min_score=args.min_score, progress=print)
            if args.llm_scores:
                report["llm_comparison"] = compare_llm(report, json.loads(args.llm_scores.read_text(encoding="utf-8")))
            write_private(args.output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Evaluation output: {args.output}")
    except (EmbeddingError, ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
