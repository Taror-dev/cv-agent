"""Offline scoring benchmark; all input/output paths are explicit.

python -m cv_agent.scoring_benchmark --dataset docs/benchmarks/scoring_seed.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import yaml

from .scoring_rules import BASELINE_VERSION, Document, dictionary_baseline, rules_fingerprint
from .scoring_location import parse_location_preferences


def dataset_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_dataset(path: Path, *, allow_unreviewed: bool = False) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Expected benchmark dataset version 1")
    for key in ("profiles", "vacancies", "judgments"):
        if not isinstance(data.get(key), dict) or not data[key]:
            raise ValueError(f"Dataset {key} must be a non-empty mapping")
    if not isinstance(data.get("label_source"), str) or not data["label_source"].strip():
        raise ValueError("Dataset must declare label_source")
    if data["label_source"] == "pending_human_review" and not allow_unreviewed:
        raise ValueError("Dataset is pending human review; fill grades 0/1/2 and declare reviewed label_source")
    groups = {}
    texts = {}
    for vacancy_id, vacancy in data["vacancies"].items():
        if not isinstance(vacancy_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", vacancy_id) or not isinstance(vacancy, dict):
            raise ValueError("Invalid vacancy record")
        if vacancy.get("split") not in {"calibration", "holdout"}:
            raise ValueError("Vacancy split must be calibration or holdout")
        if not isinstance(vacancy.get("text"), str) or not vacancy["text"].strip():
            raise ValueError("Vacancy text must be non-empty")
        group = vacancy.get("group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Vacancy must have a leakage-control group")
        if group in groups and groups[group] != vacancy["split"]:
            raise ValueError("Related vacancies must not cross calibration/holdout split")
        groups[group] = vacancy["split"]
        normalized = " ".join(vacancy["text"].casefold().split())
        if normalized in texts and texts[normalized] != vacancy["split"]:
            raise ValueError("Duplicate vacancy texts must not cross calibration/holdout split")
        texts[normalized] = vacancy["split"]
        for field in ("title", "location"):
            if not isinstance(vacancy.get(field, ""), str):
                raise ValueError(f"Vacancy {field} must be a string")
    for profile_id, profile in data["profiles"].items():
        if not isinstance(profile_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", profile_id) or not isinstance(profile, dict) or not isinstance(profile.get("documents"), list):
            raise ValueError("Invalid profile record")
        parse_location_preferences(profile.get("scoring_preferences"))
        document_ids = set()
        for document in profile["documents"]:
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise ValueError("Invalid document record")
            if not isinstance(document.get("id"), str) or not isinstance(document.get("kind"), str):
                raise ValueError("Document ID and kind must be strings")
            item = Document(document["id"], document["kind"], document["text"])
            if item.document_id in document_ids:
                raise ValueError("Duplicate document ID in profile")
            document_ids.add(item.document_id)
    for profile_id, judgments in data["judgments"].items():
        if profile_id not in data["profiles"] or not isinstance(judgments, dict) or not judgments:
            raise ValueError("Invalid judgment profile")
        for vacancy_id, grade in judgments.items():
            if grade is None and allow_unreviewed and vacancy_id in data["vacancies"]:
                continue
            if vacancy_id not in data["vacancies"] or type(grade) is not int or grade not in {0, 1, 2}:
                raise ValueError("Invalid judgment vacancy or grade")
    return data


def _metrics(rows: list[dict], k: int) -> dict:
    scored = sorted((row for row in rows if row["score"] is not None), key=lambda row: (-row["score"], row["case_id"]))
    top = scored[:k]
    suitable = sum(row["grade"] == 2 for row in rows)
    suitable_top = sum(row["grade"] == 2 for row in top)
    # Relevance for NDCG is graded; Precision explicitly counts only grade 2.
    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    ideal = dcg(sorted((row["grade"] for row in rows), reverse=True)[:k])
    return {
        "cases": len(rows), "scored": len(scored), "suitable": suitable,
        "k": len(top),
        "precision_at_k": round(suitable_top / len(top), 4) if top else None,
        "recall_at_k": round(suitable_top / suitable, 4) if suitable else None,
        "ndcg_at_k": round(dcg([row["grade"] for row in top]) / ideal, 4) if ideal else None,
        "top_cases": [row["case_id"] for row in top],
    }


def evaluate(data: dict, *, split: str = "holdout", k: int = 10) -> dict:
    if split not in {"all", "calibration", "holdout"} or type(k) is not int or k < 1:
        raise ValueError("Invalid benchmark split or k")
    results = []
    started = time.perf_counter()
    for profile_id, judgments in data["judgments"].items():
        documents = [Document(document["id"], document["kind"], document["text"]) for document in data["profiles"][profile_id]["documents"]]
        for vacancy_id, grade in judgments.items():
            vacancy = data["vacancies"][vacancy_id]
            if split != "all" and vacancy["split"] != split:
                continue
            tick = time.perf_counter()
            result = dictionary_baseline(documents, vacancy["text"])
            results.append({
                "case_id": f"{profile_id}/{vacancy_id}", "profile_id": profile_id,
                "vacancy_id": vacancy_id, "split": vacancy["split"], "grade": grade,
                "elapsed_ms": round(1000 * (time.perf_counter() - tick), 3),
                **result.to_dict(),
            })
    elapsed = time.perf_counter() - started
    return {
        "engine": BASELINE_VERSION, "rules_hash": rules_fingerprint(),
        "dataset_hash": dataset_hash(data),
        "label_source": data["label_source"], "split": split,
        "unique_vacancies": len({row["vacancy_id"] for row in results}),
        "elapsed_seconds": round(elapsed, 4),
        "metrics": _metrics(results, k),
        "per_profile": {profile_id: _metrics([row for row in results if row["profile_id"] == profile_id], k) for profile_id in data["judgments"]},
        "results": results,
        "limitations": [
            "Dictionary baseline, not production scoring or the final weighted formula.",
            "Synthetic labels do not validate quality on real vacancies.",
            "Pooled ranking is diagnostic; per-profile ranking is the application scenario.",
            "Latency covers in-memory rules only, not document extraction, LLM or embeddings.",
        ],
    }


def compare_saved_scores(report: dict, path: Path, *, k: int) -> dict:
    """Import previously exported LLM scores; never call a provider from the benchmark."""
    scores = json.loads(path.read_text(encoding="utf-8"))
    known = {row["case_id"]: row for row in report["results"]}
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Saved scores must map case IDs to numbers or null")
    comparable = []
    for case_id, score in scores.items():
        if case_id not in known:
            raise ValueError(f"Unknown case ID in saved scores: {case_id}")
        if score is not None and (type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError("Saved score must be finite and within 0–100")
        comparable.append({**known[case_id], "score": score})
    return {
        "cases_compared": len(comparable),
        "baseline": _metrics([known[row["case_id"]] for row in comparable], k),
        "saved_llm": _metrics(comparable, k),
        "provenance": "Imported scores; provider/model provenance must accompany source file.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "calibration", "holdout"), default="holdout")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llm-scores", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_dataset(args.dataset), split=args.split, k=args.k)
        if args.llm_scores:
            report["comparison"] = compare_saved_scores(report, args.llm_scores, k=args.k)
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Benchmark report: {args.output}")
        print(json.dumps(report["metrics"], ensure_ascii=False))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
