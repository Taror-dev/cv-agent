import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from cv_agent.scoring_benchmark import load_dataset
from cv_agent.scoring_config import EmbeddingConfig, ScoringConfig
from cv_agent.scoring_evaluation import calibrate, compare_llm, main, rescore, run_evaluation, summarize, threshold_metrics, write_private
from test_embeddings import FixedBackend

SEED = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "scoring_tasks_seed.yaml"
CFG = ScoringConfig(engine="algorithmic", embeddings=EmbeddingConfig(model="test:1", dimensions=2))


@pytest.fixture
def report(monkeypatch):
    from cv_agent import llm, store
    monkeypatch.setattr(llm, "LLMClient", lambda *args, **kwargs: pytest.fail("No generative client in evaluation"))
    monkeypatch.setattr(store, "Store", lambda *args, **kwargs: pytest.fail("No production DB in evaluation"))
    return run_evaluation(load_dataset(SEED), {"fixed": CFG}, backend_factory=lambda cfg: FixedBackend())


def test_evaluation_uses_real_formula_all_pairs_and_warm_cache(report):
    assert report["baseline"]["splits"]["calibration"]["cases"] == 40
    assert report["models"]["fixed"]["splits"]["holdout"]["cases"] == 20
    model = report["models"]["fixed"]
    assert model["timing"]["vector_misses_first_pass"] > 0
    assert model["timing"]["vector_misses_repeat_pass"] == 0
    assert model["timing"]["scores_identical"]
    assert len(model["results"]) == 60
    assert model["calibration"]["selection_split"] == "calibration"
    assert not model["calibration"]["applied_to_production"]
    assert model["backend"]["dimensions"] == 2
    for row in model["results"]:
        assert rescore(row, CFG)["score"] == row["score"]
    learner = [row for row in model["results"] if row["profile_id"] == "learner"]
    assert all(match["evidence"]["kind"] == "cv" for row in learner for match in row["task_matches"])


def test_holdout_poisoning_cannot_change_parameter_selection(report):
    rows = report["models"]["fixed"]["results"]
    first = calibrate(rows, CFG)
    poisoned = copy.deepcopy(rows)
    for row in poisoned:
        if row["split"] == "holdout":
            row["grade"] = 2 if row["grade"] == 0 else 0
            row["score"] = 100
            row["components"] = None
            row["matches"] = None
            row["task_matches"] = None
    second = calibrate(poisoned, CFG)
    assert first == second
    assert all(case.split("/")[1] not in {"integrations_ru", "frontend_de", "reliability_en", "office_ru"} for case in first["case_ids"])


def test_replay_handles_caps_unknown_and_rounded_contributions(report):
    row = copy.deepcopy(report["models"]["fixed"]["results"][0])
    row["task_matches"][0]["similarity_raw"] = .684467890123
    replay = rescore(row, CFG)
    assert replay["score"] is not None
    assert replay["score"] == rescore(row, CFG)["score"]
    row["matches"].append({"status": "mismatch", "requirement": {"importance": "mandatory"}})
    assert rescore(row, replace(CFG, mandatory_mismatch_cap=25))["score"] == 25
    row["state"], row["score"] = "candidate_unavailable", None
    assert rescore(row, CFG)["score"] is None


def test_threshold_metrics_count_unscored_as_missing_suitable_not_zero():
    rows = [{"grade": 2, "score": None}, {"grade": 2, "score": 70}, {"grade": 1, "score": 70}, {"grade": 0, "score": 0}]
    result = threshold_metrics(rows, 70)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["suitable_unscored"] == 1
    assert result["f1"] == .5
    assert threshold_metrics(rows, 0)["fn"] == 1


def test_suitable_mandatory_mismatch_is_reported_separately(report):
    row = copy.deepcopy(report["models"]["fixed"]["results"][0])
    row["grade"] = 2
    row["matches"] = [{"status": "mismatch", "requirement": {"importance": "mandatory"}}]
    summary = summarize([row], k=3, min_score=70)
    assert summary["suitable_mandatory_mismatch_cases"] == [row["case_id"]]
    assert summary["suitable_mandatory_mismatch_rate"] == 1


def test_llm_comparison_requires_hash_and_provenance_and_uses_identical_cases(report):
    saved = {"version": 1, "dataset_hash": report["dataset_hash"], "provider": "offline-import", "model": "recorded-model", "prompt_version": "scoring.v2",
             "scores": {"backend/api_de": 80, "office/api_de": None}}
    comparison = compare_llm(report, saved)
    assert comparison["cases_compared"] == 2
    assert comparison["cases_available"] == 60
    assert comparison["saved_llm"]["calibration"]["threshold"]["tp"] == 1
    assert comparison["models"]["fixed"]["calibration"]["cases"] == 2
    for field, value in [("dataset_hash", "stale"), ("provider", ""), ("scores", {"missing/case": 50}), ("scores", {"backend/api_de": True}), ("scores", {"backend/api_de": float("nan")})]:
        bad = {**saved, field: value}
        with pytest.raises(ValueError):
            compare_llm(report, bad)


@pytest.mark.parametrize("grid", [
    {"min_score": [70]}, {"similarity_low": [], "similarity_high": [.8], "min_score": [70]},
    {"similarity_low": [.9], "similarity_high": [.8], "min_score": [70]},
    {"similarity_low": [True], "similarity_high": [.8], "min_score": [70]},
    {"similarity_low": [.4], "similarity_high": [.8], "min_score": [float("inf")]},
])
def test_invalid_parameter_grids_fail(report, grid):
    with pytest.raises(ValueError):
        calibrate(report["models"]["fixed"]["results"], CFG, grid=grid)


def test_review_template_requires_actual_grades_and_never_calls_models(tmp_path, monkeypatch):
    import cv_agent.scoring_evaluation as evaluation
    monkeypatch.setattr(evaluation, "OllamaEmbeddings", lambda *args: pytest.fail("Template must not call a model"))
    target = tmp_path / "review.yaml"
    assert main(["--dataset", str(SEED), "--review-template", "--output", str(target)]) == 0
    draft = yaml.safe_load(target.read_text())
    assert draft["label_source"] == "pending_human_review"
    assert all(grade is None for items in draft["judgments"].values() for grade in items.values())
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        load_dataset(target)
    # Existing ungraded real-data drafts can be prepared without placeholder grades.
    assert main(["--dataset", str(target), "--review-template", "--output", str(tmp_path / "draft-copy.yaml")]) == 0
    with pytest.raises(SystemExit):
        main(["--dataset", str(SEED), "--review-template", "--output", str(target)])


def test_outputs_are_private_exclusive_and_do_not_overwrite_symlinks(tmp_path):
    target = tmp_path / "report.json"
    write_private(target, "original")
    assert target.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_private(link, "changed")
    assert target.read_text() == "original"


def test_cli_runs_explicit_models_and_missing_model_is_error(tmp_path, monkeypatch):
    import cv_agent.scoring_evaluation as evaluation
    config = tmp_path / "models.yaml"
    config.write_text("version: 1\nmodels:\n  fixed: {model: test:1, dimensions: 2}\n")
    monkeypatch.setattr(evaluation, "OllamaEmbeddings", lambda cfg: FixedBackend())
    target = tmp_path / "report.json"
    assert main(["--dataset", str(SEED), "--models", str(config), "--output", str(target)]) == 0
    assert json.loads(target.read_text())["algorithm_version"] == "algorithmic.v3"
    assert target.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(evaluation, "OllamaEmbeddings", lambda cfg: (_ for _ in ()).throw(evaluation.EmbeddingError("Not installed")))
    with pytest.raises(SystemExit) as error:
        main(["--dataset", str(SEED), "--models", str(config), "--output", str(tmp_path / "missing.json")])
    assert error.value.code == 2
    assert not (tmp_path / "missing.json").exists()


def test_duplicate_normalized_text_does_not_leak_across_splits(tmp_path):
    data = load_dataset(SEED)
    data["vacancies"]["frontend_de"]["text"] = data["vacancies"]["api_de"]["text"].upper().replace("\n", "   ")
    target = tmp_path / "leak.yaml"
    target.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Duplicate vacancy texts"):
        load_dataset(target)


def test_benchmark_refuses_inherited_production_job(monkeypatch):
    monkeypatch.setenv("CV_AGENT_RUN_ID", "42")
    with pytest.raises(ValueError, match="outside production"):
        run_evaluation(load_dataset(SEED), {"fixed": CFG}, backend_factory=lambda cfg: pytest.fail("No call from production job"))
