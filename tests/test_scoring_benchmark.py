import copy
import json
from pathlib import Path

import pytest
import yaml

from cv_agent.scoring_benchmark import compare_saved_scores, evaluate, load_dataset, main

SEED = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "scoring_seed.yaml"


def test_seed_has_sixty_pairs_and_separate_vacancy_groups():
    data = load_dataset(SEED)
    all_report = evaluate(data, split="all")
    assert all_report["metrics"]["cases"] == 60
    assert all_report["unique_vacancies"] == 10
    holdout = evaluate(data)
    calibration = evaluate(data, split="calibration")
    assert holdout["metrics"]["cases"] == 24
    assert calibration["metrics"]["cases"] == 36
    assert set(row["vacancy_id"] for row in holdout["results"]).isdisjoint(row["vacancy_id"] for row in calibration["results"])
    assert all_report["label_source"] == "synthetic_scenario_expectations"


def test_expected_seed_examples_are_ranked_and_explained():
    rows = {row["case_id"]: row for row in evaluate(load_dataset(SEED), split="all")["results"]}
    assert rows["python/backend"]["score"] == 100
    assert rows["frontend/frontend"]["score"] == 100
    assert rows["java/java_backend"]["score"] < 100
    assert rows["learner/backend"]["score"] < rows["python/backend"]["score"]
    assert rows["learner/rag"]["preliminary"]
    assert rows["python/backend"]["matches"][0]["evidence"][0]["document_id"] == "python-cv"


def test_missing_scores_are_excluded_from_ranking_not_converted_to_zero():
    data = copy.deepcopy(load_dataset(SEED))
    data["profiles"]["python"]["documents"] = []
    report = evaluate(data)
    assert report["per_profile"]["python"]["scored"] == 0
    assert report["per_profile"]["python"]["precision_at_k"] is None


def test_no_model_database_or_network_needed(monkeypatch, tmp_path):
    from cv_agent import llm, store

    def forbidden(*args, **kwargs):
        pytest.fail("Benchmark must not access DB or a generative model")

    monkeypatch.setattr(llm.LLMClient, "complete", forbidden)
    monkeypatch.setattr(store, "Store", forbidden)
    output = tmp_path / "report.json"
    assert main(["--dataset", str(SEED), "--split", "all", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["metrics"]["cases"] == 60


@pytest.mark.parametrize("mutation", ["group_leak", "invalid_grade", "unknown_profile", "duplicate_document"])
def test_invalid_datasets_fail_early(tmp_path, mutation):
    data = copy.deepcopy(load_dataset(SEED))
    if mutation == "group_leak":
        data["vacancies"]["rag"]["group"] = "backend"
    elif mutation == "invalid_grade":
        data["judgments"]["python"]["backend"] = True
    elif mutation == "unknown_profile":
        data["judgments"]["unknown"] = {"backend": 2}
    else:
        data["profiles"]["python"]["documents"] *= 2
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_comparison_uses_same_cases_and_rejects_invalid_saved_scores(tmp_path):
    report = evaluate(load_dataset(SEED))
    path = tmp_path / "llm.json"
    path.write_text(json.dumps({"java/integration": 90, "frontend/integration": 10}))
    comparison = compare_saved_scores(report, path, k=1)
    assert comparison["cases_compared"] == 2
    assert comparison["saved_llm"]["precision_at_k"] == 1
    assert comparison["baseline"]["cases"] == 2
    for invalid in ({"java/integration": float("nan")}, {"java/integration": True}, {"unknown/id": 50}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            compare_saved_scores(report, path, k=1)
