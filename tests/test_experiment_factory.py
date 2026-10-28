from pathlib import Path

import pytest
import yaml

from gpt1_factory.experiments.factory import build_plan, deep_merge, load_config, run_plan
from gpt1_factory.experiments.report import aggregate, paired_deltas


def test_deep_merge_keeps_unmodified_siblings():
    assert deep_merge({"data": {"task": "mnli", "batch_size": 32}},
                      {"data": {"task": "sst2"}}) == {
                          "data": {"task": "sst2", "batch_size": 32}}


def test_plan_isolated_paths_and_control_variable():
    jobs = build_plan("configs/ablations.yaml")
    assert len(jobs) == 3
    assert len({job.out_dir for job in jobs}) == 3
    assert all(job.config["data"]["task"] == "sst2" for job in jobs)
    assert all(job.config["data"]["tokenizer_artifact"] == jobs[0].config["data"]["tokenizer_artifact"] for job in jobs)
    assert len({job.config["exp"]["seed"] for job in jobs}) == 1
    assert {job.variant for job in jobs} == {"full", "no_pretrain", "no_aux"}
    assert all(job.config["data"]["max_len"] == 128 for job in jobs)
    assert all(job.config["model"]["max_len"] == 128 for job in jobs)
    assert all(job.config["data"]["batch_size"] == 8 for job in jobs)
    assert run_plan(jobs, execute=False)[0]["status"] == "planned"


def test_paper_table5_matrix_counts():
    jobs = build_plan("configs/paper_ablations.yaml")
    assert len(jobs) == 8 * 3 * 4 + 2 * 3 * 5 + 3 * 3 * 4
    assert len({j.name for j in jobs}) == len(jobs)
    lstm = [j for j in jobs if j.variant == "lstm"]
    assert lstm and all("lstm" in j.config["finetune"]["pretrained_path"] for j in lstm)


def test_unsupported_task_fails_loudly(tmp_path: Path):
    spec = yaml.safe_load(Path("configs/ablations.yaml").read_text())
    spec["suites"][0]["tasks"] = ["not_a_task"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="Unsupported GLUE"):
        build_plan(path)


def test_matched_delta_avoids_unpaired_seeds():
    rows = [
        {"suite": "t", "variant": "full", "task": "mnli", "seed": 1,
         "depth": -1, "lambda": 0.5, "metric": "acc", "score": 0.7},
        {"suite": "t", "variant": "no_pretrain", "task": "mnli", "seed": 1,
         "depth": -1, "lambda": 0.5, "metric": "acc", "score": 0.4},
        {"suite": "t", "variant": "no_pretrain", "task": "mnli", "seed": 3,
         "depth": -1, "lambda": 0.5, "metric": "acc", "score": 0.2},
    ]
    group = aggregate(rows)
    assert len(group) == 2
    difference = paired_deltas(rows)
    assert len(difference) == 1
    assert difference[0]["n_pairs"] == 1
    assert difference[0]["delta_mean"] == pytest.approx(-0.3)


def test_auxiliary_treatment_pairs_even_when_lambda_changes():
    rows = [
        {"suite": "paper_table5", "variant": name, "task": "sst2",
         "seed": 13, "depth": -1, "lambda": lam, "metric": "acc", "score": score}
        for name, lam, score in [("full", 0.5, 0.80), ("no_aux", 0.0, 0.77)]
    ]
    pair = paired_deltas(rows)
    assert len(pair) == 1 and pair[0]["n_pairs"] == 1
    assert pair[0]["delta_mean"] == pytest.approx(-0.03)
