import csv

from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import ScriptedSolverClient
from eval.run_serving_benchmark import run_sweep, summarize


def _fake_client():
    return ScriptedSolverClient(GOLDEN_DATASET[0])


def test_run_sweep_writes_one_row_per_cell(tmp_path):
    out = tmp_path / "bench.csv"
    run_sweep(
        [GOLDEN_DATASET[0]],
        ns=[1, 2],
        strategies=["naive"],
        trials=2,
        out_path=out,
        make_client_overrides={"naive": _fake_client},
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2 * 2  # 2 values of n x 2 trials, 1 problem, 1 strategy
    assert all(row["passed"] == "True" for row in rows)


def test_run_sweep_skips_already_completed_cells_on_rerun(tmp_path):
    out = tmp_path / "bench.csv"
    dataset = [GOLDEN_DATASET[0]]
    overrides = {"naive": _fake_client}

    run_sweep(dataset, [1], ["naive"], 1, out, make_client_overrides=overrides)
    run_sweep(dataset, [1], ["naive"], 1, out, make_client_overrides=overrides)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_summarize_reports_median_and_p95(tmp_path):
    out = tmp_path / "bench.csv"
    run_sweep(
        [GOLDEN_DATASET[0]],
        [1],
        ["naive"],
        trials=5,
        out_path=out,
        make_client_overrides={"naive": _fake_client},
    )

    summary = summarize(out)

    assert len(summary) == 1
    row = summary[0]
    assert row["strategy"] == "naive"
    assert row["n"] == 1
    assert row["trials"] == 5
    assert row["median_ms"] > 0
    assert row["p95_ms"] >= row["median_ms"]
