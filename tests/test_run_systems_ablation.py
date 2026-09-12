import csv

from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import ScriptedSolverClient
from eval.run_systems_ablation import (
    _try_fetch_metrics,
    run_no_shared_prefix_control,
    run_prefix_length_sweep,
)


def _fake_client(instance):
    return ScriptedSolverClient(instance)


def test_prefix_length_sweep_writes_one_row_per_cell(tmp_path):
    out = tmp_path / "ablation.csv"
    run_prefix_length_sweep(
        [GOLDEN_DATASET[0]],
        lengths=["short", "medium"],
        ns=[2],
        strategies=["naive"],
        trials=2,
        out_path=out,
        make_client_overrides={"naive": _fake_client},
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2 * 2  # 2 lengths x 2 trials, 1 problem, 1 n, 1 strategy
    assert all(row["condition"] == "shared" for row in rows)
    assert {row["prefix_length"] for row in rows} == {"short", "medium"}


def test_no_shared_prefix_control_writes_one_row_per_cell(tmp_path):
    out = tmp_path / "ablation.csv"
    run_no_shared_prefix_control(
        [GOLDEN_DATASET[0]],
        length="medium",
        ns=[2],
        strategies=["naive"],
        trials=2,
        out_path=out,
        make_client_overrides={"naive": _fake_client},
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert all(row["condition"] == "independent" for row in rows)
    assert all(row["prefix_length"] == "medium" for row in rows)


def test_both_conditions_can_share_one_csv(tmp_path):
    out = tmp_path / "ablation.csv"
    dataset = [GOLDEN_DATASET[0]]
    overrides = {"naive": _fake_client}

    run_prefix_length_sweep(
        dataset, ["medium"], [2], ["naive"], 1, out, make_client_overrides=overrides
    )
    run_no_shared_prefix_control(
        dataset, "medium", [2], ["naive"], 1, out, make_client_overrides=overrides
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    conditions = {row["condition"] for row in rows}
    assert conditions == {"shared", "independent"}
    assert len(rows) == 2


def test_sweep_skips_already_completed_cells_on_rerun(tmp_path):
    out = tmp_path / "ablation.csv"
    dataset = [GOLDEN_DATASET[0]]
    overrides = {"naive": _fake_client}

    run_prefix_length_sweep(
        dataset, ["medium"], [2], ["naive"], 1, out, make_client_overrides=overrides
    )
    run_prefix_length_sweep(
        dataset, ["medium"], [2], ["naive"], 1, out, make_client_overrides=overrides
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_try_fetch_metrics_returns_none_when_unreachable():
    assert _try_fetch_metrics("http://localhost:1/v1") is None


def test_try_fetch_metrics_returns_none_for_no_base_url():
    assert _try_fetch_metrics(None) is None
