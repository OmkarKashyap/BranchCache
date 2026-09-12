import argparse
import csv
import json
import time
from pathlib import Path

from agent.tree import build_independent_prefix, build_prefix
from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import run_branch
from eval.run_serving_benchmark import STRATEGIES
from serving.clients import naive_client, sglang_client, vllm_client
from serving.metrics import fetch_metrics

BASE_URLS = {
    "naive": naive_client.DEFAULT_BASE_URL,
    "vllm": vllm_client.DEFAULT_BASE_URL,
    "sglang": sglang_client.DEFAULT_BASE_URL,
}

CSV_FIELDS = [
    "problem_id",
    "prefix_length",
    "n",
    "strategy",
    "condition",
    "trial",
    "latency_ms",
    "tool_time_ms",
]


def _try_fetch_metrics(base_url):
    if base_url is None:
        return None
    try:
        return fetch_metrics(base_url)
    except Exception:
        return None


def _load_done_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {
            (row["problem_id"], row["prefix_length"], int(row["n"]), row["strategy"],
             row["condition"], int(row["trial"]))
            for row in csv.DictReader(f)
        }


def _append_row(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _save_metrics_snapshot(metrics_dir, key, before, after):
    if metrics_dir is None:
        return
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(part) for part in key) + ".json"
    (metrics_dir / name).write_text(json.dumps({"before": before, "after": after}))


def _run_cell(instance, n, make_client, make_prefix, max_steps):
    start = time.perf_counter()
    branches = [
        run_branch(instance.problem, make_client(), max_steps=max_steps, prefix=make_prefix(i))
        for i in range(n)
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000
    tool_time_ms = sum(b.tool_time_ms for b in branches)
    return elapsed_ms, tool_time_ms


def run_prefix_length_sweep(
    dataset,
    lengths,
    ns,
    strategies,
    trials,
    out_path,
    make_client_overrides=None,
    max_steps=10,
    metrics_dir=None,
):
    """Shared-prefix condition: does a longer shared prefix make caching more
    visible? Same problem, same branches, only the length of what they share
    changes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_path)
    overrides = make_client_overrides or {}

    for strategy in strategies:
        make_client = overrides.get(strategy, STRATEGIES.get(strategy))
        base_url = BASE_URLS.get(strategy)

        for instance in dataset:
            for length in lengths:
                prefix = build_prefix(instance.problem, length)

                for n in ns:
                    for trial in range(trials):
                        key = (instance.problem_id, length, n, strategy, "shared", trial)
                        if key in done:
                            continue

                        before = _try_fetch_metrics(base_url)
                        elapsed_ms, tool_time_ms = _run_cell(
                            instance, n, lambda: make_client(instance), lambda _i: prefix, max_steps
                        )
                        after = _try_fetch_metrics(base_url)

                        _append_row(
                            out_path,
                            {
                                "problem_id": instance.problem_id,
                                "prefix_length": length,
                                "n": n,
                                "strategy": strategy,
                                "condition": "shared",
                                "trial": trial,
                                "latency_ms": round(elapsed_ms, 2),
                                "tool_time_ms": round(tool_time_ms, 2),
                            },
                        )
                        _save_metrics_snapshot(metrics_dir, key, before, after)
                        done.add(key)

    return out_path


def run_no_shared_prefix_control(
    dataset,
    length,
    ns,
    strategies,
    trials,
    out_path,
    make_client_overrides=None,
    max_steps=10,
    metrics_dir=None,
):
    """Negative control: branches get distinct, non-overlapping prefixes, so
    prefix-aware serving should show close to no benefit here. If it does
    show a big win anyway, something upstream is measuring the wrong thing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_path)
    overrides = make_client_overrides or {}

    for strategy in strategies:
        make_client = overrides.get(strategy, STRATEGIES.get(strategy))
        base_url = BASE_URLS.get(strategy)

        for instance in dataset:
            for n in ns:
                for trial in range(trials):
                    key = (instance.problem_id, length, n, strategy, "independent", trial)
                    if key in done:
                        continue

                    before = _try_fetch_metrics(base_url)
                    elapsed_ms, tool_time_ms = _run_cell(
                        instance,
                        n,
                        lambda: make_client(instance),
                        lambda i: build_independent_prefix(instance.problem, i, length),
                        max_steps,
                    )
                    after = _try_fetch_metrics(base_url)

                    _append_row(
                        out_path,
                        {
                            "problem_id": instance.problem_id,
                            "prefix_length": length,
                            "n": n,
                            "strategy": strategy,
                            "condition": "independent",
                            "trial": trial,
                            "latency_ms": round(elapsed_ms, 2),
                            "tool_time_ms": round(tool_time_ms, 2),
                        },
                    )
                    _save_metrics_snapshot(metrics_dir, key, before, after)
                    done.add(key)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", default=["short", "medium", "long"])
    parser.add_argument("--control-length", default="medium")
    parser.add_argument("--ns", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--strategies", nargs="+", default=["naive", "vllm", "sglang"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="results/raw/systems_ablation.csv")
    parser.add_argument("--metrics-dir", default="results/raw/ablation_metrics")
    args = parser.parse_args()

    run_prefix_length_sweep(
        GOLDEN_DATASET,
        args.lengths,
        args.ns,
        args.strategies,
        args.trials,
        args.out,
        max_steps=args.max_steps,
        metrics_dir=args.metrics_dir,
    )
    run_no_shared_prefix_control(
        GOLDEN_DATASET,
        args.control_length,
        args.ns,
        args.strategies,
        args.trials,
        args.out,
        max_steps=args.max_steps,
        metrics_dir=args.metrics_dir,
    )

    print(f"wrote {args.out}")
    print(f"raw before/after metrics snapshots (if reachable) in {args.metrics_dir}")


if __name__ == "__main__":
    main()
