import argparse
import csv
import statistics
import time
from pathlib import Path

from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import run_branch
from serving.clients import naive_client, sglang_client, vllm_client

STRATEGIES = {
    "naive": naive_client.make_client,
    "vllm": vllm_client.make_client,
    "sglang": sglang_client.make_client,
}

CSV_FIELDS = ["problem_id", "n", "strategy", "trial", "latency_ms", "passed", "tool_time_ms"]


def _run_trial(instance, n, strategy_name, make_client, trial, max_steps):
    start = time.perf_counter()
    branches = [
        run_branch(instance.problem, make_client(), max_steps=max_steps) for _ in range(n)
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "problem_id": instance.problem_id,
        "n": n,
        "strategy": strategy_name,
        "trial": trial,
        "latency_ms": round(elapsed_ms, 2),
        "passed": any(b.test_result and b.test_result.get("passed") for b in branches),
        "tool_time_ms": round(sum(b.tool_time_ms for b in branches), 2),
    }


def _load_done_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {
            (row["problem_id"], int(row["n"]), row["strategy"], int(row["trial"]))
            for row in csv.DictReader(f)
        }


def _append_row(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(dataset, ns, strategies, trials, out_path, make_client_overrides=None, max_steps=10):
    """The main N x strategy x trial grid. Checkpointed: a cell already sitting
    in out_path gets skipped, so a killed Colab/Kaggle session can just be
    re-run and it'll pick up where it left off instead of starting over.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_path)
    overrides = make_client_overrides or {}

    for strategy_name in strategies:
        make_client = overrides.get(strategy_name, STRATEGIES.get(strategy_name))
        if make_client is None:
            raise ValueError(f"unknown strategy: {strategy_name}")

        for instance in dataset:
            for n in ns:
                for trial in range(trials):
                    key = (instance.problem_id, n, strategy_name, trial)
                    if key in done:
                        continue
                    row = _run_trial(instance, n, strategy_name, make_client, trial, max_steps)
                    _append_row(out_path, row)
                    done.add(key)

    return out_path


def _percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def summarize(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for row in rows:
        grouped.setdefault((row["strategy"], int(row["n"])), []).append(float(row["latency_ms"]))

    summary = []
    for (strategy, n), latencies in sorted(grouped.items()):
        latencies.sort()
        summary.append(
            {
                "strategy": strategy,
                "n": n,
                "median_ms": statistics.median(latencies),
                "p95_ms": _percentile(latencies, 95),
                "trials": len(latencies),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--strategies", nargs="+", default=["naive", "vllm", "sglang"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="results/raw/serving_benchmark.csv")
    args = parser.parse_args()

    out_path = run_sweep(
        GOLDEN_DATASET, args.ns, args.strategies, args.trials, args.out, max_steps=args.max_steps
    )

    print(f"\nwrote {out_path}\n")
    for row in summarize(out_path):
        print(
            f"{row['strategy']:8s} N={row['n']:<3d} "
            f"median={row['median_ms']:.1f}ms  p95={row['p95_ms']:.1f}ms  (trials={row['trials']})"
        )


if __name__ == "__main__":
    main()
