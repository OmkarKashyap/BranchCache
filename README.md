# BranchCache

**Cache-aware inference for agentic code repair.**

## The question

How much redundant inference computation do branching agentic workloads create, and how
effectively can prefix-aware serving eliminate it?

BranchCache is a tree-structured, test-driven best-of-N code-repair agent: given a buggy file
and a failing test suite, it forks N parallel debugging branches from a shared context prefix,
runs each through a custom ReAct tool-calling loop, and verifies candidate patches against
pytest. The same agent and problem set are then benchmarked across three serving strategies —
naive independent inference calls, vLLM automatic prefix caching, and SGLang RadixAttention —
to isolate how much of the branching overhead is redundant prefill computation, and how much of
that redundancy prefix-aware serving actually removes. Every latency claim is backed by a
mechanism → measurement → outcome chain (prefill vs. decode vs. tool-execution time), not a
single end-to-end number.

## Status

🚧 Under active development. This repo is being built in phases (agent core → eval harness →
serving layer → GPU sweeps → results/deliverables); results, plots, and the architecture
diagram will land once real experiments have been run — nothing below is a projection or a
placeholder for numbers that don't exist yet.

## Repo structure

```
agent/      # ReAct tool-calling loop, shared-prefix tree construction, pytest verifier
serving/    # naive / vLLM / SGLang client wrappers, launch scripts, metrics scraping
eval/       # golden dataset, correctness eval, serving benchmark, systems ablation
scripts/    # dataset setup, Colab/Kaggle notebooks, plotting, dashboard build
docker/     # CPU agent image + CUDA vLLM/SGLang server images
tests/      # CPU-only unit tests (mocked LLM, no GPU) — run in CI
results/    # raw CSVs + summarized tables from real benchmark runs
```

Directories are scaffolded ahead of the code that will populate them, so the intended shape is
visible from the start; each fills in as its corresponding build phase lands.

## Setup

```bash
py -3.12 -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -e ".[dev]"     # core deps + lint/test tooling
pip install -e ".[gpu]"     # + vLLM/SGLang, only needed on a GPU (Colab/Kaggle) session
pytest tests/
```

No paid services are used anywhere in this project — all GPU experiments run on free-tier
Google Colab and Kaggle Notebooks.
