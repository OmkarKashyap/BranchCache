#!/usr/bin/env bash
set -euo pipefail

MODEL="${BRANCHCACHE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${PORT:-8000}"

# Naive baseline: same vLLM binary as launch_vllm.sh, caching just turned off.
# That's the point — the only thing allowed to change between "naive" and
# "vllm" runs is this one flag.
vllm serve "$MODEL" \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
