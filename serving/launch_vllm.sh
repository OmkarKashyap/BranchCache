#!/usr/bin/env bash
set -euo pipefail

MODEL="${BRANCHCACHE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${PORT:-8001}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
