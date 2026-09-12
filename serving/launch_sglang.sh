#!/usr/bin/env bash
set -euo pipefail

MODEL="${BRANCHCACHE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${PORT:-8002}"

# RadixAttention is on by default in SGLang, no flag needed — the naive
# comparison point for this project is the vLLM no-caching server, not a
# separately-disabled SGLang mode.
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT"
