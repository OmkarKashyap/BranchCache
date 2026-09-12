import os

from serving.clients._openai_compatible import OpenAICompatibleClient

# Same vLLM binary as vllm_client, just started with --no-enable-prefix-caching
# (see serving/launch_naive.sh) — this is what isolates caching as the one
# variable that changes between "naive" and "vllm".
DEFAULT_BASE_URL = os.environ.get("BRANCHCACHE_NAIVE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("BRANCHCACHE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")


def make_client(base_url=None, model=None, **kwargs):
    return OpenAICompatibleClient(
        base_url=base_url or DEFAULT_BASE_URL,
        model=model or DEFAULT_MODEL,
        strategy="naive",
        **kwargs,
    )
