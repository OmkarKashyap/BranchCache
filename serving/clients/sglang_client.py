import os

from serving.clients._openai_compatible import OpenAICompatibleClient

DEFAULT_BASE_URL = os.environ.get("BRANCHCACHE_SGLANG_URL", "http://localhost:8002/v1")
DEFAULT_MODEL = os.environ.get("BRANCHCACHE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")


def make_client(base_url=None, model=None, **kwargs):
    return OpenAICompatibleClient(
        base_url=base_url or DEFAULT_BASE_URL,
        model=model or DEFAULT_MODEL,
        strategy="sglang",
        **kwargs,
    )
