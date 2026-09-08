from pathlib import Path
from typing import Protocol

from agent.state import RolloutNode
from agent.tools import TOOL_SCHEMAS, dispatch_tool


class LLMResponse:
    """A minimal, engine-agnostic response shape.

    Real serving clients (Phase 3) translate whatever their backend returns
    into this shape, so react_loop never needs to know which engine is behind
    it — that's the seam that keeps prefix-cache-sensitive prompt construction
    fully under our control.
    """

    def __init__(self, content: str | None = None, tool_call: dict | None = None):
        self.content = content
        self.tool_call = tool_call  # {"name": str, "args": dict}

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None


class LLMClient(Protocol):
    def call(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def run_react_loop(
    node: RolloutNode,
    workdir: Path,
    llm_client: LLMClient,
    max_steps: int = 10,
) -> RolloutNode:
    """Drive one branch through read/patch/test tool calls until it either
    passes its tests, gives a final non-tool answer, or hits max_steps.
    """
    for _ in range(max_steps):
        response = llm_client.call(node.messages, TOOL_SCHEMAS)
        node.depth += 1

        if not response.is_tool_call:
            node.messages.append({"role": "assistant", "content": response.content})
            break

        name = response.tool_call["name"]
        args = response.tool_call.get("args", {})
        node.messages.append(
            {"role": "assistant", "content": None, "tool_call": {"name": name, "args": args}}
        )

        result = dispatch_tool(node, workdir, name, args)
        node.messages.append({"role": "tool", "name": name, "content": str(result)})

        if name == "run_tests" and isinstance(result, dict) and result.get("passed"):
            node.test_result = result
            break

    return node
