import uuid
from dataclasses import dataclass, field


@dataclass
class RolloutNode:
    """A single branch's mutable state as it moves through the ReAct loop."""

    branch_id: int
    messages: list[dict] = field(default_factory=list)
    depth: int = 0
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    patch: str | None = None
    test_result: dict | None = None
    tool_time_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)

    def record_tool_call(self, name: str, args: dict, result: object, elapsed_ms: float) -> None:
        self.tool_time_ms += elapsed_ms
        self.tool_calls.append({"name": name, "args": args, "elapsed_ms": elapsed_ms})

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "branch_id": self.branch_id,
            "depth": self.depth,
            "messages": self.messages,
            "patch": self.patch,
            "test_result": self.test_result,
            "tool_time_ms": self.tool_time_ms,
            "tool_calls": self.tool_calls,
        }
