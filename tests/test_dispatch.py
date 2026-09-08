from agent.state import RolloutNode
from agent.tools import dispatch_tool


def test_dispatch_tool_records_timing(sandbox):
    node = RolloutNode(branch_id=0)
    dispatch_tool(node, sandbox, "list_files", {})
    assert node.tool_time_ms > 0
    assert len(node.tool_calls) == 1
    assert node.tool_calls[0]["name"] == "list_files"


def test_dispatch_tool_unknown_tool_returns_error_not_raise(sandbox):
    node = RolloutNode(branch_id=0)
    result = dispatch_tool(node, sandbox, "not_a_tool", {})
    assert "error" in result
    assert node.tool_calls[0]["name"] == "not_a_tool"


def test_dispatch_tool_bad_args_returns_error(sandbox):
    node = RolloutNode(branch_id=0)
    result = dispatch_tool(node, sandbox, "read_file", {"path": "nope.py"})
    assert "error" in result
