from agent.react_loop import LLMResponse, run_react_loop
from agent.state import RolloutNode


class FakeLLMClient:
    """Returns scripted responses in order — no real model involved."""

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls = 0

    def call(self, messages, tools):
        self.calls += 1
        return self._responses.pop(0)


def test_loop_reads_patches_and_verifies_success(sandbox):
    node = RolloutNode(branch_id=0, messages=[{"role": "system", "content": "sys"}])
    client = FakeLLMClient(
        [
            LLMResponse(tool_call={"name": "read_file", "args": {"path": "solution.py"}}),
            LLMResponse(
                tool_call={
                    "name": "apply_patch",
                    "args": {
                        "path": "solution.py",
                        "new_content": "def add(a, b):\n    return a + b\n",
                    },
                }
            ),
            LLMResponse(tool_call={"name": "run_tests", "args": {"test_path": "test_solution.py"}}),
        ]
    )

    result = run_react_loop(node, sandbox, client, max_steps=10)

    assert result.test_result is not None
    assert result.test_result["passed"] is True
    assert client.calls == 3


def test_loop_terminates_gracefully_without_success(sandbox):
    node = RolloutNode(branch_id=0, messages=[{"role": "system", "content": "sys"}])
    client = FakeLLMClient(
        [
            LLMResponse(tool_call={"name": "read_file", "args": {"path": "solution.py"}}),
            LLMResponse(tool_call={"name": "read_file", "args": {"path": "solution.py"}}),
        ]
    )

    result = run_react_loop(node, sandbox, client, max_steps=2)

    assert result.test_result is None
    assert result.depth == 2


def test_loop_stops_on_final_non_tool_answer(sandbox):
    node = RolloutNode(branch_id=0, messages=[{"role": "system", "content": "sys"}])
    client = FakeLLMClient([LLMResponse(content="I give up.")])

    result = run_react_loop(node, sandbox, client, max_steps=5)

    assert client.calls == 1
    assert result.messages[-1]["content"] == "I give up."


def test_loop_gives_tool_call_and_result_matching_ids(sandbox):
    node = RolloutNode(branch_id=0, messages=[{"role": "system", "content": "sys"}])
    client = FakeLLMClient(
        [
            LLMResponse(tool_call={"name": "list_files", "args": {}}),
            LLMResponse(content="done"),
        ]
    )

    result = run_react_loop(node, sandbox, client, max_steps=5)

    call_msg, tool_msg = result.messages[1], result.messages[2]
    assert call_msg["tool_call"]["id"] == tool_msg["tool_call_id"]


def test_loop_records_tool_time_on_node(sandbox):
    node = RolloutNode(branch_id=0, messages=[{"role": "system", "content": "sys"}])
    client = FakeLLMClient(
        [
            LLMResponse(tool_call={"name": "list_files", "args": {}}),
            LLMResponse(content="giving up"),
        ]
    )

    result = run_react_loop(node, sandbox, client, max_steps=5)

    assert result.tool_time_ms > 0
    assert len(result.tool_calls) == 1
