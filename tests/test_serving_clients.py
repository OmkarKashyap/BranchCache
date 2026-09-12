import json

from serving.clients._openai_compatible import parse_response_message, to_wire_messages


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def test_to_wire_messages_passes_plain_turns_through():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert to_wire_messages(messages) == messages


def test_to_wire_messages_translates_tool_call_and_result():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_call": {"id": "call_1", "name": "read_file", "args": {"path": "solution.py"}},
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
    ]

    wire = to_wire_messages(messages)

    call = wire[0]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "solution.py"}
    assert wire[1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_parse_response_message_with_tool_call():
    message = _FakeMessage(tool_calls=[_FakeToolCall("call_9", "run_tests", "{}")])
    response = parse_response_message(message)
    assert response.is_tool_call
    assert response.tool_call == {"id": "call_9", "name": "run_tests", "args": {}}


def test_parse_response_message_final_answer():
    message = _FakeMessage(content="all done")
    response = parse_response_message(message)
    assert not response.is_tool_call
    assert response.content == "all done"


def test_naive_vllm_sglang_clients_carry_distinct_strategy_labels():
    from serving.clients import naive_client, sglang_client, vllm_client

    naive = naive_client.make_client(base_url="http://x/v1", model="m")
    vllm = vllm_client.make_client(base_url="http://x/v1", model="m")
    sglang = sglang_client.make_client(base_url="http://x/v1", model="m")

    assert {naive.strategy, vllm.strategy, sglang.strategy} == {"naive", "vllm", "sglang"}
