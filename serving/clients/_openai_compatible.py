import json
import time

from openai import OpenAI

from agent.react_loop import LLMResponse


def to_wire_messages(messages):
    wire = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_call"):
            tc = m["tool_call"]
            wire.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                    ],
                }
            )
        elif role == "tool":
            wire.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
            )
        else:
            wire.append({"role": role, "content": m["content"]})
    return wire


def parse_response_message(message):
    if message.tool_calls:
        tc = message.tool_calls[0]
        return LLMResponse(
            tool_call={
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments or "{}"),
            }
        )
    return LLMResponse(content=message.content)


class OpenAICompatibleClient:
    """Talks to whatever's behind base_url using the OpenAI chat-completions
    wire format — vLLM and SGLang both speak this, so naive/vllm/sglang
    clients are really just this class pointed at different servers.
    """

    def __init__(
        self,
        base_url,
        model,
        strategy,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=512,
        seed=0,
    ):
        self.strategy = strategy
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.last_call_metrics = None
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def call(self, messages, tools):
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=to_wire_messages(messages),
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage", None)
        self.last_call_metrics = {
            "strategy": self.strategy,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

        return parse_response_message(response.choices[0].message)
