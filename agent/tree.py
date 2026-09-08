import hashlib
from dataclasses import dataclass
from typing import Literal

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.state import RolloutNode

PrefixLength = Literal["short", "medium", "long"]


@dataclass
class Problem:
    problem_id: str
    description: str
    buggy_file_path: str
    buggy_file_content: str
    test_file_content: str
    repo_context: dict[str, str] | None = None  # extra files, used by the "long" condition


def build_prefix(problem: Problem, length: PrefixLength) -> list[dict]:
    """The frozen, shared-prefix message list for a given length condition.

    Must be byte-identical across calls for the same (problem, length) — see
    build_user_prompt's docstring. This is what caching keys on.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem, length)},
    ]


def build_independent_prefix(
    problem: Problem, branch_index: int, length: PrefixLength = "medium"
) -> list[dict]:
    """A prefix with a per-branch nonce so nothing overlaps across branches.

    Used for the no-shared-prefix negative control (Phase 5): prefix-aware
    serving should show ~no benefit here, since there is nothing to share.
    """
    nonce_system = f"{SYSTEM_PROMPT}\n\n[session-nonce: {problem.problem_id}-{branch_index}]"
    return [
        {"role": "system", "content": nonce_system},
        {"role": "user", "content": build_user_prompt(problem, length)},
    ]


def prefix_hash(messages: list[dict]) -> str:
    serialized = "".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha256(serialized.encode()).hexdigest()


def fork(prefix: list[dict], n: int) -> list[RolloutNode]:
    """Produce n independent branch roots sharing the same starting prefix."""
    return [
        RolloutNode(branch_id=i, messages=[dict(m) for m in prefix], depth=0)
        for i in range(n)
    ]
