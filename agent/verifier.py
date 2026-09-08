from pathlib import Path

from agent.state import RolloutNode
from agent.tools import run_tests


def verify(node: RolloutNode, workdir: Path, test_path: str = "test_solution.py") -> dict:
    """Authoritative pass/fail check, independent of whatever the loop itself
    observed — a branch that ran out of steps without calling run_tests
    successfully still gets checked here.
    """
    result = run_tests(workdir, test_path)
    node.test_result = result
    return result


def select_winner(branches: list[RolloutNode]) -> RolloutNode | None:
    """Among passing branches, pick the cheapest one (by tool time). Returns
    None if no branch passed — callers should report that honestly rather
    than picking a failing branch as a fallback.
    """
    passing = [b for b in branches if b.test_result and b.test_result.get("passed")]
    if not passing:
        return None
    return min(passing, key=lambda b: b.tool_time_ms)
