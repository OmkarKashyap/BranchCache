from agent.state import RolloutNode
from agent.tools import apply_patch
from agent.verifier import select_winner, verify


def make_node(branch_id, tool_time_ms, passed):
    node = RolloutNode(branch_id=branch_id)
    node.tool_time_ms = tool_time_ms
    node.test_result = {"passed": passed}
    return node


def test_verify_runs_tests_and_updates_node(sandbox):
    apply_patch(sandbox, "solution.py", "def add(a, b):\n    return a + b\n")
    node = RolloutNode(branch_id=0)

    result = verify(node, sandbox, "test_solution.py")

    assert result["passed"] is True
    assert node.test_result["passed"] is True


def test_verify_reports_failure_honestly(sandbox):
    node = RolloutNode(branch_id=0)
    result = verify(node, sandbox, "test_solution.py")
    assert result["passed"] is False


def test_select_winner_picks_cheapest_passing_branch():
    branches = [
        make_node(0, tool_time_ms=50.0, passed=True),
        make_node(1, tool_time_ms=10.0, passed=True),
        make_node(2, tool_time_ms=5.0, passed=False),
    ]
    winner = select_winner(branches)
    assert winner.branch_id == 1


def test_select_winner_returns_none_if_nothing_passed():
    branches = [make_node(0, 10.0, False), make_node(1, 20.0, False)]
    assert select_winner(branches) is None
