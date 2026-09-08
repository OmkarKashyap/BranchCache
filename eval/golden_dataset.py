from dataclasses import dataclass

from agent.tree import Problem


@dataclass
class ProblemInstance:
    problem_id: str
    difficulty: str
    problem: Problem
    reference_fix: str


def _instance(problem_id, difficulty, description, buggy, fix, test):
    return ProblemInstance(
        problem_id=problem_id,
        difficulty=difficulty,
        reference_fix=fix,
        problem=Problem(
            problem_id=problem_id,
            description=description,
            buggy_file_path="solution.py",
            buggy_file_content=buggy,
            test_file_content=test,
        ),
    )


# Starter set, hand-written so the harness has something real to run against.
# The full 30-50 problem set (HumanEvalFix-derived) gets pulled in by
# scripts/setup_humanevalfix_subset.py in a later phase.
GOLDEN_DATASET = [
    _instance(
        "off_by_one_range",
        "easy",
        "count_up_to(n) should return integers from 0 up to and including n, but the "
        "test says it's coming up one short.",
        "def count_up_to(n):\n    return list(range(n))\n",
        "def count_up_to(n):\n    return list(range(n + 1))\n",
        "from solution import count_up_to\n\n\n"
        "def test_count_up_to():\n    assert count_up_to(5) == [0, 1, 2, 3, 4, 5]\n",
    ),
    _instance(
        "threshold_check",
        "easy",
        "is_adult(age) should treat 18 as an adult, but 18-year-olds are failing the check.",
        "def is_adult(age):\n    return age > 18\n",
        "def is_adult(age):\n    return age >= 18\n",
        "from solution import is_adult\n\n\n"
        "def test_is_adult():\n    assert is_adult(18) is True\n    assert is_adult(17) is False\n",
    ),
    _instance(
        "mutable_default_arg",
        "medium",
        "add_item(item) is supposed to start fresh every call, but items are leaking "
        "between unrelated calls.",
        "def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n",
        "def add_item(item, bucket=None):\n"
        "    if bucket is None:\n"
        "        bucket = []\n"
        "    bucket.append(item)\n"
        "    return bucket\n",
        "from solution import add_item\n\n\n"
        "def test_add_item_does_not_leak_between_calls():\n"
        '    first = add_item("a")\n'
        '    second = add_item("b")\n'
        '    assert first == ["a"]\n'
        '    assert second == ["b"]\n',
    ),
    _instance(
        "safe_lookup",
        "easy",
        "get_score(scores, name) should return 0 for a name that isn't in the dict, "
        "but it's throwing a KeyError instead.",
        "def get_score(scores, name):\n    return scores[name]\n",
        "def get_score(scores, name):\n    return scores.get(name, 0)\n",
        "from solution import get_score\n\n\n"
        "def test_missing_name_returns_zero():\n"
        '    assert get_score({"amy": 10}, "brian") == 0\n\n\n'
        "def test_existing_name_returns_score():\n"
        '    assert get_score({"amy": 10}, "amy") == 10\n',
    ),
    _instance(
        "reverse_words",
        "medium",
        "reverse_words(sentence) is meant to reverse word order, not spell every word backwards.",
        "def reverse_words(sentence):\n    return sentence[::-1]\n",
        'def reverse_words(sentence):\n    return " ".join(sentence.split()[::-1])\n',
        "from solution import reverse_words\n\n\n"
        "def test_reverse_words():\n"
        '    assert reverse_words("the quick fox") == "fox quick the"\n',
    ),
    _instance(
        "running_total",
        "medium",
        "running_totals(nums) should return a cumulative sum at each position, but "
        "everything after the first entry is wrong.",
        "def running_totals(nums):\n"
        "    totals = []\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total = n\n"
        "        totals.append(total)\n"
        "    return totals\n",
        "def running_totals(nums):\n"
        "    totals = []\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total += n\n"
        "        totals.append(total)\n"
        "    return totals\n",
        "from solution import running_totals\n\n\n"
        "def test_running_totals():\n"
        "    assert running_totals([1, 2, 3, 4]) == [1, 3, 6, 10]\n",
    ),
    _instance(
        "fib_base_case",
        "hard",
        "fib(n) should return the nth Fibonacci number with fib(0) = 0 and fib(1) = 1, "
        "but the small cases are off.",
        "def fib(n):\n    if n <= 1:\n        return 1\n    return fib(n - 1) + fib(n - 2)\n",
        "def fib(n):\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    if n == 1:\n"
        "        return 1\n"
        "    return fib(n - 1) + fib(n - 2)\n",
        "from solution import fib\n\n\n"
        "def test_fib_base_cases():\n"
        "    assert fib(0) == 0\n"
        "    assert fib(1) == 1\n\n\n"
        "def test_fib_general():\n"
        "    assert fib(6) == 8\n",
    ),
    _instance(
        "sorted_by_score",
        "medium",
        "rank_players(players) is supposed to rank by score highest first, but the "
        "order comes out backwards.",
        'def rank_players(players):\n    return sorted(players, key=lambda p: p["score"])\n',
        "def rank_players(players):\n"
        '    return sorted(players, key=lambda p: p["score"], reverse=True)\n',
        "from solution import rank_players\n\n\n"
        "def test_rank_players_descending():\n"
        "    players = [\n"
        '        {"name": "a", "score": 3},\n'
        '        {"name": "b", "score": 9},\n'
        '        {"name": "c", "score": 1},\n'
        "    ]\n"
        "    ranked = rank_players(players)\n"
        '    assert [p["name"] for p in ranked] == ["b", "a", "c"]\n',
    ),
]


def get_problem(problem_id):
    for instance in GOLDEN_DATASET:
        if instance.problem_id == problem_id:
            return instance
    raise KeyError(problem_id)
