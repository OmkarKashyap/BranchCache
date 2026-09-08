from agent.tree import Problem, build_independent_prefix, build_prefix, fork, prefix_hash


def make_problem():
    return Problem(
        problem_id="p1",
        description="Fix the add function.",
        buggy_file_path="solution.py",
        buggy_file_content="def add(a, b):\n    return a - b\n",
        test_file_content="def test_add():\n    assert add(2, 3) == 5\n",
    )


def test_build_prefix_is_byte_identical_across_calls():
    problem = make_problem()
    a = build_prefix(problem, "medium")
    b = build_prefix(problem, "medium")
    assert a == b
    assert prefix_hash(a) == prefix_hash(b)


def test_build_prefix_length_variants_differ():
    problem = make_problem()
    lengths = ("short", "medium", "long")
    hashes = {length: prefix_hash(build_prefix(problem, length)) for length in lengths}
    assert len(set(hashes.values())) == 3


def test_build_prefix_long_includes_repo_context():
    problem = make_problem()
    problem.repo_context = {"utils.py": "def helper(): pass\n"}
    prefix = build_prefix(problem, "long")
    assert "utils.py" in prefix[1]["content"]


def test_fork_produces_n_independent_branches():
    problem = make_problem()
    prefix = build_prefix(problem, "medium")
    branches = fork(prefix, 4)

    assert len(branches) == 4
    assert len({b.node_id for b in branches}) == 4

    branches[0].messages.append({"role": "user", "content": "mutated"})
    assert len(branches[1].messages) == len(prefix)


def test_independent_prefix_differs_per_branch():
    problem = make_problem()
    p0 = build_independent_prefix(problem, 0)
    p1 = build_independent_prefix(problem, 1)
    assert prefix_hash(p0) != prefix_hash(p1)
