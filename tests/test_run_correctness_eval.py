from eval.golden_dataset import GOLDEN_DATASET
from eval.run_correctness_eval import (
    ScriptedSolverClient,
    make_flaky_client_factory,
    pass_at_n,
    run_branch,
)


def test_scripted_solver_fixes_every_problem_in_the_dataset():
    for instance in GOLDEN_DATASET:
        node = run_branch(instance.problem, ScriptedSolverClient(instance))
        assert node.test_result["passed"] is True, instance.problem_id


def test_give_up_client_never_passes():
    instance = GOLDEN_DATASET[0]
    node = run_branch(instance.problem, ScriptedSolverClient(instance, give_up=True))
    assert node.test_result["passed"] is False


def test_pass_at_n_shows_headroom_over_pass_at_1():
    dataset = [GOLDEN_DATASET[0]]

    p1 = pass_at_n(dataset, make_flaky_client_factory(succeed_from_attempt=3), n=1)
    p4 = pass_at_n(dataset, make_flaky_client_factory(succeed_from_attempt=3), n=4)

    assert p1 == 0.0
    assert p4 == 1.0
