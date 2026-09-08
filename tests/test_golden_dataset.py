import pytest

from agent.tree import build_prefix
from eval.golden_dataset import GOLDEN_DATASET, get_problem


def test_all_problem_ids_are_unique():
    ids = [instance.problem_id for instance in GOLDEN_DATASET]
    assert len(ids) == len(set(ids))


def test_get_problem_roundtrips():
    instance = GOLDEN_DATASET[0]
    assert get_problem(instance.problem_id) is instance


def test_get_problem_missing_id_raises():
    with pytest.raises(KeyError):
        get_problem("does-not-exist")


def test_every_problem_builds_a_medium_prefix():
    for instance in GOLDEN_DATASET:
        prefix = build_prefix(instance.problem, "medium")
        assert len(prefix) == 2
        assert instance.problem.buggy_file_content in prefix[1]["content"]
