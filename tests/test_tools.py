import pytest

from agent.tools import ToolError, apply_patch, list_files, read_file, run_tests


def test_list_files(sandbox):
    files = list_files(sandbox)
    assert "solution.py" in files
    assert "test_solution.py" in files


def test_read_file(sandbox):
    assert "def add" in read_file(sandbox, "solution.py")


def test_read_file_missing_raises(sandbox):
    with pytest.raises(ToolError):
        read_file(sandbox, "nope.py")


def test_read_file_path_escape_blocked(sandbox):
    with pytest.raises(ToolError):
        read_file(sandbox, "../outside.py")


def test_apply_patch_writes_file(sandbox):
    apply_patch(sandbox, "solution.py", "def add(a, b):\n    return a + b\n")
    assert "a + b" in read_file(sandbox, "solution.py")


def test_apply_patch_missing_file_raises(sandbox):
    with pytest.raises(ToolError):
        apply_patch(sandbox, "nope.py", "x = 1\n")


def test_run_tests_fails_on_buggy_solution(sandbox):
    result = run_tests(sandbox, "test_solution.py")
    assert result["passed"] is False


def test_run_tests_passes_after_fix(sandbox):
    apply_patch(sandbox, "solution.py", "def add(a, b):\n    return a + b\n")
    result = run_tests(sandbox, "test_solution.py")
    assert result["passed"] is True
