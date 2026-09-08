import pytest


@pytest.fixture
def sandbox(tmp_path):
    """A minimal problem workdir: an add() with a one-character bug, plus its test."""
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return tmp_path
