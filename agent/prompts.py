SYSTEM_PROMPT = """You are a careful software engineer fixing a bug in a small Python project.
You have access to tools: list_files, read_file, apply_patch, run_tests.
Investigate the failing test, find the bug, apply a fix with apply_patch, then confirm with \
run_tests. Once run_tests passes, stop."""

_TOOLS_BLOCK = """Available tools:
- list_files(): list all files in the working directory
- read_file(path): read a file's contents
- apply_patch(path, new_content): overwrite a file with new contents
- run_tests(test_path): run the test suite and report pass/fail"""


def build_user_prompt(problem, length: str) -> str:
    """Assemble the user-turn prompt for a given prefix-length condition.

    Must be a pure function of (problem, length) — no timestamps, random ids, or
    unordered dict iteration — so the same inputs always produce byte-identical
    output, which prefix-aware serving depends on.
    """
    parts = [f"Problem: {problem.description}"]

    if length == "short":
        return "\n\n".join(parts)

    parts.append(
        f"Buggy file ({problem.buggy_file_path}):\n```python\n{problem.buggy_file_content}\n```"
    )

    if length == "medium":
        return "\n\n".join(parts)

    if length != "long":
        raise ValueError(f"unknown prefix length: {length}")

    if problem.repo_context:
        for path in sorted(problem.repo_context):
            parts.append(f"Additional file ({path}):\n```python\n{problem.repo_context[path]}\n```")
    parts.append(_TOOLS_BLOCK)
    return "\n\n".join(parts)
