import argparse
import tempfile
from pathlib import Path

from agent.react_loop import LLMResponse, run_react_loop
from agent.tree import build_prefix, fork
from agent.verifier import verify
from eval.golden_dataset import GOLDEN_DATASET


def run_branch(problem, llm_client, max_steps=10, prefix=None):
    with tempfile.TemporaryDirectory(prefix=f"bc_{problem.problem_id}_") as tmp:
        workdir = Path(tmp)
        (workdir / problem.buggy_file_path).write_text(problem.buggy_file_content)
        (workdir / "test_solution.py").write_text(problem.test_file_content)
        if problem.repo_context:
            for name, content in problem.repo_context.items():
                (workdir / name).write_text(content)

        node = fork(prefix or build_prefix(problem, "medium"), 1)[0]
        node = run_react_loop(node, workdir, llm_client, max_steps=max_steps)
        if node.test_result is None:
            verify(node, workdir, "test_solution.py")
        return node


def pass_at_n(dataset, make_client, n, max_steps=10):
    solved = 0
    for instance in dataset:
        branches = [
            run_branch(instance.problem, make_client(instance), max_steps=max_steps)
            for _ in range(n)
        ]
        if any(b.test_result and b.test_result.get("passed") for b in branches):
            solved += 1
    return solved / len(dataset)


class ScriptedSolverClient:
    """Reads the buggy file, writes the known-good fix, runs the tests.

    This is a stand-in for a real model — the fix comes from
    ProblemInstance.reference_fix, which never gets put in the agent's own
    prompt, only used here to drive the harness before Phase 3 wires up
    naive/vLLM/SGLang clients.
    """

    def __init__(self, instance, give_up=False):
        self._instance = instance
        self._give_up = give_up
        self._step = 0

    def call(self, messages, tools):
        if self._give_up:
            return LLMResponse(content="couldn't find the bug")

        problem = self._instance.problem
        if self._step == 0:
            self._step += 1
            return LLMResponse(
                tool_call={"name": "read_file", "args": {"path": problem.buggy_file_path}}
            )
        if self._step == 1:
            self._step += 1
            return LLMResponse(
                tool_call={
                    "name": "apply_patch",
                    "args": {
                        "path": problem.buggy_file_path,
                        "new_content": self._instance.reference_fix,
                    },
                }
            )
        return LLMResponse(tool_call={"name": "run_tests", "args": {}})


def make_perfect_client_factory():
    return lambda instance: ScriptedSolverClient(instance)


def make_flaky_client_factory(succeed_from_attempt):
    """A branch only succeeds once it's been tried succeed_from_attempt times
    on the same problem — good enough to demonstrate pass@N > pass@1 without
    a real model in the loop yet.
    """
    attempts = {}

    def make_client(instance):
        attempts[instance.problem_id] = attempts.get(instance.problem_id, 0) + 1
        give_up = attempts[instance.problem_id] < succeed_from_attempt
        return ScriptedSolverClient(instance, give_up=give_up)

    return make_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    print("No real serving client wired in yet (that's Phase 3) - running the")
    print("scripted solver instead, just to exercise the harness end to end.\n")

    for n in args.ns:
        make_client = make_flaky_client_factory(succeed_from_attempt=3)
        score = pass_at_n(GOLDEN_DATASET, make_client, n, max_steps=args.max_steps)
        print(f"N={n}: pass@N = {score:.0%}")


if __name__ == "__main__":
    main()
