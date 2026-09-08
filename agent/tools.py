import subprocess
import sys
import time
from pathlib import Path


class ToolError(Exception):
    """Raised when a tool cannot complete (bad path, missing file, etc.)."""


def _resolve(workdir: Path, rel_path: str) -> Path:
    resolved = (workdir / rel_path).resolve()
    workdir_resolved = workdir.resolve()
    if resolved != workdir_resolved and workdir_resolved not in resolved.parents:
        raise ToolError(f"path escapes workdir: {rel_path}")
    return resolved


def list_files(workdir: Path) -> list[str]:
    workdir = Path(workdir)
    return sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*") if p.is_file())


def read_file(workdir: Path, path: str) -> str:
    target = _resolve(Path(workdir), path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    return target.read_text()


def apply_patch(workdir: Path, path: str, new_content: str) -> dict:
    target = _resolve(Path(workdir), path)
    if not target.exists():
        raise ToolError(f"no such file: {path}")
    target.write_text(new_content)
    return {"path": path, "bytes_written": len(new_content)}


def run_tests(workdir: Path, test_path: str = "test_solution.py") -> dict:
    workdir = Path(workdir)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the working directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Overwrite a file with new contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["path", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the pytest test suite and report pass/fail.",
            "parameters": {
                "type": "object",
                "properties": {"test_path": {"type": "string"}},
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "apply_patch": apply_patch,
    "run_tests": run_tests,
}


def dispatch_tool(node, workdir: Path, name: str, args: dict) -> dict:
    """Run a tool by name, timing it and recording the call onto `node`."""
    start = time.perf_counter()

    if name not in TOOL_FUNCTIONS:
        elapsed_ms = (time.perf_counter() - start) * 1000
        node.record_tool_call(name, args, None, elapsed_ms)
        return {"error": f"unknown tool: {name}"}

    try:
        result = TOOL_FUNCTIONS[name](workdir, **args)
    except ToolError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        node.record_tool_call(name, args, None, elapsed_ms)
        return {"error": str(e)}

    elapsed_ms = (time.perf_counter() - start) * 1000
    node.record_tool_call(name, args, result, elapsed_ms)
    return result
