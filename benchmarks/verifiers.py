from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# AIME (integer answers, 0-999)
# ---------------------------------------------------------------------------

def extract_aime_answer(output: str) -> int | None:
    text = output.strip()

    # 1) Explicit "Answer: N" / "final answer is N".
    matches = re.findall(
        r"(?:final\s+)?answer\s*(?:is|=|:)?\s*:?\s*(\d{1,4})\b",
        text,
        re.IGNORECASE,
    )
    if matches:
        return int(matches[-1])

    # 2) LaTeX \boxed{N}.
    matches = re.findall(r"\\boxed\s*\{\s*(\d{1,4})\s*\}", text)
    if matches:
        return int(matches[-1])

    # 3) Last integer in the response (AIME answers are <= 999).
    matches = re.findall(r"(?<![\d.])(\d{1,4})(?![\d.])", text)
    if matches:
        return int(matches[-1])

    return None


# ---------------------------------------------------------------------------
# GPQA (multiple choice, single letter A-D)
# ---------------------------------------------------------------------------

def extract_gpqa_answer(output: str) -> str | None:
    text = output.strip()

    # "(D)" / "( d )".
    matches = re.findall(r"\(\s*([A-D])\s*\)", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Standalone letter, prefer the last one (final answer is at the end).
    matches = re.findall(r"\b([A-Da-d])\b", text)
    if matches:
        return matches[-1].upper()

    return None


# ---------------------------------------------------------------------------
# LiveCodeBench (code extracted from the output, run against test cases)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_CODE_START = re.compile(
    r"^(?:import\s+\w|from\s+\w|def\s+\w|class\s+\w|if\s+__name__\s*==)"
)


def extract_code(output: str) -> str:
    """Pull a Python program out of a model response."""
    match = _FENCE.search(output)
    if match:
        return match.group(1).strip()

    text = output.strip()
    text = re.sub(r"^```[A-Za-z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If prose precedes the code, start at the first line that looks like code.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _CODE_START.match(line.strip()):
            return "\n".join(lines[i:]).strip()

    return text


def _normalize(text: str) -> str:
    # Match LiveCodeBench's output comparison: drop trailing whitespace on
    # each line and trailing blank lines.
    return "\n".join(line.rstrip() for line in text.rstrip("\n").splitlines()).strip()


def _run_stdin_test(script: str, test: dict, timeout: float) -> tuple[bool, str]:
    expected = _normalize(test.get("output", ""))
    try:
        proc = subprocess.run(
            [sys.executable, "-u", script],
            input=test.get("input", ""),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return False, f"exit {proc.returncode}: {'; '.join(tail)}"

    if _normalize(proc.stdout) != expected:
        return False, "output mismatch"
    return True, ""


def _run_functional_test(code: str, test: dict, timeout: float) -> tuple[bool, str]:
    del timeout  # functional tests are in-process; a runaway call is unlikely here
    namespace: dict = {}
    try:
        exec(compile(code, "<solution>", "exec"), namespace)
        got = eval(test.get("input", ""), namespace)
    except Exception as exc:  # noqa: BLE001 - we report the exception text
        return False, f"{type(exc).__name__}: {exc}"

    if str(got).strip() != str(test.get("output", "")).strip():
        return False, f"got {got!r}, want {test.get('output')!r}"
    return True, ""


def run_python_tests(
    code: str,
    tests: list[dict],
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Run ``code`` against LiveCodeBench-style test cases.

    ``tests`` entries look like ``{"input": ..., "output": ..., "testtype":
    "stdin" | "functional"}``.  Returns ``(passed, reason)``.
    """
    if not code.strip():
        return False, "empty solution"
    if not tests:
        return False, "no test cases"

    stdin_tests = [t for t in tests if t.get("testtype") != "functional"]
    functional_tests = [t for t in tests if t.get("testtype") == "functional"]

    script: str | None = None
    if stdin_tests:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code)
            script = handle.name

    try:
        for test in stdin_tests:
            ok, reason = _run_stdin_test(script, test, timeout)
            if not ok:
                return False, reason
        for test in functional_tests:
            ok, reason = _run_functional_test(code, test, timeout)
            if not ok:
                return False, reason
        return True, ""
    finally:
        if script is not None:
            Path(script).unlink(missing_ok=True)
