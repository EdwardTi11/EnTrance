"""LiveCodeBench — competitive-programming code generation.

Verified by executing the model's code against the official public and
private test cases shipped with the dataset.
"""

from __future__ import annotations

import json

from .base import Benchmark, Problem, default_data_path
from .verifiers import extract_code, run_python_tests

_LCB_PROMPT = (
    "Solve the following competitive programming problem in Python 3.\n\n"
    "{question}\n\n"
    "{starter}"
    "Output only the complete, self-contained Python solution. Read from "
    "standard input and write to standard output. Wrap the code in a single "
    "```python code block. Do not include any explanation."
)


class LiveCodeBenchBenchmark(Benchmark):
    name = "livecodebench"

    def __init__(self, data_file: str | None = None, timeout: float = 10.0):
        self.data_file = data_file or str(default_data_path("livecodebench.json"))
        self.timeout = timeout

    def load(self) -> list[Problem]:
        rows = json.loads(open(self.data_file, encoding="utf-8").read())
        problems = []
        for row in rows:
            starter = (row.get("starter_code") or "").strip()
            starter_block = (
                f"The following starter code is provided:\n```python\n{starter}\n```\n\n"
                if starter
                else ""
            )
            problems.append(
                Problem(
                    id=row["id"],
                    prompt=_LCB_PROMPT.format(
                        question=row["question_content"], starter=starter_block
                    ),
                    answer="",
                    raw=row,
                )
            )
        return problems

    def verify(self, problem: Problem, output: str) -> bool:
        code = extract_code(output)
        starter = (problem.raw.get("starter_code") or "").strip()
        if starter:
            code = f"{starter}\n\n{code}"

        # Official hidden tests first; fall back to public tests if the
        # private tests are unavailable.
        tests = problem.raw.get("private_tests") or problem.raw.get("public_tests") or []
        ok, _reason = run_python_tests(code, tests, timeout=self.timeout)
        return ok
