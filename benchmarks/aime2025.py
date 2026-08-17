"""AIME 2025 (I + II) — 30 problems with integer answers."""

from __future__ import annotations

import json

from .base import Benchmark, Problem, default_data_path
from .verifiers import extract_aime_answer

_AIME_PROMPT = (
    "{question}\n\n"
    "Solve this problem. Show your step-by-step reasoning, then state your "
    "final answer on the last line in the form `Answer: <integer>` "
    "(a single integer from 0 to 999)."
)


class AIME2025Benchmark(Benchmark):
    name = "aime2025"

    def __init__(self, data_file: str | None = None):
        self.data_file = data_file or str(default_data_path("aime2025.json"))

    def load(self) -> list[Problem]:
        rows = json.loads(open(self.data_file, encoding="utf-8").read())
        return [
            Problem(
                id=row["id"],
                prompt=_AIME_PROMPT.format(question=row["question"]),
                answer=str(row["answer"]).strip(),
                raw=row,
            )
            for row in rows
        ]

    def verify(self, problem: Problem, output: str) -> bool:
        return extract_aime_answer(output) == int(problem.answer)
