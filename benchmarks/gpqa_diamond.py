"""GPQA Diamond — graduate-level multiple-choice reasoning."""

from __future__ import annotations

import json

from .base import Benchmark, Problem, default_data_path
from .verifiers import extract_gpqa_answer

_GPQA_PROMPT = (
    "{question}\n\n"
    "Answer with only the letter of the correct option (A, B, C, or D)."
)


class GPQADiamondBenchmark(Benchmark):
    name = "gpqa_diamond"

    def __init__(self, data_file: str | None = None):
        self.data_file = data_file or str(default_data_path("gpqa_diamond.json"))

    def load(self) -> list[Problem]:
        rows = json.loads(open(self.data_file, encoding="utf-8").read())
        return [
            Problem(
                id=row["id"],
                prompt=_GPQA_PROMPT.format(question=row["question"]),
                answer=str(row["answer"]).strip().upper(),
                raw=row,
            )
            for row in rows
        ]

    def verify(self, problem: Problem, output: str) -> bool:
        return extract_gpqa_answer(output) == problem.answer
