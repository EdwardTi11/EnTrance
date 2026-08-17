from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Where the bundled, normalized benchmark data lives.
DATA_DIR = Path(__file__).resolve().parent / "data"

def default_data_path(filename: str) -> Path:
    return DATA_DIR / filename

@dataclass
class Problem:
    id: str
    prompt: str
    answer: str
    raw: dict[str, Any] = field(default_factory=dict)

class Benchmark(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def load(self) -> list[Problem]:
        """Return the problems to evaluate (already prompt-formatted)."""

    @abc.abstractmethod
    def verify(self, problem: Problem, output: str) -> bool:
        """Return True if ``output`` is a correct answer for ``problem``."""
