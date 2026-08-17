"""Benchmark interface for the EnTrance evaluation runner.

A benchmark is any object that can produce a list of :class:`Problem`
objects (a prompt plus ground truth) and score a single model output with
:meth:`Benchmark.verify`.  ``evaluate.py`` depends only on this interface,
so a new benchmark is added by subclassing :class:`Benchmark`, implementing
``load``/``verify`` and registering the class in ``benchmarks/__init__.py``.
"""

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
    """A single evaluation item.

    ``prompt`` is fed verbatim to the model.  ``answer`` holds the ground
    truth in whatever normalized form the benchmark's ``verify`` expects.
    ``raw`` carries benchmark-specific extras (e.g. unit tests for coding).
    """

    id: str
    prompt: str
    answer: str
    raw: dict[str, Any] = field(default_factory=dict)


class Benchmark(abc.ABC):
    """Minimal interface the runner depends on."""

    name: str = ""

    @abc.abstractmethod
    def load(self) -> list[Problem]:
        """Return the problems to evaluate (already prompt-formatted)."""

    @abc.abstractmethod
    def verify(self, problem: Problem, output: str) -> bool:
        """Return True if ``output`` is a correct answer for ``problem``."""
