from __future__ import annotations

from .base import Benchmark, Problem
from .aime2025 import AIME2025Benchmark
from .gpqa_diamond import GPQADiamondBenchmark
from .livecodebench import LiveCodeBenchBenchmark

__all__ = ["Benchmark", "Problem", "BENCHMARKS", "get_benchmark"]

BENCHMARKS: dict[str, type[Benchmark]] = {
    "aime2025": AIME2025Benchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
}

def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]()
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}"
        ) from None