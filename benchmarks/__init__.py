from __future__ import annotations

from ._verification import (
    extract_aime_answer,
    extract_code,
    extract_gpqa_answer,
    verify_livecodebench,
)
from .datasets import DATASETS, aime2025, gpqa_diamond, livecodebench
from .scorers import SCORERS, aime2025 as aime2025_scorer
from .scorers import gpqa_diamond as gpqa_diamond_scorer
from .scorers import livecodebench as livecodebench_scorer

__all__ = [
    # Datasets
    "DATASETS",
    "aime2025",
    "gpqa_diamond",
    "livecodebench",
    # Scorers
    "SCORERS",
    "aime2025_scorer",
    "gpqa_diamond_scorer",
    "livecodebench_scorer",
    # Verification
    "extract_aime_answer",
    "extract_gpqa_answer",
    "extract_code",
    "verify_livecodebench",
]

BENCHMARKS: list[str] = list(DATASETS)


def get_dataset(name: str, limit: int | None = None):
    """Return an inspect_ai MemoryDataset for *name*."""
    try:
        loader = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(DATASETS)}"
        ) from None
    return loader(limit=limit)