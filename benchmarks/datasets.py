from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Prompt templates (unchanged from the original benchmarks)
# ---------------------------------------------------------------------------

_AIME_PROMPT = (
    "{question}\n\n"
    "Solve this problem. Show your step-by-step reasoning, then state your "
    "final answer on the last line in the form `Answer: <integer>` "
    "(a single integer from 0 to 999)."
)

_GPQA_PROMPT = (
    "{question}\n\n"
    "Answer with only the letter of the correct option (A, B, C, or D)."
)

_LCB_PROMPT = (
    "Solve the following competitive programming problem in Python 3.\n\n"
    "{question}\n\n"
    "{starter}"
    "Output only the complete, self-contained Python solution. Read from "
    "standard input and write to standard output. Wrap the code in a single "
    "```python code block. Do not include any explanation."
)


def _load_json(filename: str) -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def aime2025(limit: int | None = None) -> MemoryDataset:
    """AIME 2025 — 30 competition-math problems (I + II)."""
    rows = _load_json("aime2025.json")
    if limit is not None:
        rows = rows[:limit]
    return MemoryDataset([
        Sample(
            id=row["id"],
            input=_AIME_PROMPT.format(question=row["question"]),
            target=str(row["answer"]).strip(),
            metadata={"raw": row},
        )
        for row in rows
    ])


def gpqa_diamond(limit: int | None = None) -> MemoryDataset:
    """GPQA Diamond — 198 graduate-level multiple-choice science questions."""
    rows = _load_json("gpqa_diamond.json")
    if limit is not None:
        rows = rows[:limit]
    return MemoryDataset([
        Sample(
            id=row["id"],
            input=_GPQA_PROMPT.format(question=row["question"]),
            target=str(row["answer"]).strip().upper(),
            metadata={"raw": row},
        )
        for row in rows
    ])


def livecodebench(limit: int | None = None) -> MemoryDataset:
    """LiveCodeBench — competitive-programming problems with hidden tests."""
    rows = _load_json("livecodebench.json")
    if limit is not None:
        rows = rows[:limit]
    samples: list[Sample] = []
    for row in rows:
        starter = (row.get("starter_code") or "").strip()
        starter_block = (
            f"The following starter code is provided:\n```python\n{starter}\n```\n\n"
            if starter
            else ""
        )
        samples.append(
            Sample(
                id=row["id"],
                input=_LCB_PROMPT.format(
                    question=row["question_content"], starter=starter_block
                ),
                target="",  # correctness via test execution, not string match
                metadata={"raw": row},
            )
        )
    return MemoryDataset(samples)


# Map benchmark name → loader function.
DATASETS: dict[str, object] = {
    "aime2025": aime2025,
    "gpqa_diamond": gpqa_diamond,
    "livecodebench": livecodebench,
}