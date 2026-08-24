from __future__ import annotations

from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr

from ._verification import (
    extract_aime_answer,
    extract_code,
    extract_gpqa_answer,
    verify_livecodebench,
)


@scorer(metrics=[accuracy(), stderr()])
async def aime2025(state, target):
    """Score AIME 2025: extract integer from model output, compare with target."""
    extracted = extract_aime_answer(state.output.completion)
    if extracted is None:
        return Score(value=INCORRECT, explanation="Could not extract integer answer")
    correct = extracted == int(target.text)
    return Score(
        value=CORRECT if correct else INCORRECT,
        answer=str(extracted),
        explanation=f"Extracted={extracted}, Expected={target.text}",
    )


@scorer(metrics=[accuracy(), stderr()])
async def gpqa_diamond(state, target):
    """Score GPQA Diamond: extract letter from model output, compare with target."""
    extracted = extract_gpqa_answer(state.output.completion)
    if extracted is None:
        return Score(value=INCORRECT, explanation="Could not extract letter answer")
    correct = extracted == target.text
    return Score(
        value=CORRECT if correct else INCORRECT,
        answer=extracted,
        explanation=f"Extracted={extracted}, Expected={target.text}",
    )


@scorer(metrics=[accuracy(), stderr()])
def livecodebench(timeout: float = 10.0):
    """Score LiveCodeBench: extract code, run against hidden/public tests."""

    async def score(state, target):
        code = extract_code(state.output.completion)
        raw = state.metadata.get("raw", {}) if state.metadata else {}
        starter = (raw.get("starter_code") or "").strip()
        if starter:
            code = f"{starter}\n\n{code}"
        tests = raw.get("private_tests") or raw.get("public_tests") or []
        ok, reason = verify_livecodebench(code, tests, timeout=timeout)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=code[:200],
            explanation=reason if not ok else "All tests passed",
        )

    return score


SCORERS = {
    "aime2025": aime2025,
    "gpqa_diamond": gpqa_diamond,
    "livecodebench": livecodebench,
}