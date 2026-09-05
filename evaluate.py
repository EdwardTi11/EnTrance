import argparse
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Generate, TaskState, solver
from inspect_evals.aime2025 import aime2025
from inspect_evals.bigcodebench import bigcodebench
from inspect_evals.gpqa import gpqa_diamond
from llama_cpp import Llama

from model_design.engine import generate_text
from model_design.adaptive_control import DecoderController

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
MODES = ("baseline", "entranced")
DEFAULT_LIMITS = {"aime2025": 30, "gpqa_diamond": 50, "bigcodebench": 40}

@solver
def entrance_generation(
    model_instance: Llama,
    seed: int,
    gen_config: dict[str, Any],
    decoder_controller=None,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_instance.reset()
        text, _ = generate_text(
            model=model_instance,
            prompt=state.user_prompt.text,
            seed=seed,
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            decoder_controller=decoder_controller,
        )
        state.output.completion = text
        return state
    return solve

def inspect_task(name: str, solver_instance, limit: int) -> Task:
    if name == "aime2025":
        task = aime2025()
    elif name == "gpqa_diamond":
        task = gpqa_diamond()
    elif name == "bigcodebench":
        # Keep the official scorer; it requires Docker at runtime.
        task = bigcodebench()
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    if limit and len(task.dataset) > limit:
        task.dataset = MemoryDataset(list(task.dataset)[:limit])
    task.solver = solver_instance
    return task

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EnTrance through Inspect AI.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_LIMITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aime-limit", type=int, default=DEFAULT_LIMITS["aime2025"])
    parser.add_argument("--gpqa-limit", type=int, default=DEFAULT_LIMITS["gpqa_diamond"])
    parser.add_argument("--bcb-limit", type=int, default=DEFAULT_LIMITS["bigcodebench"])

    args = parser.parse_args(argv)

    limits = {
        "aime2025": args.aime_limit,
        "gpqa_diamond": args.gpqa_limit,
        "bigcodebench": args.bcb_limit,
    }
    if args.limit is not None:
        limits = {name: args.limit for name in limits}

    selected = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    model = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False,
        logits_all=True,
    )
    gen_config = {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
    }

    for name in selected:
        logs_by_mode = {}
        for mode in MODES:
            controller = DecoderController() if mode == "entranced" else None
            solver_comp = entrance_generation(
                model_instance=model,
                seed=42,
                gen_config=gen_config,
                decoder_controller=controller
            )
            logs_by_mode[mode] = eval(
                inspect_task(name, solver_comp, limits[name]),
                max_connections=1,
            )[0]
        for mode, log in logs_by_mode.items():
            scores = log.results.scores if log.results else None
            print(f"{name} [{mode}]: {scores}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())