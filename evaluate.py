from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Generate, TaskState, solver
from inspect_evals.aime2025 import aime2025
from inspect_evals.bigcodebench import bigcodebench
from inspect_evals.gpqa import gpqa_diamond
from llama_cpp import Llama, llama_model_n_params

from model_design.energy import EnergyProcessor
from model_design.search import EGALBSSearch
from model_design.engine import generate_text

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
MODES = ("baseline", "entranced")
DEFAULT_LIMITS = {"aime2025": 30, "gpqa_diamond": 50, "bigcodebench": 40}


def count_forward_passes(trace: list[dict[str, Any]]) -> int:
    linear_tokens = sum(entry.get("source") == "linear" for entry in trace)
    search_forward_passes = sum(
        int(entry.get("search_forward_passes") or 0) for entry in trace
    )
    return linear_tokens + search_forward_passes

def format_flops(flops: float) -> str:
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops / 1e6:.2f} MFLOPs"


@solver
def entrance_generation(
    model_instance: Llama,
    energy_gate: EnergyProcessor,
    search_engine: EGALBSSearch | None,
    k_multiplier: float,
    seed: int,
    gen_config: dict[str, Any],
):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_instance.reset()
        text, trace = generate_text(
            model=model_instance,
            prompt=state.user_prompt.text,
            energy_gate=energy_gate,
            k_multiplier=k_multiplier,
            search_engine=search_engine,
            seed=seed,
            max_tokens=gen_config["max_tokens"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
        )

        state.output.completion = text
        state.metadata["forward_passes"] = count_forward_passes(trace)
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


def summarize(name: str, logs_by_mode, parameter_count: int) -> None:
    print(f"\n{name}\n" + "-" * 78)
    print(f"{'mode':<12} {'samples':<10} {'Pass@1':<10} {'total FLOPs':<18} {'avg FLOPs':<18}")
    for mode in MODES:
        samples = logs_by_mode[mode].samples or []
        correct = sum(
            sample.score is not None and sample.score.value in (True, 1, 1.0)
            for sample in samples
        )
        total_flops = sum(
            2 * parameter_count * int((sample.metadata or {}).get("forward_passes", 0))
            for sample in samples
        )
        average = total_flops / len(samples) if samples else 0
        pass_at_1 = correct / len(samples) if samples else 0
        print(f"{mode:<12} {len(samples):<10} {pass_at_1 * 100:>6.2f}%   {format_flops(total_flops):<18} {format_flops(average):<18}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EnTrance through Inspect AI.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_LIMITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aime-limit", type=int, default=DEFAULT_LIMITS["aime2025"])
    parser.add_argument("--gpqa-limit", type=int, default=DEFAULT_LIMITS["gpqa_diamond"])
    parser.add_argument("--bcb-limit", type=int, default=DEFAULT_LIMITS["bigcodebench"])

    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.2116)
    parser.add_argument("--gamma", type=float, default=0.8749)
    parser.add_argument("--k-multiplier", type=float, default=2.8360)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=9)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    limits = {
        "aime2025": args.aime_limit,
        "gpqa_diamond": args.gpqa_limit,
        "bigcodebench": args.bcb_limit,
    }
    if args.limit is not None:
        limits = {name: args.limit for name in limits}

    selected = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    if args.list_only:
        for name in selected:
            task = inspect_task(name, None, limits[name])
            print(f"{name}: {len(task.dataset)} problems to evaluate")
        return 0

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    model = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False,
        logits_all=True,
    )
    parameter_count = llama_model_n_params(model.model)
    energy_gate = EnergyProcessor(model=model, alpha=args.alpha, gamma=args.gamma)
    search_engine = EGALBSSearch(
        beam_width=args.beam_width, lookahead_depth=args.lookahead_depth
    )
    gen_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    for name in selected:
        logs_by_mode = {}
        for mode in MODES:
            solver_comp = entrance_generation(
                model_instance=model,
                energy_gate=energy_gate,
                search_engine=search_engine if mode == "entranced" else None,
                k_multiplier=args.k_multiplier,
                seed=args.seed,
                gen_config=gen_config,
            )

            # Construct task from inspect_evals module
            logs_by_mode[mode] = eval(inspect_task(name, solver_comp, limits[name]))[0]

        summarize(name, logs_by_mode, parameter_count)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())