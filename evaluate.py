from __future__ import annotations

import argparse
from pathlib import Path

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

BENCHMARK_TASKS = {
    "aime2025": aime2025,
    "gpqa_diamond": gpqa_diamond,
    "bigcodebench": bigcodebench,
}

def count_forward_passes(trace: list[dict]) -> int:
    linear = sum(1 for entry in trace if entry.get("source") == "linear")
    search = sum(int(entry.get("search_forward_passes") or 0) for entry in trace)
    return linear + search

def format_flops(flops: float) -> str:
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops / 1e6:.2f} MFLOPs"

def format_count(n: int) -> str:
    return f"{n:,}"

@solver
def entrance_generation(
    model_instance,
    energy_gate,
    search_engine,
    k_multiplier: float,
    seed: int,
    gen_config: dict,
):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_instance.reset()
        prompt = state.user_prompt.text

        text, trace = generate_text(
            model=model_instance,
            prompt=prompt,
            energy_gate=energy_gate,
            k_multiplier=k_multiplier,
            search_engine=search_engine,
            seed=seed,
            max_tokens=gen_config["max_tokens"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
        )

        passes = count_forward_passes(trace)
        state.output.completion = text
        state.metadata["forward_passes"] = passes
        return state

    return solve

def get_inspect_task(bench_name: str, limit: int, custom_solver) -> Task:
    task_func = BENCHMARK_TASKS[bench_name]
    
    # Initialize the inspect_evals task with built-in dataset and default scorer
    task_instance = task_func()
    
    # Cap dataset length if limit is provided
    if limit and len(task_instance.dataset) > limit:
        task_instance.dataset = MemoryDataset(task_instance.dataset[:limit])
        
    # Inject custom generation solver ahead of task processing/scoring
    task_instance.solver = custom_solver
    return task_instance

def summarize_log_results(bench_name: str, eval_results: dict) -> dict:
    summary = {"benchmark": bench_name, "num_problems": len(eval_results["baseline"])}
    for mode in MODES:
        results = eval_results[mode]
        correct = sum(1 for r in results if r["correct"])
        total_flops = sum(r["flops"] for r in results)
        summary[mode] = {
            "correct": correct,
            "num_problems": len(results),
            "pass_at_1": correct / len(results) if results else 0.0,
            "total_flops": total_flops,
            "avg_flops": total_flops / len(results) if results else 0.0,
        }
    return summary

def print_benchmark_table(summary: dict) -> None:
    print(f"\n{summary['benchmark']} ({summary['num_problems']} problems)")
    print("-" * 78)
    header = (
        f"{'mode':<10} {'correct':<16} {'Pass@1':<10} "
        f"{'total FLOPs':<16} {'avg FLOPs/problem':<18}"
    )
    print(header)
    for mode in MODES:
        row = summary[mode]
        print(
            f"{mode:<10} "
            f"{row['correct']}/{row['num_problems']:<14} "
            f"{row['pass_at_1'] * 100:>6.2f}%   "
            f"{format_flops(row['total_flops']):<16} "
            f"{format_flops(row['avg_flops']):<18}"
        )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_TASKS.keys()))
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
            task_obj = BENCHMARK_TASKS[name]()
            dataset_len = min(len(task_obj.dataset), limits[name])
            print(f"{name}: {dataset_len} problems to evaluate")
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

    summaries = []

    for name in selected:
        eval_mode_results = {}
        for mode in MODES:
            engine = search_engine if mode == "entranced" else None
            solver_comp = entrance_generation(
                model_instance=model,
                energy_gate=energy_gate,
                search_engine=engine,
                k_multiplier=args.k_multiplier,
                seed=args.seed,
                gen_config=gen_config,
            )

            # Construct task from inspect_evals module
            task_obj = get_inspect_task(name, limits[name], solver_comp)
            
            # Execute evaluation with inspect_ai engine
            results_log = eval(task_obj)[0]

            parsed_results = []
            for sample in results_log.samples:
                passes = sample.metadata.get("forward_passes", 0)
                # Parse inspect_ai score result
                is_correct = (
                    sample.score.value == 1.0
                    if (sample.score and sample.score.value is not None)
                    else False
                )
                flops = 2 * parameter_count * passes
                parsed_results.append({
                    "correct": is_correct,
                    "passes": passes,
                    "flops": flops,
                    "output": sample.output.completion if sample.output else "",
                })

            eval_mode_results[mode] = parsed_results

        summary = summarize_log_results(name, eval_mode_results)
        summaries.append(summary)
        print_benchmark_table(summary)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())