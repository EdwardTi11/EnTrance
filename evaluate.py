from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from benchmarks import BENCHMARKS, get_dataset
from benchmarks._verification import (
    extract_aime_answer,
    extract_code,
    extract_gpqa_answer,
    verify_livecodebench,
)
from model_design.engine import generate_text

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

MODES = ("baseline", "entranced")

# Default evaluation subsets.  AIME 2025 always uses the full 30 problems;
# the two larger benchmarks default to a fixed, deterministic subset.
DEFAULT_LIMITS = {"aime2025": 30, "gpqa_diamond": 32, "livecodebench": 12}

# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def count_forward_passes(trace: list[dict]) -> int:
    linear = sum(1 for entry in trace if entry.get("source") == "linear")
    search = sum(
        int(entry.get("search_forward_passes") or 0) for entry in trace
    )
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    model, problem_prompt, energy_gate, k_multiplier, search_engine, seed, gen
):
    model.reset()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        text, trace = generate_text(
            model=model,
            prompt=problem_prompt,
            energy_gate=energy_gate,
            k_multiplier=k_multiplier,
            search_engine=search_engine,
            seed=seed,
            max_tokens=gen["max_tokens"],
            temperature=gen["temperature"],
            top_k=gen["top_k"],
            top_p=gen["top_p"],
        )
    return text, count_forward_passes(trace)


# ---------------------------------------------------------------------------
# Scoring — dispatches to the right verification function per benchmark
# ---------------------------------------------------------------------------

def verify(bench_name: str, sample, output: str) -> bool:
    """Return True if *output* is correct for *sample* on *bench_name*."""
    if bench_name == "aime2025":
        extracted = extract_aime_answer(output)
        return extracted is not None and extracted == int(sample.target)
    elif bench_name == "gpqa_diamond":
        extracted = extract_gpqa_answer(output)
        return extracted is not None and extracted == sample.target
    elif bench_name == "livecodebench":
        code = extract_code(output)
        raw = sample.metadata.get("raw", {}) if sample.metadata else {}
        starter = (raw.get("starter_code") or "").strip()
        if starter:
            code = f"{starter}\n\n{code}"
        tests = raw.get("private_tests") or raw.get("public_tests") or []
        ok, _reason = verify_livecodebench(code, tests)
        return ok
    return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(bench_name: str, problems: list[dict]) -> dict:
    summary = {"benchmark": bench_name, "num_problems": len(problems)}
    for mode in MODES:
        correct = sum(1 for p in problems if p[mode]["correct"])
        total_flops = sum(p[mode]["flops"] for p in problems)
        summary[mode] = {
            "correct": correct,
            "num_problems": len(problems),
            "pass_at_1": correct / len(problems) if problems else 0.0,
            "total_flops": total_flops,
            "avg_flops": total_flops / len(problems) if problems else 0.0,
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

def print_overall(summaries: list[dict], parameter_count: int) -> None:
    total_problems = sum(s["num_problems"] for s in summaries)
    print(f"\nOverall summary ({total_problems} problems, baseline vs. EnTrance)")
    print("=" * 78)
    print(f"{'mode':<10} {'correct':<16} {'Pass@1':<10} {'total FLOPs':<16}")
    for mode in MODES:
        correct = sum(s[mode]["correct"] for s in summaries)
        flops = sum(s[mode]["total_flops"] for s in summaries)
        print(
            f"{mode:<10} {correct}/{total_problems:<14} "
            f"{correct / total_problems * 100:>6.2f}%   {format_flops(flops):<16}"
        )

    base = {m: sum(s[m]["total_flops"] for s in summaries) for m in MODES}
    base_acc = {
        m: sum(s[m]["correct"] for s in summaries) / total_problems
        for m in MODES
    }
    if base["baseline"] > 0:
        ratio = base["entranced"] / base["baseline"]
        acc_delta = (base_acc["entranced"] - base_acc["baseline"]) * 100
        print("-" * 78)
        print(
            f"EnTrance vs baseline: accuracy {acc_delta:+.2f} pts, "
            f"FLOPs x{ratio:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARKS),
        help="comma-separated benchmark names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max problems per benchmark (overrides the per-benchmark defaults)",
    )
    parser.add_argument("--aime-limit", type=int, default=DEFAULT_LIMITS["aime2025"])
    parser.add_argument("--gpqa-limit", type=int, default=DEFAULT_LIMITS["gpqa_diamond"])
    parser.add_argument("--lcb-limit", type=int, default=DEFAULT_LIMITS["livecodebench"])

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

    parser.add_argument("--list-only", action="store_true", help="print problem list and exit")
    parser.add_argument("--save-results", type=str, default=None, help="write JSON results to this path")
    parser.add_argument("--save-outputs", action="store_true", help="include model outputs in saved JSON")
    args = parser.parse_args(argv)

    limits = {
        "aime2025": args.aime_limit,
        "gpqa_diamond": args.gpqa_limit,
        "livecodebench": args.lcb_limit,
    }
    if args.limit is not None:
        limits = {name: args.limit for name in limits}

    selected = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        print(f"Unknown benchmark(s): {unknown}. Available: {sorted(BENCHMARKS)}")
        return 2

    # --list-only: validate data without loading the model.
    if args.list_only:
        for name in selected:
            dataset = get_dataset(name, limit=limits[name])
            samples = list(dataset)
            print(f"{name}: {len(samples)} problems to evaluate")
            for sample in samples:
                print(f"  - {sample.id}")
        return 0

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Point --model at your GGUF file (e.g. models/<name>.gguf).")
        return 1

    # Lazy imports — only needed when we have a model to evaluate.
    from llama_cpp import Llama, llama_model_n_params  # noqa: PLC0415
    from model_design.energy import EnergyProcessor  # noqa: PLC0415
    from model_design.engine import generate_text  # noqa: PLC0415
    from model_design.search import EGALBSSearch  # noqa: PLC0415

    print(f"Loading model: {model_path}")
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
    generation = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    print("=" * 78)
    print("EnTrance Evaluation Report")
    print("=" * 78)
    print(f"Model      : {model_path}")
    print(f"Parameters : {format_count(parameter_count)}")
    print("FLOPs model: estimated_flops = 2 * parameter_count * forward_passes")
    print(f"Seed       : {args.seed}")
    print("Baseline   : linear decoding (search_engine=None)")
    print(
        f"EnTrance   : EGALBS beam={args.beam_width} "
        f"depth={args.lookahead_depth} alpha={args.alpha} "
        f"gamma={args.gamma} k={args.k_multiplier}"
    )

    summaries: list[dict] = []
    all_results: list[dict] = []

    for name in selected:
        dataset = get_dataset(name, limit=limits[name])
        samples = list(dataset)
        results: list[dict] = []

        for index, sample in enumerate(samples, start=1):
            row = {"benchmark": name, "id": str(sample.id)}
            prompt = sample.input

            for mode in MODES:
                search = search_engine if mode == "entranced" else None
                text, passes = run_generation(
                    model, prompt, energy_gate, args.k_multiplier,
                    search, args.seed, generation,
                )
                correct = verify(name, sample, text)
                flops = 2 * parameter_count * passes
                row[mode] = {"correct": correct, "passes": passes, "flops": flops}
                if args.save_outputs:
                    row[mode]["output"] = text
                status = "OK " if correct else "BAD"
                print(
                    f"[{name} {index}/{len(samples)}] {str(sample.id):<24} "
                    f"{mode:<9} {status}  ({format_count(passes)} passes, "
                    f"{format_flops(flops)})"
                )
            results.append(row)

        summary = summarize(name, results)
        summaries.append(summary)
        all_results.extend(results)
        print_benchmark_table(summary)

    print_overall(summaries, parameter_count)

    if args.save_results:
        payload = {
            "model": str(model_path),
            "parameter_count": parameter_count,
            "seed": args.seed,
            "config": {
                "alpha": args.alpha,
                "gamma": args.gamma,
                "k_multiplier": args.k_multiplier,
                "beam_width": args.beam_width,
                "lookahead_depth": args.lookahead_depth,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "summaries": summaries,
            "results": all_results,
        }
        with open(args.save_results, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.save_results}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())