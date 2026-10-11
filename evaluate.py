import random
import os
from typing import Any

from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Generate, TaskState, solver
from inspect_evals.aime2025 import aime2025
from inspect_evals.gpqa import gpqa_diamond
from llama_cpp import Llama

from model_design.engine import generate_text
from model_design.adaptive_control import DecoderController

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "Phi-4-mini-reasoning-Q4_K_M.gguf"))
CPU_COUNT = os.cpu_count() or 4

GEN_CONFIG = {
    "temperature": 0.8,
    "min_p": 0.05,
    "max_tokens": 4096,
}

@solver
def entrance_generation(
    model_instance: Llama,
    seed: int,
    gen_config: dict[str, Any],
    decoder_controller: bool = False,  # True for "entranced", False for "baseline"
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_instance.reset()
        
        controller = DecoderController() if decoder_controller else None
        text, token_stats = generate_text(
            model=model_instance,
            prompt=state.user_prompt.text,
            seed=seed,
            temperature=gen_config["temperature"],
            min_p=gen_config["min_p"],
            max_tokens=gen_config["max_tokens"],
            decoder_controller=controller,
        )
        state.output.completion = text
        state.metadata["usage"] = token_stats.get("token_usage", {})

        return state
    return solve

def main() -> int:
    tasks = [
        ("AIME 2025", aime2025(), 10),
        ("GPQA Diamond", gpqa_diamond(), 10),
    ]

    for task_name, task, limit in tasks:
        dataset_samples = list(task.dataset)
        
        random.seed(42)
        random.shuffle(dataset_samples)
        
        if limit and len(dataset_samples) > limit:
            dataset_samples = dataset_samples[:limit]
            
        task.dataset = MemoryDataset(dataset_samples)

        print(f"\n==============================")
        print(f"Benchmark Suite: {task_name} (Shuffled, {len(task.dataset)} samples)")
        print(f"==============================")

        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=(CPU_COUNT // 2),
            verbose=False,
            logits_all=False,
        )

        for mode in ("baseline", "entranced"):
            task.solver = entrance_generation(
                model_instance=model,
                seed=42,
                gen_config=GEN_CONFIG,
                decoder_controller=(mode == "entranced"),
            )

            log = eval(task, max_connections=1, display="none")[0]
            
            total_prompt_tokens = 0
            total_completion_tokens = 0
            sample_count = len(log.samples) if log.samples else 0

            if log.samples:
                for sample in log.samples:
                    usage = sample.metadata.get("usage", {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)

            avg_completion = total_completion_tokens / sample_count if sample_count > 0 else 0

            print(f"[{mode.upper()}] Processed {sample_count} samples:")
            print(f"  - Total Prompt Tokens: {total_prompt_tokens}")
            print(f"  - Total Completion Tokens: {total_completion_tokens}")
            print(f"  - Avg Completion Tokens / Sample: {avg_completion:.2f}")

            del model
            
    return 0

if __name__ == "__main__":
    raise SystemExit(main())