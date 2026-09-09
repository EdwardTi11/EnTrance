from typing import Any

from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Generate, TaskState, solver
from inspect_evals.aime2025 import aime2025
from inspect_evals.gpqa import gpqa_diamond
# from inspect_evals.hle import hle
from llama_cpp import Llama

from model_design.engine import generate_text
from model_design.adaptive_control import DecoderController

MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\Phi-4-mini-reasoning-Q4_K_M.gguf"

GEN_CONFIG = {
    "temperature": 0.8,
    "min_p": 0.05,
}

@solver
def entrance_generation(
    model_instance: Llama,
    seed: int,
    gen_config: dict[str, Any],
    decoder_controller: bool = False,  # Expects True for "entranced", False for "baseline"
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_instance.reset()
        
        controller = DecoderController() if decoder_controller else None
        text, _ = generate_text(
            model=model_instance,
            prompt=state.user_prompt.text,
            seed=seed,
            temperature=gen_config["temperature"],
            min_p=gen_config["min_p"],
            decoder_controller=controller,  # Pass the fresh instance or None
        )
        state.output.completion = text
        return state
    return solve

def main() -> int:
    tasks = [
        ("aime2025", aime2025(), None),
        ("gpqa_diamond", gpqa_diamond(), None),
    ]

    for name, task, limit in tasks:
        if limit and len(task.dataset) > limit:
            task.dataset = MemoryDataset(list(task.dataset)[:limit])

        for mode in ("baseline", "entranced"):
            model = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=4,
                verbose=False,
                logits_all=True,
            )

            task.solver = entrance_generation(
                model_instance=model,
                seed=42,
                gen_config=GEN_CONFIG,
                decoder_controller=(mode == "entranced"),
            )

            log = eval(task, max_connections=1)[0]
            scores = log.results.scores if log.results else None
            print(f"{name} [{mode}]: {scores}")
            del model
    return 0

if __name__ == "__main__":
    raise SystemExit(main())