import os
import re
from datasets import load_dataset
from llama_cpp import Llama
from model_design.adaptive_control import DecoderController
from model_design.engine import generate_text

MODEL_PATH = os.getenv("MODEL_PATH", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
SEED = 42
GEN_CONFIG = {"temperature": 0.8, "min_p": 0.05, "max_tokens": 4096}

# Added the ground-truth answer column name for each dataset
DATASETS = [
    ("AIME 2025", "math-ai/aime25", "test", "problem", "answer"),
    ("GPQA Diamond", "fingertap/GPQA-Diamond", "test", "Question", "Correct Answer"),
]

model = Llama(model_path=MODEL_PATH, n_ctx=8192, n_threads=(os.cpu_count() or 4) // 2, verbose=False)

def check_correctness(generated_text, ground_truth, dataset_name):
    """Simple parser to evaluate model correctness against ground truth."""
    if not ground_truth:
        return False
        
    if "AIME" in dataset_name:
        # AIME answers are integers (0-999). Extract numbers and check the last one.
        numbers = re.findall(r'\b\d+\b', generated_text)
        if numbers:
            predicted = numbers[-1]
            return str(predicted).strip() == str(ground_truth).strip()
        return False
    else:
        # GPQA Diamond uses multiple-choice options. Check if the answer string is present.
        return str(ground_truth).strip().upper() in generated_text.strip().upper()

for name, repo, split, prompt_col, answer_col in DATASETS:
    dataset = load_dataset(repo, split=split).shuffle(seed=SEED)
    prompts = dataset[prompt_col]
    answers = dataset[answer_col] if answer_col in dataset.column_names else [None] * len(prompts)
    
    print(f"\n=== {name} ({len(prompts)} samples) ===")

    for mode, use_ctrl in [("BASELINE", False), ("ENTRANCED", True)]:
        p_toks = c_toks = correct_count = 0
        
        for prompt, gt in zip(prompts, answers):
            model.reset()
            output, stats = generate_text(
                model=model,
                prompt=prompt,
                seed=SEED,
                decoder_controller=DecoderController() if use_ctrl else None,
                **GEN_CONFIG,
            )
            
            # Extract generation text (adjust if your generate_text returns a dict/object)
            gen_text = output if isinstance(output, str) else str(output)
            
            usage = stats.get("token_usage", {})
            p_toks += usage.get("prompt_tokens", 0)
            c_toks += usage.get("completion_tokens", 0)
            
            if check_correctness(gen_text, gt, name):
                correct_count += 1

        accuracy = (correct_count / len(prompts)) * 100 if len(prompts) > 0 else 0
        print(f"[{mode}] Accuracy: {accuracy:.1f}% ({correct_count}/{len(prompts)}) | Prompt: {p_toks} | Completion: {c_toks}")