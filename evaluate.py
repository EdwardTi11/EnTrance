import re
import numpy as np
from llama_cpp import Llama
from engine import generate_text
from energy import EnergyProcessor
from search import EGALBSSearch
from prompt_config import TESTING_SUITE

# 1. Model File Paths
MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
REASONING_MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf"

# 2. Instantiate Base Model Pipeline
print("🌀 Initializing Base Instruct Model...")
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

# 3. Tuning parameters Configuration
ALPHA = 2.2679
BETA = 1.8757
GAMMA = 0.3362
ENERGY_THRESHOLD = 5.2913
BEAM_WIDTH = 2
LOOKAHEAD_DEPTH = 7

energy_gate = EnergyProcessor(model=model, alpha=ALPHA, beta=BETA, gamma=GAMMA)
search_engine = EGALBSSearch(beam_width=BEAM_WIDTH, lookahead_depth=LOOKAHEAD_DEPTH)

def verify_output(task, text):
    if task["verification"] == "regex":
        return bool(re.search(task["expected_pattern"], text))
    elif task["verification"] == "unit_test":
        try:
            local_vars = {}
            exec(text + "\n" + task["test_code"], {}, local_vars)
            return True
        except Exception:
            return False
    return False

def run_evaluation(mode="steered"):
    results = {}
    current_threshold = float('inf') if mode == "baseline" else ENERGY_THRESHOLD
    
    print(f"\n🚀 Running evaluation suite in [{mode.upper()}] mode...")
    
    for task_name, task in TESTING_SUITE.items():
        difficulty = task.get("difficulty", "Medium")
        if difficulty == "Easy":
            max_tokens = 128
        elif difficulty == "Medium":
            max_tokens = 256
        elif difficulty == "Hard":
            max_tokens = 512
        else:
            max_tokens = 256

        model.reset()
        
        generated_text, trace = generate_text(
            model=model,
            prompt=task["prompt"],
            energy_gate=energy_gate,
            energy_threshold=current_threshold,
            seed=42,  # Fixed seed for a fair deterministic baseline comparison
            search_engine=search_engine,
            max_tokens=max_tokens
        )
        
        # Calculate evaluation metrics
        linear_tokens = sum(1 for step in trace if step.get("source") == "linear")
        search_passes = sum(step.get("search_forward_passes", 0) for step in trace if "search_forward_passes" in step)
        total_passes = linear_tokens + search_passes
        
        is_correct = verify_output(task, generated_text)
        
        results[task_name] = {
            "correct": is_correct,
            "forward_passes": total_passes,
            "text_length": len(generated_text)
        }
        print(f"  - {task_name:<25} | Correct: {str(is_correct):<5} | Forward Passes: {total_passes}")
        
    return results

def run_native_reasoning_evaluation():
    """Runs the testing suite natively using the specialized Phi-4-mini-reasoning pipeline."""
    results = {}
    
    print("\n🧠 Initializing Native Reasoning Model (Phi-4-mini-reasoning)...")
    reasoning_model = Llama(
        model_path=REASONING_MODEL_PATH,
        n_ctx=4096, 
        n_threads=4,
        verbose=False,
    )
    
    print("🚀 Running evaluation suite in [NATIVE REASONING] mode...")
    
    for task_name, task in TESTING_SUITE.items():
        reasoning_model.reset()
        
        difficulty = task.get("difficulty", "Medium")
        if difficulty == "Easy":
            reasoning_budget = 512
        elif difficulty == "Medium":
            reasoning_budget = 1024
        else:
            reasoning_budget = 2048
            
        formatted_prompt = f"<|user|>\n{task['prompt']}<|end|>\n<|assistant|>\n"
            
        output = reasoning_model(
            prompt=formatted_prompt,
            max_tokens=reasoning_budget,
            seed=42,
            temperature=0.0
        )
        
        raw_text = output["choices"][0]["text"]
        
        if "<|solution|>" in raw_text:
            cleaned_text = raw_text.split("<|solution|>")[-1].strip()
        elif "</thought>" in raw_text:
            cleaned_text = raw_text.split("</thought>")[-1].strip()
        else:
            cleaned_text = raw_text.strip()
            
        forward_passes = output["usage"]["completion_tokens"] + output["usage"]["prompt_tokens"]
        
        # Verify using the isolated solution payload rather than the full thinking sequence
        is_correct = verify_output(task, cleaned_text)
        
        results[task_name] = {
            "correct": is_correct,
            "forward_passes": forward_passes,
            "text_length": len(cleaned_text)
        }
        print(f"  - {task_name:<25} | Correct: {str(is_correct):<5} | Forward Passes: {forward_passes}")
        
    return results

if __name__ == "__main__":
    baseline_results = run_evaluation(mode="baseline")
    steered_results = run_evaluation(mode="steered")
    reasoning_results = run_native_reasoning_evaluation()
    
    print("\n" + "="*112)
    print(f"{'🏆 THE THREE-WAY SHOWDOWN: RUNTIME STEERING VS NATIVE REASONING MODELS':^112}")
    print("="*112)
    print(f"{'Task Name':<25} | {'Base Acc':<9} | {'Steered Acc':<11} | {'Reason Acc':<10} | {'Base Pass':<9} | {'Steered Pass':<12} | {'Reason Pass':<11}")
    print("-"*112)
    
    totals = {
        "base_ok": 0, "steered_ok": 0, "reason_ok": 0,
        "base_p": 0, "steered_p": 0, "reason_p": 0
    }
    
    for task_name in TESTING_SUITE.keys():
        b = baseline_results[task_name]
        s = steered_results[task_name]
        r = reasoning_results[task_name]
        
        totals["base_ok"] += 1 if b["correct"] else 0
        totals["steered_ok"] += 1 if s["correct"] else 0
        totals["reason_ok"] += 1 if r["correct"] else 0
        
        totals["base_p"] += b["forward_passes"]
        totals["steered_p"] += s["forward_passes"]
        totals["reason_p"] += r["forward_passes"]
        
        b_acc = "✅ PASS" if b["correct"] else "❌ FAIL"
        s_acc = "✅ PASS" if s["correct"] else "❌ FAIL"
        r_acc = "✅ PASS" if r["correct"] else "❌ FAIL"
        
        print(f"{task_name:<25} | {b_acc:<9} | {s_acc:<11} | {r_acc:<10} | {b['forward_passes']:<9} | {s['forward_passes']:<12} | {r['forward_passes']:<11}")
        
    print("-"*112)
    
    num_tasks = len(TESTING_SUITE)
    base_accuracy = (totals["base_ok"] / num_tasks) * 100
    steered_accuracy = (totals["steered_ok"] / num_tasks) * 100
    reason_accuracy = (totals["reason_ok"] / num_tasks) * 100
    
    print(f"{'TOTAL SUMMARY':<25} | {f'{base_accuracy:.1f}%':<9} | {f'{steered_accuracy:.1f}%':<11} | {f'{reason_accuracy:.1f}%':<10} | {totals['base_p']:<9} | {totals['steered_p']:<12} | {totals['reason_p']:<11}")
    print("="*112)
    
    # Structural Compute Comparison vs Native Reasoning Model
    steered_vs_reason_reduction = 0.0
    if totals["reason_p"] > 0:
        steered_vs_reason_reduction = ((totals["reason_p"] - totals["steered_p"]) / totals["reason_p"]) * 100

    print("\n📊 CROSS-ARCHITECTURE ANALYSIS:")
    print(f"  • Base Instruct Accuracy:     {base_accuracy:.1f}%")
    print(f"  • EGALBS Steered Accuracy:    {steered_accuracy:.1f}%")
    print(f"  • Native Reasoning Accuracy:  {reason_accuracy:.1f}%")
    print(f"  • EGALBS vs Native Reasoning Token Compression: Your runtime engine used {steered_vs_reason_reduction:.1f}% fewer tokens than the native reasoning loops! 📉")
    print("="*112 + "\n")