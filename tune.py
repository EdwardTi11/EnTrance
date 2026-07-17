import optuna
from llama_cpp import Llama
import time
import re
from engine import generate_text
from energy import EnergyProcessor
from search import EGALBSSearch
from prompt_config import TUNING_SUITE

MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=4,
        verbose=False,
        logits_all=True
    )

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.5, 2.5)
    beta = trial.suggest_float("beta", 0.0, 2.0)
    gamma = trial.suggest_float("gamma", 0.1, 1.5)
    energy_threshold = trial.suggest_float("energy_threshold", 1.5, 6.0)
    beam_width = trial.suggest_int("beam_width", 2, 6)
    lookahead_depth = trial.suggest_int("lookahead_depth", 4, 12)
    
    energy_gate = EnergyProcessor(model=model, alpha=alpha, beta=beta, gamma=gamma)
    search_engine = EGALBSSearch(beam_width=beam_width, lookahead_depth=lookahead_depth)

    total_score = 0.0
    
    for _, task in TUNING_SUITE.items():
            prompt_text = task["prompt"]
            model.reset()
            
            start_time = time.perf_counter()
            
            generated_text, trace = generate_text(
                model=model,
                prompt=prompt_text,
                energy_gate=energy_gate,
                energy_threshold=energy_threshold,
                seed=42,
                search_engine=search_engine,
                max_tokens=512
            )
            
            # End timer
            elapsed_time = time.perf_counter() - start_time
            
            # 1. Verify correctness
            is_correct = False
            if task["verification"] == "regex":
                if re.search(task["expected_pattern"], generated_text):
                    is_correct = True
            elif task["verification"] == "unit_test":
                try:
                    local_vars = {}
                    exec(generated_text + "\n" + task["test_code"], {}, local_vars)
                    is_correct = True
                except Exception:
                    is_correct = False

            # 2. Score calculation with latency penalty
            if is_correct:
                task_score = -100.0 
                
                search_triggers = sum(1 for step in trace if step.get("source") == "search")
                total_tokens = len(trace)
                
                task_score += (search_triggers * 0.5) + (total_tokens * 0.05) + (elapsed_time * 0.2)
            else:
                task_score = 100.0
                
            total_score += task_score

    return total_score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Clean terminal printout for instant manual copy-paste
    best = study.best_params
    print("\n" + "="*50)
    print("🏆 OPTUNA TUNING COMPLETE! COPY & PASTE THESE VALUES:")
    print("="*50)
    print(f"tuned_alpha = {best['alpha']:.4f}")
    print(f"tuned_beta = {best['beta']:.4f}")
    print(f"tuned_gamma = {best['gamma']:.4f}")
    print(f"tuned_beam_width = {best['beam_width']}")
    print(f"tuned_lookahead_depth = {best['lookahead_depth']}")
    print(f"ENERGY_THRESHOLD = {best['energy_threshold']:.4f}")
    print("="*50 + "\n")