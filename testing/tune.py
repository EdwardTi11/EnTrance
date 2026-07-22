import optuna
import optunahub
import re
from llama_cpp import Llama
from model_design.engine import generate_text
from model_design.energy import EnergyProcessor
from model_design.search import EGALBSSearch
from .prompt_config import TUNING_SUITE

mm_module = optunahub.load_module("pruners/multi_metric_pruner")
MultiMetricPruner = mm_module.MultiMetricPruner
MultiMetricPrunerTrial = mm_module.MultiMetricPrunerTrial

MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

def objective(trial):
    mmt = MultiMetricPrunerTrial(trial)
    
    alpha = trial.suggest_float("alpha", 0.5, 2.5)
    beta = trial.suggest_float("beta", 0.0, 2.0)
    gamma = trial.suggest_float("gamma", 0.1, 1.5)
    k_multiplier = trial.suggest_float("k_multiplier", 1.5, 3.5)
    beam_width = trial.suggest_int("beam_width", 2, 6)
    lookahead_depth = trial.suggest_int("lookahead_depth", 4, 12)
    
    energy_gate = EnergyProcessor(model=model, alpha=alpha, beta=beta, gamma=gamma)
    search_engine = EGALBSSearch(beam_width=beam_width, lookahead_depth=lookahead_depth)

    total_errors = 0
    total_forward_passes = 0
    
    for step, (task_name, task) in enumerate(TUNING_SUITE.items()):
        prompt_text = task["prompt"]

        difficulty = task.get("difficulty", "Medium")
        if difficulty == "Easy":
            max_tokens_budget = 128
        elif difficulty in ("Medium"):
            max_tokens_budget = 256
        elif difficulty in ("Hard"):
            max_tokens_budget = 512
        else:
            max_tokens_budget = 256  # Fallback safety net
        
        model.reset()
        
        generated_text, trace = generate_text(
            model=model,
            prompt=prompt_text,
            energy_gate=energy_gate,
            k_multiplier=k_multiplier,
            seed=42,
            search_engine=search_engine,
            max_tokens=max_tokens_budget
        )
        
        linear_tokens = sum(1 for entry in trace if entry.get("source") == "linear")
        search_passes = sum(entry.get("search_forward_passes", 0) for entry in trace if "search_forward_passes" in entry)
        
        total_forward_passes += (linear_tokens + search_passes)
        
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

        if not is_correct:
            total_errors += 1
            
        mmt.report({"errors": total_errors, "passes": total_forward_passes}, step=step)
        if mmt.should_prune():
            raise optuna.TrialPruned()

    return total_errors, total_forward_passes

if __name__ == "__main__":
    base_pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    
    mo_pruner = MultiMetricPruner(
        base_pruner, 
        metric_directions={"errors": "minimize", "passes": "minimize"}, 
        joint=True
    )
    
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        pruner=mo_pruner
    )
    study.optimize(objective, n_trials=40)

    best_trials = study.best_trials
    max_errors = max(t.values[0] for t in best_trials) or 1
    max_passes = max(t.values[1] for t in best_trials) or 1
    
    def get_distance_to_perfect(trial):
        norm_error = trial.values[0] / max_errors
        norm_passes = trial.values[1] / max_passes
        return (norm_error**2 + norm_passes**2) ** 0.5

    golden_trial = min(best_trials, key=get_distance_to_perfect)
    best = golden_trial.params
    
    print("\n🌟 THE GOLDEN UNIFIED ENTRANCE CONFIGURATION:")
    print(f"  Errors: {golden_trial.values[0]} | Total Forward Passes: {golden_trial.values[1]}")
    print("="*50)
    print(f"tuned_alpha = {best['alpha']:.4f}")
    print(f"tuned_beta = {best['beta']:.4f}")
    print(f"tuned_gamma = {best['gamma']:.4f}")
    print(f"tuned_k_multiplier = {best['k_multiplier']:.4f}")
    print(f"tuned_beam_width = {best['beam_width']}")
    print(f"tuned_lookahead_depth = {best['lookahead_depth']}")
    print("="*50 + "\n")