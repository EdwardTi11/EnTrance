import optuna
from llama_cpp import Llama
from engine import generate_text
from energy import EnergyProcessor
from search import EGALBSSearch
from prompt_config import TUNING_SUITE

MODEL_PATH = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.5, 2.0)
    beta = trial.suggest_float("beta", 0.0, 1.5)
    gamma = trial.suggest_float("gamma", 0.0, 1.5)
    energy_threshold = trial.suggest_float("energy_threshold", 2.0, 6.0)

    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=4,
        verbose=False,
        logits_all=True
    )
    
    energy_gate = EnergyProcessor(model=model, alpha=alpha, beta=beta, gamma=gamma)
    search_engine = EGALBSSearch(beam_width=4, lookahead_depth=6)

    total_score = 0.0
    
    try:
        for prompt in list(TUNING_SUITE.values()):
            _, trace = generate_text(
                model=model,
                prompt=prompt,
                energy_gate=energy_gate,
                energy_threshold=energy_threshold,
                seed=42,
                search_engine=search_engine,
                max_tokens=150
            )
            
            search_triggers = sum(1 for step in trace if step.get("source") == "search")
            
            total_score += (search_triggers * 2.0) + (len(trace) * 0.1)
            
    finally:
        del model

    return total_score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Clean terminal printout for instant manual copy-paste
    best = study.best_params
    print("\n" + "="*50)
    print("🏆 OPTUNA TUNING COMPLETE! COPY & PASTE THESE VALUES:")
    print("="*50)
    print(f"tuned_alpha = {best['alpha']:.4f}")
    print(f"tuned_beta = {best['beta']:.4f}")
    print(f"tuned_gamma = {best['gamma']:.4f}")
    print(f"ENERGY_THRESHOLD = {best['energy_threshold']:.4f}")
    print("="*50 + "\n")