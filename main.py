from llama_cpp import Llama
from engine import generate_with_trace, format_prompt
from energy import EnergyProcessor

model_path = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

energy_gate = EnergyProcessor(
    model=model,
    alpha=1.0,
    beta=0.3,
    gamma=0.8,
    repetition_window=32
)

ENERGY_THRESHOLD = 4.0

steering_processor = lambda logits, prev: energy_gate.call(logits, prev)

text, trace = generate_with_trace(
    model,
    input("Enter a prompt: "),
    max_tokens=128,
    prompt_formatter=format_prompt,
    seed=0,
)

print("\n--- Model Output ---")
print(text + "\n")

# 6. Audit your live threshold metrics
print("\n=== Live Energy Evaluation ===")
historical_tokens = []
for entry in trace:
    token_id = entry["selected_token_id"]
    token_str = entry["selected_token_str"]
    step_logits = entry["logits"]
    
    # Run the raw energy function post-step to see how well it guarded the loop
    energy_landscape = energy_gate.energy(step_logits, historical_tokens)
    token_energy = energy_landscape[token_id]
    
    print(f"Token: {token_str!r:<10} | Energy: {token_energy:.4f}")
    
    if token_energy > ENERGY_THRESHOLD:
        print(f"🛑 [ENERGY SPIKE] Value {token_energy:.2f} crossed your threshold limit!")
        
    historical_tokens.append(token_id)