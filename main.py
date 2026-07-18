from llama_cpp import Llama
from engine import generate_text
from energy import EnergyProcessor
from search import EGALBSSearch

model_path = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

energy_gate = EnergyProcessor(model=model, alpha=1.7766, beta=0.891, gamma=1.4413)
ENERGY_THRESHOLD = 3.6743

search_engine = EGALBSSearch(beam_width=4, lookahead_depth=11)

text, trace = generate_text(
    model=model,
    prompt=input("Enter a prompt: "),
    energy_gate=energy_gate,
    energy_threshold=ENERGY_THRESHOLD,
    search_engine=search_engine,
    seed=0,
)

print("\n--- Final Model Output ---")
print(text + "\n")