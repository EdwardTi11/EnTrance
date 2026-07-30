from llama_cpp import Llama
from model_design.engine import generate_text
from model_design.energy import EnergyProcessor
from model_design.search import EGALBSSearch

model_path = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

energy_gate = EnergyProcessor(model=model, alpha=1.5784, gamma=0.8099)
search_engine = EGALBSSearch(beam_width=4, lookahead_depth=9)

text, trace = generate_text(
    model=model,
    prompt=input("Enter a prompt: "),
    energy_gate=energy_gate,
    k_multiplier=2.9780,
    search_engine=search_engine,
    seed=0,
)

print(text + "\n")
print(trace)