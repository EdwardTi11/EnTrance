import os

from llama_cpp import Llama
from model_design.engine import generate_text
from model_design.adaptive_control import DecoderController

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "Phi-4-mini-reasoning-Q4_K_M.gguf"))
CPU_COUNT = os.cpu_count() or 4

model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=(CPU_COUNT // 2),
    verbose=False,
    logits_all=False
)
decoder_controller = DecoderController()

text, trace = generate_text(
    model=model,
    prompt=input("Enter a prompt: "),
    seed=0,
    decoder_controller=decoder_controller,
)

print("\n")
print(text + "\n")