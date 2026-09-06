from llama_cpp import Llama
from model_design.engine import generate_text
from model_design.adaptive_control import DecoderController

model_path = r"C:\Users\etito\Projects\EnTrance\models\Phi-4-mini-reasoning-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
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
print(trace)