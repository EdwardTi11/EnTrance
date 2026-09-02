from llama_cpp import Llama
from model_design.engine import generate_text

model_path = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

text, trace = generate_text(
    model=model,
    prompt=input("Enter a prompt: "),
    seed=0,
)

print("\n")
print(text + "\n")