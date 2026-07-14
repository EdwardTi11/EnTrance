import time
import numpy as np
from llama_cpp import Llama
from energy import EnergyLogitProcessor

def format_prompt(user_message: str) -> str:
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def full_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()

def topk_softmax(logits: np.ndarray, k: int):
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    shifted = logits[idx] - logits[idx][0]
    exp = np.exp(shifted)
    return idx, exp / exp.sum()

def sample_token(logits: np.ndarray, temperature: float, top_k: int, top_p: float,
                  rng: np.random.Generator, logit_processors=None, prev_tokens=None):
    if logit_processors:
        for proc in logit_processors:
            logits = proc(logits, prev_tokens or [])

    scaled = logits / max(temperature, 1e-6)
    idx, probs = topk_softmax(scaled, top_k)

    cum = np.cumsum(probs)
    cutoff = np.searchsorted(cum, top_p) + 1
    idx, probs = idx[:cutoff], probs[:cutoff]
    probs = probs / probs.sum()

    choice = rng.choice(len(idx), p=probs)
    selected_id = int(idx[choice])
    selected_prob = float(probs[choice])
    return selected_id, selected_prob, idx, probs

def generate_with_trace(
    model: Llama,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    stop_tokens: list[int] | None = None,
    prompt_formatter=None,
    logit_processors=None,
    trace_mode: str = "full",
    seed: int | None = None,
):
    if trace_mode not in ("full", "compact"):
        raise ValueError("trace_mode must be 'full' or 'compact'")
    if prompt_formatter:
        prompt = prompt_formatter(prompt)

    rng = np.random.default_rng(seed)
    tokens = model.tokenize(prompt.encode("utf-8"))

    t_eval0 = time.perf_counter()
    model.eval(tokens)
    trace = []
    generated_tokens = []
    stop_tokens = stop_tokens or [model.token_eos()]

    for step in range(max_tokens):
        eval_latency = time.perf_counter() - t_eval0

        logits = np.array(model.scores[model.n_tokens - 1, :], dtype=np.float32)

        t_sample0 = time.perf_counter()
        selected_id, selected_prob, top_idx, top_probs = sample_token(
            logits, temperature, top_k, top_p, rng, logit_processors, generated_tokens
        )
        sample_latency = time.perf_counter() - t_sample0

        top5 = top_probs[:5]
        entry = {
            "token_position": model.n_tokens,
            "cumulative_tokens": step + 1,
            "selected_token_id": selected_id,
            "selected_token_str": model.detokenize([selected_id]).decode("utf-8", errors="replace"),
            "selected_token_prob": selected_prob,
            "top5_token_ids": top_idx[:5].tolist(),
            "top5_probs": top5.tolist(),
            "entropy": float(-np.sum(top_probs * np.log(top_probs + 1e-12))),
            "margin": float(logits[top_idx[0]] - logits[top_idx[1]]) if len(top_idx) > 1 else 0.0,
            "sample_latency_s": sample_latency,
            "eval_latency_s": eval_latency,
        }
        if trace_mode == "full":
            entry["logits"] = logits
            entry["probabilities"] = full_softmax(logits)

        trace.append(entry)
        generated_tokens.append(selected_id)

        if selected_id in stop_tokens:
            break

        t_eval0 = time.perf_counter()
        model.eval([selected_id])

    generated_text = model.detokenize(generated_tokens).decode("utf-8", errors="replace")
    return generated_text, trace

model_path = r"C:\Users\etito\Projects\EnTrance\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    logits_all=True
)

energy_gate = EnergyLogitProcessor(
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
    max_tokens=256,
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