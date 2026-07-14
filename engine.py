import numpy as np
from llama_cpp import Llama

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

def generate_text(
    model: Llama,
    prompt: str,
    energy_gate,
    energy_threshold: float,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    stop_tokens: list[int] | None = None,
    prompt_formatter=None,
    seed: int | None = None,
):
    if prompt_formatter:
        prompt = prompt_formatter(prompt)

    rng = np.random.default_rng(seed)
    tokens = model.tokenize(prompt.encode("utf-8"))

    model.eval(tokens)
    trace = []
    generated_tokens = []
    stop_tokens = stop_tokens or [model.token_eos()]

    for step in range(max_tokens):
        logits = np.array(model.scores[model.n_tokens - 1, :], dtype=np.float32)

        selected_id, selected_prob, top_idx, top_probs = sample_token(
            logits, temperature, top_k, top_p, rng, logit_processors=None, prev_tokens=generated_tokens
        )

        energy_landscape = energy_gate.energy(logits, generated_tokens)
        token_energy = energy_landscape[selected_id]

        token_str = model.detokenize([selected_id]).decode("utf-8", errors="replace")

        if token_energy > energy_threshold:
            print(f"\n🛑 [ENERGY SPIKE DETECTED] Token: {token_str!r} | Energy: {token_energy:.4f} > {energy_threshold}")
            print("Halting linear execution immediately to prepare for search...")
            break

        entry = {
            "token_position": model.n_tokens,
            "cumulative_tokens": step + 1,
            "selected_token_id": selected_id,
            "selected_token_str": token_str,
            "selected_token_prob": selected_prob,
            "energy": token_energy,
        }
        trace.append(entry)
        generated_tokens.append(selected_id)

        if selected_id in stop_tokens:
            break

        model.eval([selected_id])

    generated_text = model.detokenize(generated_tokens).decode("utf-8", errors="replace")
    return generated_text, trace