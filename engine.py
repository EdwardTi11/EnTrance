import numpy as np
from llama_cpp import Llama
from search import should_trigger_search

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
    return selected_id, selected_prob

def generate_text(
    model: Llama,
    prompt: str,
    energy_gate,
    energy_threshold: float,
    search_engine,
    max_tokens: int | None = None,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    stop_tokens: list[int] | None = None,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    tokens = model.tokenize(prompt.encode("utf-8"))

    remaining_budget = model.n_ctx() - len(tokens) - 4

    if max_tokens is None or max_tokens <= 0:
        max_tokens = remaining_budget
    else:
        max_tokens = min(max_tokens, remaining_budget)

    model.eval(tokens)
    trace_data = []
    generated_tokens = []
    stop_tokens = stop_tokens or [model.token_eos()]

    while len(generated_tokens) < max_tokens:
        logits = model.scores[model.n_tokens - 1]

        selected_id, selected_prob = sample_token(
            logits, temperature, top_k, top_p, rng, logit_processors=None, prev_tokens=generated_tokens
        )

        token_energy = energy_gate.energy(logits, generated_tokens, token_id=selected_id)

        if should_trigger_search(token_energy, energy_threshold):
            token_str = model.detokenize([selected_id]).decode("utf-8", errors="replace")
            print(f"\n🛑 [ENERGY SPIKE DETECTED] Token: {token_str!r} | Energy: {token_energy:.4f} > {energy_threshold}")
            print("Pausing linear decoding, running EGALBS search...")

            search_result = search_engine.run(model, energy_gate, logits, generated_tokens)
            start_position = model.n_tokens
            winning_tokens = search_result["winning_tokens"]

            if winning_tokens:
                model.eval(winning_tokens)

            for i, winning_id in enumerate(winning_tokens):
                entry = {
                    "token_position": start_position + i,
                    "cumulative_tokens": len(generated_tokens) + 1,
                    "selected_token_id": winning_id,
                    "selected_token_str": None,
                    "selected_token_prob": None,
                    "energy": search_result["winning_token_energies"][i],
                    "source": "search",
                }
                trace_data.append(entry)
                generated_tokens.append(winning_id)

                if winning_id in stop_tokens or len(generated_tokens) >= max_tokens:
                    break

            print(
                f"Resuming linear decoding after {len(winning_tokens)} "
                f"search-injected tokens (winning energy: {search_result['winning_energy']:.4f})\n"
            )

            if generated_tokens and generated_tokens[-1] in stop_tokens:
                break
            continue

        entry = {
            "token_position": model.n_tokens,
            "cumulative_tokens": len(generated_tokens) + 1,
            "selected_token_id": selected_id,
            "selected_token_str": None,
            "selected_token_prob": selected_prob,
            "energy": token_energy,
            "source": "linear"
        }
        trace_data.append(entry)
        generated_tokens.append(selected_id)

        if selected_id in stop_tokens:
            break

        model.eval([selected_id])

    generated_text = model.detokenize(generated_tokens).decode("utf-8", errors="replace")
    
    for entry in trace_data:
        entry["selected_token_str"] = model.detokenize([entry["selected_token_id"]]).decode("utf-8", errors="replace")

    return generated_text, trace_data