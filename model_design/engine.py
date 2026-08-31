import numpy as np
from llama_cpp import Llama
from model_design.energy import AdaptiveThresholdTracker
from model_design.adaptive_control import observe, ObserverTracker, DecoderController

def topk_softmax(logits: np.ndarray, k: int):
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    top_logits = logits[idx]
    order = np.argsort(top_logits)[::-1]
    idx = idx[order]
    top_logits = top_logits[order]
    shifted = top_logits - top_logits[0]
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
    k_multiplier: float = 1.5,
    max_tokens: int | None = None,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    stop_tokens: list[int] | None = None,
    seed: int | None = None,
    decoder_controller: DecoderController | None = None,
    acl_window: int | None = None,
):
    rng = np.random.default_rng(seed)

    threshold_tracker = AdaptiveThresholdTracker(k_multiplier=k_multiplier)
    observer_tracker = ObserverTracker(window=acl_window) if decoder_controller else None

    messages = [{"role": "user", "content": prompt}]

    try:
        formatted_prompt = model.chat_format_handler(messages=messages)["prompt"]
    except Exception:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    tokens = model.tokenize(formatted_prompt.encode("utf-8"))
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

        if decoder_controller is not None:
            obs = observe(logits)
            entropy_state = observer_tracker.update(obs)
            policy = decoder_controller.policy(entropy_state, observer_tracker.warmed_up, top_p, top_k)
            step_temperature = policy["temperature"]
            step_top_p = policy["top_p"]
            step_top_k = policy["top_k"]
        else:
            step_temperature = temperature
            step_top_p = top_p
            step_top_k = top_k

        selected_id, selected_prob = sample_token(
            logits, step_temperature, step_top_k, step_top_p, rng, logit_processors=None, prev_tokens=generated_tokens
        )
        token_energy = energy_gate.energy(logits, generated_tokens, token_id=selected_id)

        # Dynamic threshold from running energy statistics (mu + k * sigma),
        # kept for observability; decoding itself is fully adaptive-controller driven.
        current_threshold = threshold_tracker.update_and_get_threshold(token_energy)

        entry = {
            "token_position": model.n_tokens,
            "cumulative_tokens": len(generated_tokens) + 1,
            "selected_token_id": selected_id,
            "selected_token_str": None,
            "selected_token_prob": selected_prob,
            "energy": token_energy,
            "threshold_used": current_threshold,
            "source": "linear",
            "temperature_used": step_temperature,
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