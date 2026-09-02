import numpy as np
from jinja2 import Template
from llama_cpp import Llama
from model_design.adaptive_control import observe, ObserverTracker

def topk_softmax(logits, k):
    k = min(k, len(logits))
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    probs = np.exp(logits[idx] - logits[idx[0]])
    return idx, probs / probs.sum()

def sample_token(logits, temperature, top_k, top_p, rng):
    idx, probs = topk_softmax(logits / max(temperature, 1e-6), top_k)

    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    idx, probs = idx[:cutoff], probs[:cutoff]
    probs /= probs.sum()

    choice = rng.choice(len(idx), p=probs)
    return int(idx[choice]), float(probs[choice])

def generate_text(
    model: Llama,
    prompt: str,
    max_tokens=None,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop_tokens=None,
    seed=None,
    decoder_controller=None,
):
    rng = np.random.default_rng(seed)
    tracker = ObserverTracker() if decoder_controller else None

    messages = [{"role": "user", "content": prompt}]

    tmpl = model.metadata.get("tokenizer.chat_template")
    if tmpl:
        # Ensure template is converted from bytes to string if needed
        tmpl_str = tmpl.decode("utf-8") if isinstance(tmpl, bytes) else tmpl
        formatted_prompt = Template(tmpl_str).render(messages=messages, add_generation_prompt=True)
    else:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


    tokens = model.tokenize(formatted_prompt.encode())
    budget = model.n_ctx() - len(tokens) - 4
    max_tokens = budget if not max_tokens or max_tokens <= 0 else min(max_tokens, budget)

    model.eval(tokens)
    generated, trace = [], []
    stop_tokens = stop_tokens or [model.token_eos()]

    while len(generated) < max_tokens:
        logits = model.scores[model.n_tokens - 1]

        if decoder_controller:
            assert tracker is not None
            state = tracker.update(observe(logits))
            policy = decoder_controller.policy(
                state, tracker.warmed_up, top_p, top_k
            )
            temp, step_p, step_k = (
                policy["temperature"],
                policy["top_p"],
                policy["top_k"],
            )
        else:
            temp, step_p, step_k = temperature, top_p, top_k

        token_id, prob = sample_token(
            logits, temp, step_k, step_p, rng
        )

        trace.append({
            "token_position": model.n_tokens,
            "cumulative_tokens": len(generated) + 1,
            "selected_token_id": token_id,
            "selected_token_prob": prob,
            "temperature_used": temp,
        })

        generated.append(token_id)

        if token_id in stop_tokens:
            break

        model.eval([token_id])

    text = model.detokenize(generated).decode("utf-8", errors="replace")

    for entry in trace:
        entry["selected_token_str"] = model.detokenize(
            [entry["selected_token_id"]]
        ).decode("utf-8", errors="replace")

    return text, trace