from typing import cast, Dict, Any

from llama_cpp import Llama, LogitsProcessorList
from model_design.adaptive_control import observe, ObserverTracker

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
    tracker = ObserverTracker()
    trace = []

    def adaptive_logits_processor(input_ids, logits):
        if decoder_controller:
            # 1. Compute observation and state update using your adaptive control module
            obs = observe(logits)
            state = tracker.update(obs)
            policy = decoder_controller.policy(state, tracker.warmed_up, top_p, top_k)
            
            temp = policy["temperature"]
            
            # 2. Scale logits in-place before llama.cpp runs top-k/top-p filtering
            if temp > 0 and temp != 1.0:
                logits /= max(temp, 1e-6)

            # 3. Record trace metrics
            trace.append({
                "cumulative_tokens": len(input_ids),
                "entropy": state.get("entropy"),
                "entropy_zscore": state.get("entropy_zscore"),
                "margin": state.get("margin"),
                "margin_zscore": state.get("margin_zscore"),
                "concentration": state.get("concentration"),
                "concentration_zscore": state.get("concentration_zscore"),
                "temperature_used": temp,
            })
            
        return logits

    logits_processors = LogitsProcessorList([adaptive_logits_processor]) if decoder_controller else None

    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=1.0 if decoder_controller else temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop_tokens or ["</think>", "<|im_end|>", "</s>"],
        seed=seed,
        logits_processor=logits_processors,
        stream=False,
    )

    response_dict = cast(Dict[str, Any], response)
    text = response_dict["choices"][0]["message"]["content"]
    return text, trace