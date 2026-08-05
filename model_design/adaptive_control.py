import math
from collections import deque
import numpy as np

def observe(logits: np.ndarray, top_k: int = 5) -> dict:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / exp.sum()
 
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(logits)))
 
    top_two = np.partition(logits, -2)[-2:]
    margin = float(np.max(top_two) - np.min(top_two))
 
    k = min(top_k, probs.shape[0])
    top_k_idx = np.argpartition(probs, -k)[-k:]
    concentration = float(probs[top_k_idx].sum())
 
    return {"entropy": entropy, "margin": margin, "concentration": concentration}

def zscore(hist: deque, value: float) -> float:
    if len(hist) < 2:
        return 0.0
    arr = np.fromiter(hist, dtype=np.float64)
    sigma = float(arr.std(ddof=1))
    if sigma < 1e-8:
        return 0.0
    return float((value - arr.mean()) / sigma)
 
 
def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

class ObserverTracker:
    # Rolling window, independent of AdaptiveThresholdTracker in energy.py.
    # window defaults to 64 to match EnergyProcessor's repetition_window,
    # just to avoid a new constant -- no state is shared between them.
    def __init__(self, window: int = 64):
        self.window = window
        self.entropy_hist = deque(maxlen=window)
        self.margin_hist = deque(maxlen=window)
        self.concentration_hist = deque(maxlen=window)
 
    def update(self, obs: dict) -> dict:
        self.entropy_hist.append(obs["entropy"])
        self.margin_hist.append(obs["margin"])
        self.concentration_hist.append(obs["concentration"])
 
        return {
            "entropy": obs["entropy"],
            "entropy_zscore": zscore(self.entropy_hist, obs["entropy"]),
            "margin": obs["margin"],
            "margin_zscore": zscore(self.margin_hist, obs["margin"]),
            "concentration": obs["concentration"],
            "concentration_zscore": zscore(self.concentration_hist, obs["concentration"]),
        }
 
    @property
    def warmed_up(self) -> bool:
        return len(self.entropy_hist) >= 2
 
class DecoderController:
    # temperature = 2 * Phi(entropy_zscore), no free constants.
    # z = 0 -> T = 1.0 (identity). z -> -inf -> T -> 0. z -> +inf -> T -> 2.
    def policy(self, state: dict, warmed_up: bool, top_p: float, top_k: int) -> dict:
        temperature = 2.0 * phi(state["entropy_zscore"]) if warmed_up else 1.0
        return {"temperature": temperature, "top_p": top_p, "top_k": top_k}