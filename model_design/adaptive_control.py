import math
import numpy as np

def observe(logits: np.ndarray, min_p: float = 0.05) -> dict:
    # Numerically stable softmax
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    # Full-vocabulary normalized entropy
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(logits)))

    # Logit margin (Top 1 vs Top 2 gap)
    top_two = np.partition(logits, -2)[-2:]
    margin = float(top_two[1] - top_two[0])

    # --- Native Min-P Candidate Metrics ---
    p_max = float(probs.max())
    cutoff = min_p * p_max
    
    active_mask = probs >= cutoff
    min_p_count = int(np.sum(active_mask))
    min_p_mass = float(probs[active_mask].sum())

    return {
        "entropy": entropy,
        "margin": margin,
        "p_max": p_max,
        "min_p_count": min_p_count,
        "min_p_mass": min_p_mass,
    }

class ObserverTracker:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.entropy_mean = 0.0
        self.entropy_var = 0.0
        self.count = 0

    def update(self, obs: dict) -> dict:
        x = obs["entropy"]
        self.count += 1

        if self.count == 1:
            self.entropy_mean = x
            self.entropy_var = 1e-4  # Seed with small non-zero variance
        else:
            # Incremental EMA updates for mean and variance
            delta = x - self.entropy_mean
            self.entropy_mean += self.alpha * delta
            self.entropy_var = (1.0 - self.alpha) * (self.entropy_var + self.alpha * (delta ** 2))

        # Safe standard deviation calculation
        sigma = math.sqrt(max(self.entropy_var, 1e-8))
        z_score = (x - self.entropy_mean) / sigma if sigma > 1e-4 else 0.0

        return {
            "entropy": x,
            "entropy_zscore": z_score,
            "margin": obs["margin"],
            "p_max": obs["p_max"],
            "min_p_count": obs["min_p_count"],
            "min_p_mass": obs["min_p_mass"],
        }

    @property
    def warmed_up(self) -> bool:
        return self.count >= 2

def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

class DecoderController:
    def policy(self, state: dict, warmed_up: bool) -> dict:
        # z = 0 -> T = 1.0; z -> -inf -> T -> 0.0; z -> +inf -> T -> 2.0
        temperature = 2.0 * phi(state["entropy_zscore"]) if warmed_up else 1.0
        return {"temperature": temperature}