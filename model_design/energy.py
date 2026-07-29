import numpy as np

class AdaptiveThresholdTracker:
    def __init__(self, k_multiplier: float = 2.5):
        self.k = k_multiplier
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_stats(self, energy_value: float):
        self.count += 1
        delta = energy_value - self.mean
        self.mean += delta / self.count
        delta2 = energy_value - self.mean
        self.M2 += delta * delta2

    def update_and_get_threshold(self, current_energy: float, warmup_steps: int = 10) -> float:
        self.update_stats(current_energy)

        if self.count < warmup_steps:
            return 5.2913
        
        variance = self.M2 / (self.count - 1) if self.count > 1 else 0.0
        sigma = np.sqrt(max(variance, 1e-6))

        # Dynamic cutoff threshold
        threshold = self.mean + (self.k * sigma)

        return threshold

class EnergyProcessor:
    def __init__(
        self,
        model,
        alpha: float = 1.0,
        gamma: float = 0.0,
        repetition_window: int = 64,
    ):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.repetition_window = repetition_window
        self.vocab_size = (
            model.n_vocab() if hasattr(model, "n_vocab") else len(model.scores[0])
        )
        
    def energy(self, logits: np.ndarray, prev_tokens: list[int], token_id: int) -> float:
        shifted = logits - np.max(logits)
        sum_exp = np.exp(shifted).sum()
        logp_token = shifted[token_id] - np.log(sum_exp)

        recent = prev_tokens[-self.repetition_window:] if prev_tokens else []
        r_token = float(recent.count(token_id))

        return float(-self.alpha * logp_token + self.gamma * r_token)