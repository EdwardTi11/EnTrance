import numpy as np

def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())

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
        beta: float = 0.0,
        gamma: float = 0.0,
        repetition_window: int = 64,
        cost_fn=None,
        repetition_fn=None,
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.repetition_window = repetition_window
        self.repetition_fn = repetition_fn or self.default_repetition

        self.vocab_size = (
            model.n_vocab() if hasattr(model, "n_vocab") else len(model.scores[0])
        )

        if beta != 0.0:
            build_cost = cost_fn or self.default_cost
            self.cost = build_cost(self.vocab_size, model)
        else:
            self.cost = np.zeros(self.vocab_size, dtype=np.float32)

    def default_cost(self, vocab_size: int, model) -> np.ndarray:
        cost = np.empty(vocab_size, dtype=np.float32)
        for tid in range(vocab_size):
            try:
                length = max(len(model.detokenize([tid])), 1)
            except Exception:
                length = 1
            cost[tid] = 1.0 / length
        return cost

    def default_repetition(
        self, vocab_size: int, prev_tokens: list[int], window: int
    ) -> np.ndarray:
        if not prev_tokens:
            return np.zeros(vocab_size, dtype=np.float32)
        recent = prev_tokens[-window:]
        counts = np.bincount(recent, minlength=vocab_size).astype(np.float32)
        return counts[:vocab_size]

    def energy(self, logits: np.ndarray, prev_tokens: list[int], token_id: int | None = None) -> float | np.ndarray:
        if token_id is not None:
            shifted = logits - np.max(logits)
            sum_exp = np.exp(shifted).sum()
            logp_token = shifted[token_id] - np.log(sum_exp)
            
            recent = prev_tokens[-self.repetition_window:] if prev_tokens else []
            r_token = float(recent.count(token_id))
            
            return float(-self.alpha * logp_token + self.beta * self.cost[token_id] + self.gamma * r_token)

        logp = log_softmax(logits)
        r = self.repetition_fn(self.vocab_size, prev_tokens or [], self.repetition_window)
        return -self.alpha * logp + self.beta * self.cost + self.gamma * r