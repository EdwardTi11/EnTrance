import numpy as np

def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())

class EnergyProcessor:
    _cached_cost: np.ndarray | None = None
    _cached_vocab_size: int | None = None

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
            if (
                EnergyProcessor._cached_cost is not None 
                and EnergyProcessor._cached_vocab_size == self.vocab_size
            ):
                self.cost = EnergyProcessor._cached_cost
            else:
                build_cost = cost_fn or self.default_cost
                self.cost = build_cost(self.vocab_size, model)
                EnergyProcessor._cached_cost = self.cost
                EnergyProcessor._cached_vocab_size = self.vocab_size
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
            logp_token = shifted[token_id] - np.log(np.exp(shifted).sum())
            
            recent = prev_tokens[-self.repetition_window:] if prev_tokens else []
            r_token = float(recent.count(token_id))
            
            return float(-self.alpha * logp_token + self.beta * self.cost[token_id] + self.gamma * r_token)

        logp = log_softmax(logits)
        r = self.repetition_fn(self.vocab_size, prev_tokens or [], self.repetition_window)
        return -self.alpha * logp + self.beta * self.cost + self.gamma * r