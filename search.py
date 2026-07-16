import numpy as np
 
def greedy_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits))
 
def top_k_candidates(logits: np.ndarray, k: int) -> list[int]:
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return idx.tolist()
 
def should_trigger_search(current_energy: float, threshold: float) -> bool:
    return current_energy > threshold
 
class EGALBSSearch:
    def __init__(self, beam_width: int = 4, lookahead_depth: int = 8, token_selector=None):
        self.beam_width = beam_width
        self.lookahead_depth = lookahead_depth
        self.token_selector = token_selector or greedy_token
 
    def run(self, model, energy_gate, logits: np.ndarray, prev_tokens: list[int]) -> dict:
        pause_state = model.save_state()
        candidates = top_k_candidates(logits, self.beam_width)
 
        beams = []
        for candidate in candidates:
            model.load_state(pause_state)
 
            beam_tokens = []
            token_energies = []
            local_history = list(prev_tokens)
            cumulative_energy = 0.0
            next_token = candidate
            step_logits = logits
 
            for step in range(self.lookahead_depth):
                token_energy = float(energy_gate.energy(step_logits, local_history)[next_token])
                cumulative_energy += token_energy
                beam_tokens.append(next_token)
                token_energies.append(token_energy)
                local_history.append(next_token)
                model.eval([next_token])
 
                if next_token == model.token_eos():
                    break
 
                step_logits = np.array(model.scores[model.n_tokens - 1, :], dtype=np.float32)
                next_token = self.token_selector(step_logits)
 
            beams.append((beam_tokens, cumulative_energy, token_energies))
 
        winning_tokens, winning_energy, winning_token_energies = min(beams, key=lambda beam: beam[1])

        model.load_state(pause_state)
        model.eval(winning_tokens)
 
        return {
            "winning_tokens": winning_tokens,
            "winning_energy": winning_energy,
            "winning_token_energies": winning_token_energies,
            "beam_energies": [energy for _, energy, _ in beams],
        }