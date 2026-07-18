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
        original_n_tokens = model.n_tokens
        candidates = top_k_candidates(logits, self.beam_width)

        beams = []
        best_winning_energy = float('inf')  
        search_forward_passes = 0  # Combined Metric: Counts active model evaluations

        for candidate in candidates:
            model.n_tokens = original_n_tokens

            beam_tokens = []
            token_energies = []
            local_history = list(prev_tokens)
            cumulative_energy = 0.0
            next_token = candidate
            step_logits = logits

            for _ in range(self.lookahead_depth):
                search_forward_passes += 1  # Track every speculative forward pass
                
                token_energy = energy_gate.energy(step_logits, local_history, token_id=next_token)
                cumulative_energy += token_energy

                if cumulative_energy >= best_winning_energy:
                    break

                beam_tokens.append(next_token)
                token_energies.append(token_energy)
                local_history.append(next_token)
                
                model.eval([next_token])

                if next_token == model.token_eos():
                    break

                step_logits = model.scores[model.n_tokens - 1]
                next_token = self.token_selector(step_logits)

            if cumulative_energy < best_winning_energy:
                beams.append((beam_tokens, cumulative_energy, token_energies))
                best_winning_energy = cumulative_energy

        model.n_tokens = original_n_tokens

        if not beams:
            winning_tokens = [candidates[0]]
            winning_energy = 0.0
            winning_token_energies = [0.0]
        else:
            winning_tokens, winning_energy, winning_token_energies = min(beams, key=lambda beam: beam[1])

        return {
            "winning_tokens": winning_tokens,
            "winning_energy": winning_energy,
            "winning_token_energies": winning_token_energies,
            "beam_energies": [energy for _, energy, _ in beams],
            "search_forward_passes": search_forward_passes  # Clear computational workload
        }