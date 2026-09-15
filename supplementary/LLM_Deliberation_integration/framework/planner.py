from __future__ import annotations

from typing import Any, Dict, List

from astra_integration.tools.deals import enumerate_deals, normalized_score, score_deal


class MultiPartyCandidatePlanner:
    def __init__(self, candidate_top_k: int = 8):
        self.candidate_top_k = candidate_top_k

    def generate(self, game: Any, focal: str, beliefs: Dict[str, Any], turn_id: int, max_turns: int) -> List[Dict[str, Any]]:
        rows = []
        for deal in enumerate_deals(game.issues):
            self_norm = normalized_score(game.players[focal].scores, deal)
            acceptance = {}
            uncertainty = []
            for opponent, belief in beliefs.items():
                posteriors = belief.option_posteriors()
                estimated = sum(posteriors[issue][deal[issue] - 1] for issue in game.issues) / len(game.issues)
                acceptance[opponent] = max(0.02, min(0.98, 0.20 + estimated + 0.25 * belief.flexibility - 0.25 * belief.threshold_ratio))
                uncertainty.append(sum(1.0 - max(values) for values in posteriors.values()) / len(posteriors))
            # The final coalition needs focal P1, veto player P2, and enough other
            # parties to reach n-1 supporters. Use the weakest required member as
            # a conservative coalition probability rather than a population mean.
            other_probs = sorted(
                (prob for opponent, prob in acceptance.items() if opponent != game.p2), reverse=True
            )
            other_needed = max(len(game.players) - 3, 0)
            coalition_floor = other_probs[other_needed - 1] if other_needed else 1.0
            accept_prob = min(acceptance.get(game.p2, 0.0), coalition_floor)
            time_ratio = turn_id / max(max_turns, 1)
            info_gain = (sum(uncertainty) / max(len(uncertainty), 1)) * (1.0 - time_ratio)
            belief_score = 0.58 * self_norm + 0.34 * accept_prob + 0.08 * info_gain
            if score_deal(game.players[focal].scores, deal) < game.players[focal].threshold:
                belief_score -= 1.0
            rows.append({
                "deal": deal,
                "self_score": score_deal(game.players[focal].scores, deal),
                "self_norm": round(self_norm, 5),
                "accept_prob": round(accept_prob, 5),
                "info_gain": round(info_gain, 5),
                "belief_score": round(belief_score, 5),
            })
        rows.sort(key=lambda row: (row["belief_score"], row["self_score"]), reverse=True)
        return rows[: self.candidate_top_k]

    def choose(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {"action_type": "abstain", "deal": None}
        return {**candidates[0], "action_type": "propose"}
