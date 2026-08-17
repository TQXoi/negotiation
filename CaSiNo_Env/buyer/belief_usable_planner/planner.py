from __future__ import annotations

from typing import Any, Dict, List

from CaSiNo_Env.buyer.belief_usable_planner.belief_model import MathPartnerBeliefModel, PartnerBeliefState
from CaSiNo_Env.environment.casino import complement_allocation, enumerate_allocations, score_allocation


class RatioCandidatePlanner:
    """Generate candidates under different self/opponent utility weights."""

    def __init__(self, *, candidate_top_k: int = 8):
        self.candidate_top_k = candidate_top_k
        self.weight_ratios = [
            ("aggressive_self", 0.92, 0.22),
            ("self_leaning", 0.78, 0.32),
            ("balanced_tradeoff", 0.62, 0.42),
            ("cooperative", 0.48, 0.52),
            ("high_accept", 0.34, 0.62),
        ]

    def generate_candidates(
        self,
        scenario: Any,
        belief: PartnerBeliefState,
        belief_model: MathPartnerBeliefModel,
        *,
        round_id: int,
        max_rounds: int,
    ) -> List[Dict[str, Any]]:
        partner_values = belief.estimated_partner_values()
        p1_max = max(score_allocation(scenario.pool, scenario.p1_values), 1)
        p2_max = max(sum(float(scenario.pool[item]) * float(partner_values[item]) for item in scenario.pool), 1e-6)
        raw_candidates: List[Dict[str, Any]] = []
        for label, self_weight, min_opp_norm in self.weight_ratios:
            best = None
            for p1_alloc in enumerate_allocations(scenario.pool):
                p2_alloc = complement_allocation(p1_alloc, scenario.pool)
                self_norm = score_allocation(p1_alloc, scenario.p1_values) / p1_max
                opp_norm = sum(float(p2_alloc[item]) * float(partner_values[item]) for item in p2_alloc) / p2_max
                if opp_norm < min_opp_norm:
                    continue
                time_ratio = round_id / max(max_rounds, 1)
                exploration_bonus = 0.06 * (1 - time_ratio) * abs(self_norm - opp_norm)
                weighted_value = self_weight * self_norm + (1 - self_weight) * opp_norm + exploration_bonus
                row = {
                    "candidate_type": label,
                    "self_weight": round(self_weight, 3),
                    "opponent_weight": round(1 - self_weight, 3),
                    "allocation": p1_alloc,
                    "planner_weighted_value": round(weighted_value, 4),
                }
                if best is None or row["planner_weighted_value"] > best["planner_weighted_value"]:
                    best = row
            if best is not None:
                raw_candidates.append(best)

        # Deduplicate same allocation while preserving the best label.
        dedup: Dict[tuple, Dict[str, Any]] = {}
        for candidate in raw_candidates:
            key = tuple(candidate["allocation"][item] for item in scenario.pool)
            if key not in dedup or candidate["planner_weighted_value"] > dedup[key]["planner_weighted_value"]:
                dedup[key] = candidate

        scored = [
            belief_model.score_candidate(scenario, belief, candidate, round_id=round_id, max_rounds=max_rounds)
            for candidate in dedup.values()
        ]
        scored.sort(key=lambda row: (row["belief_score"], row["self_score"]), reverse=True)
        return scored[: self.candidate_top_k]

    def choose_final(self, candidates: List[Dict[str, Any]], *, round_id: int, max_rounds: int) -> Dict[str, Any]:
        if not candidates:
            return {"type": "walk_away", "message": "No feasible candidate.", "allocation": None}
        time_ratio = round_id / max(max_rounds, 1)
        if time_ratio > 0.78:
            viable = [c for c in candidates if c["accept_prob"] >= 0.45]
            viable.sort(key=lambda row: (row["belief_score"], row["accept_prob"]), reverse=True)
            return viable[0] if viable else candidates[0]
        return candidates[0]
