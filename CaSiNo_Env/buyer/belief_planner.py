from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from CaSiNo_Env.buyer.base import ASTRABuyer
from CaSiNo_Env.environment.actions import NegotiationAction, action_format_instructions, parse_action
from CaSiNo_Env.environment.casino import ITEMS, complement_allocation, enumerate_allocations, score_allocation
from CaSiNo_Env.environment.episode import visible_history


def _softmax(scores: Mapping[str, float]) -> Dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def _numbers_near_item(text: str, item: str) -> float:
    low_text = text.lower()
    item_low = item.lower()
    score = 0.0
    for match in re.finditer(re.escape(item_low), low_text):
        window = low_text[max(0, match.start() - 70) : match.end() + 70]
        self_ref = any(marker in window for marker in ("i ", "i'd", "i'll", "my ", "me ", "we ", "our ", "i need", "i prefer"))
        other_ref = any(marker in window for marker in ("you ", "your ", "for you", "your health", "you need"))
        if other_ref and not self_ref:
            continue
        for kw, delta in [
            ("need", 0.45),
            ("important", 0.40),
            ("prefer", 0.35),
            ("want", 0.25),
            ("must", 0.45),
            ("keep", 0.25),
            ("take", 0.20),
            ("less", -0.25),
            ("not important", -0.45),
            ("give up", -0.35),
            ("give you", -0.25),
        ]:
            if kw in window:
                score += delta if self_ref else 0.35 * delta
    return score


class ASTRABeliefPlannerBuyer(ASTRABuyer):
    name = "astra_style_belief_planner"

    def infer_belief(self, scenario: Any, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence = []
        item_scores = {item: 0.0 for item in ITEMS}
        offer_pressure = []
        for turn in history:
            action = turn.get("action", {})
            actor = turn.get("actor")
            message = str(action.get("message") or "")
            if actor == "seller":
                for item in ITEMS:
                    item_scores[item] += _numbers_near_item(message, item)
                allocation = action.get("allocation")
                if action.get("type") == "offer" and allocation:
                    seller_score_proxy = sum(int(allocation.get(item, 0)) for item in ITEMS)
                    offer_pressure.append(seller_score_proxy)
                    for item in ITEMS:
                        item_scores[item] += 0.45 * int(allocation.get(item, 0))
                if message:
                    evidence.append(message[:220])

        posterior = _softmax(item_scores)
        sorted_items = sorted(posterior, key=posterior.get, reverse=True)
        confidence = max(posterior.values()) - min(posterior.values()) if posterior else 0.0
        stance = "unknown"
        if len(offer_pressure) >= 2:
            stance = "generous" if offer_pressure[-1] < offer_pressure[0] else "stubborn"
        elif offer_pressure:
            stance = "anchoring"
        return {
            "partner_issue_priority_posterior": {k: round(v, 4) for k, v in posterior.items()},
            "partner_priority_rank": sorted_items,
            "confidence": round(confidence, 4),
            "stance": stance,
            "evidence": evidence[-5:],
        }

    def candidate_offers(self, scenario: Any, belief: Dict[str, Any], round_id: int, max_rounds: int) -> List[Dict[str, Any]]:
        partner_probs = belief["partner_issue_priority_posterior"]
        partner_values = {item: 3.0 + 2.0 * float(partner_probs.get(item, 1 / 3)) for item in ITEMS}
        rows = []
        time_pressure = round_id / max(max_rounds, 1)
        fairness_weight = 0.18 + 0.34 * time_pressure
        for p1_alloc in enumerate_allocations(scenario.pool):
            p2_alloc = complement_allocation(p1_alloc, scenario.pool)
            self_score = score_allocation(p1_alloc, scenario.p1_values)
            partner_score_est = sum(float(p2_alloc[item]) * partner_values[item] for item in ITEMS)
            score_gap = abs(self_score - partner_score_est)
            accept_prob = 1 / (1 + math.exp(-(partner_score_est - 10.0) / 2.2))
            ev = accept_prob * (self_score - fairness_weight * score_gap)
            rows.append(
                {
                    "allocation": p1_alloc,
                    "partner_gets": p2_alloc,
                    "self_score": round(self_score, 3),
                    "estimated_partner_score": round(partner_score_est, 3),
                    "score_gap": round(score_gap, 3),
                    "accept_prob": round(accept_prob, 4),
                    "planner_value": round(ev, 4),
                }
            )
        rows.sort(key=lambda row: (row["planner_value"], row["self_score"]), reverse=True)
        return rows[: self.candidate_top_k]

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        belief = self.infer_belief(scenario, history)
        candidates = self.candidate_offers(scenario, belief, round_id, max_rounds)
        prompt = f"""You are participant P1 in a CaSiNo camping negotiation using an ASTRA-style belief/planner wrapper.

{scenario.public_context_for_p1()}

Round: {round_id}/{max_rounds}
Dialogue so far:
{visible_history(history)}

Current belief about P2:
{json.dumps(belief, ensure_ascii=False, indent=2)}

Candidate offers. allocation means how many units YOU keep; P2 receives the complement:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Decision rules:
- In the first third of the negotiation, ask at most one direct priority question but still anchor with a concrete offer.
- Use the candidate table as advice, not as a hard constraint.
- Prefer high self-score offers early; increase estimated partner score gradually when P2 resists.
- Accept P2's offer only if it gives you a strong score or the final round is near.
- Avoid walk-away before the final rounds unless P2 explicitly refuses all feasible compromise.

{action_format_instructions("buyer/P1")}
"""
        raw = self.model.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        action = parse_action(raw)
        if action.type == "ask_preference" and round_id > max(2, max_rounds // 3) and candidates:
            action = NegotiationAction(
                type="offer",
                message=action.message or "I can move toward a split that respects both of our priorities.",
                allocation=candidates[0]["allocation"],
                raw_text=raw,
                parse_error="late_ask_preference_converted_to_offer",
            )
        return action, {"prompt_type": self.name, "belief": belief, "candidates": candidates, "raw": raw}
