"""LLM candidate planner scored by a Bayesian seller-reservation posterior."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from experiments.agenticpay_framework.components import json_from_text
from experiments.agenticpay_framework.schemas import StrategicPlan
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    RLVRScenario,
    format_transcript,
    last_price,
)
from experiments.model_clients import ModelClient

from ..belief_model.base import BeliefState, clamp
from ..belief_model.posterior import ReservationPosterior
from .base import CandidateAction, PlannerResult


@dataclass
class BayesianCandidateDiagnostics:
    """Diagnostics for candidate scoring under the posterior."""

    candidate_source: str
    confidence_override: str
    exploration_weight: float
    deal_weight: float
    posterior_mean: Optional[float]
    posterior_p10: Optional[float]
    posterior_p50: Optional[float]
    posterior_p90: Optional[float]
    posterior_confidence: float
    lowest_seller_offer: Optional[float]
    last_buyer_offer: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BayesianLLMCandidatePlanner:
    """Ask the LLM planner for candidate actions, then score them mathematically.

    This keeps the LLM strategic planner in the loop while making the seller
    reservation belief an explicit posterior object. Early rounds intentionally
    value low-probability anchors because they gather information and create a
    negotiation reference point. Later rounds increasingly value acceptance
    probability and closing EV.
    """

    def __init__(self, client: ModelClient, *, candidate_k: int = 5, max_tokens: int = 1400):
        self.client = client
        self.candidate_k = max(2, int(candidate_k))
        self.max_tokens = max_tokens
        self.last_raw = ""

    def plan(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        posterior: ReservationPosterior,
        round_id: int,
        max_turns: int,
    ) -> PlannerResult:
        llm_candidates = self._llm_candidates(
            scenario=scenario,
            history=history,
            belief=belief,
            round_id=round_id,
            max_turns=max_turns,
        )
        if not llm_candidates:
            llm_candidates = self._fallback_candidate_specs(scenario, history, belief, round_id, max_turns)

        scored: List[CandidateAction] = []
        for spec in llm_candidates:
            candidate = self._score_candidate(
                spec=spec,
                scenario=scenario,
                history=history,
                belief=belief,
                posterior=posterior,
                round_id=round_id,
                max_turns=max_turns,
            )
            if candidate is not None:
                scored.append(candidate)

        if not scored:
            for spec in self._fallback_candidate_specs(scenario, history, belief, round_id, max_turns):
                candidate = self._score_candidate(
                    spec=spec,
                    scenario=scenario,
                    history=history,
                    belief=belief,
                    posterior=posterior,
                    round_id=round_id,
                    max_turns=max_turns,
                )
                if candidate is not None:
                    scored.append(candidate)

        selected = max(scored, key=lambda x: x.ev)
        plan = self._candidate_to_plan(selected, scenario, history, belief)
        diagnostics = BayesianCandidateDiagnostics(
            candidate_source="llm_with_rule_fallback",
            confidence_override=(
                "early_anchor_allowed_even_if_p_accept_low"
                if self._early_information_round(round_id, max_turns, belief)
                else "posterior_acceptance_prob_used_normally"
            ),
            exploration_weight=self._exploration_weight(round_id, max_turns, belief),
            deal_weight=self._deal_weight(round_id, max_turns, belief),
            posterior_mean=belief.reservation.mean,
            posterior_p10=belief.reservation.p10,
            posterior_p50=belief.reservation.p50,
            posterior_p90=belief.reservation.p90,
            posterior_confidence=belief.reservation.confidence,
            lowest_seller_offer=self._lowest_seller_offer(history),
            last_buyer_offer=last_price(history, "buyer"),
        ).to_dict()
        return PlannerResult(
            plan=plan,
            candidates=sorted(scored, key=lambda x: x.ev, reverse=True),
            selected=selected,
            planner_type="bayesian_llm_candidate_planner",
            diagnostics={
                **diagnostics,
                "planner_raw": self.last_raw,
                "candidate_specs": llm_candidates,
            },
        )

    def _llm_candidates(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> List[Dict[str, Any]]:
        prompt = f"""You are the strategic planner for a buyer in an RLVR negotiation.

Generate diverse candidate actions. Do not write the final buyer message.

Private buyer budget: {scenario.buyer_budget:.2f}
Reference/list price: {scenario.reference_price:.2f}
Product codename: {scenario.codename_or_default()}
Quantity: {scenario.quantity}
Round: {round_id}/{max_turns}

Persistent Bayesian seller-reservation belief:
{json.dumps(self._compact_belief_for_prompt(belief), ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Planning policy:
- Keep the buyer budget as a hard limit, never as a target.
- In rounds 1-2, include at least one aggressive low anchor. It is fine if it
  has low immediate acceptance probability; it can extract information and
  pressure the seller.
- In early rounds, do not quit and do not accept merely because the seller price
  is under budget.
- Generate concrete [BUY] counteroffers unless accepting is clearly superior.
- If the seller made a formal quote, any [BUY] candidate must be below the
  lowest seller quote so far. [DEAL] may only copy an exact seller quote.
- Prefer slow concessions: do not jump toward the budget.
- Include one safer closing candidate only in late rounds or if seller quit risk
  seems high.

Return only valid JSON:
{{
  "candidates": [
    {{
      "action_type": "offer|accept|reject|quit",
      "price": <number or null>,
      "rationale": "short private reason",
      "language_style": "firm_anchor|firm_counter|tactical_finality|polite_close|walkaway"
    }}
  ]
}}
"""
        self.last_raw = self.client.generate(prompt, temperature=0.3, top_p=1.0, max_tokens=self.max_tokens)
        try:
            parsed = json_from_text(self.last_raw)
        except Exception:
            return []
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list):
            return []
        clean = []
        for item in candidates[: max(2, self.candidate_k + 2)]:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("action_type") or "").strip().lower()
            if action_type not in {"offer", "accept", "reject", "quit"}:
                continue
            clean.append(item)
        return clean

    @staticmethod
    def _compact_belief_for_prompt(belief: BeliefState) -> Dict[str, Any]:
        data = belief.to_dict()
        return {
            "item_id": data.get("item_id"),
            "round_id": data.get("round_id"),
            "buyer_budget": data.get("buyer_budget"),
            "reference_price": data.get("reference_price"),
            "reservation": data.get("reservation"),
            "acceptance_curve": [
                {
                    "price": point.get("price"),
                    "p_accept": point.get("p_accept"),
                    "confidence": point.get("confidence"),
                }
                for point in (data.get("acceptance_curve") or [])[:8]
                if isinstance(point, dict)
            ],
            "concession": data.get("concession"),
            "interaction": data.get("interaction"),
            "recent_evidence": [
                {
                    "round_id": item.get("round_id"),
                    "source": item.get("source"),
                    "type": item.get("evidence_type"),
                    "effect": item.get("effect"),
                    "confidence": item.get("confidence"),
                }
                for item in (data.get("evidence") or [])[-4:]
                if isinstance(item, dict)
            ],
            "posterior_entropy": (data.get("metadata") or {}).get("posterior_entropy"),
        }

    def _fallback_candidate_specs(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> List[Dict[str, Any]]:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        seller_lowest = self._lowest_seller_offer(history)
        cap = min(budget, seller_lowest * 0.995 if seller_lowest else budget)
        last_buyer = last_price(history, "buyer")
        specs: List[Dict[str, Any]] = []
        if round_id <= 2:
            ratios = [0.34, 0.42, 0.50, 0.58]
        elif round_id <= max_turns - 2:
            ratios = [0.48, 0.56, 0.64, 0.72]
        else:
            ratios = [0.60, 0.70, 0.80, 0.90]
        for ratio in ratios:
            price = min(cap, max(0.01, budget * ratio, ref * min(ratio, 0.55)))
            if last_buyer is not None:
                price = max(price, float(last_buyer) + budget * (0.015 if round_id <= 3 else 0.03))
            price = min(cap, price)
            specs.append(
                {
                    "action_type": "offer",
                    "price": round(price, 2),
                    "rationale": "fallback anchor/concession candidate",
                    "language_style": "firm_counter",
                }
            )
        if seller_lowest is not None and seller_lowest <= budget and round_id >= max_turns - 1:
            specs.append(
                {
                    "action_type": "accept",
                    "price": round(seller_lowest, 2),
                    "rationale": "late budget-safe seller quote",
                    "language_style": "polite_close",
                }
            )
        return specs

    def _score_candidate(
        self,
        *,
        spec: Dict[str, Any],
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        posterior: ReservationPosterior,
        round_id: int,
        max_turns: int,
    ) -> Optional[CandidateAction]:
        action_type = str(spec.get("action_type") or "").strip().lower()
        price = self._float_or_none(spec.get("price"))
        if action_type == "accept":
            seller_lowest = self._lowest_seller_offer(history)
            if seller_lowest is None or seller_lowest > scenario.buyer_budget:
                return None
            price = round(float(seller_lowest), 2)
            reward = self._reward_if_deal(price, scenario)
            remaining = max(0, max_turns - round_id)
            early_accept_penalty = 0.12 * remaining / max(1, max_turns)
            if remaining >= 2 and reward < 0.22:
                early_accept_penalty += 0.18
            return CandidateAction(
                action_type="accept",
                price=price,
                p_accept=1.0,
                p_quit=0.0,
                reward_if_deal=reward,
                future_value=0.0,
                ev=reward - early_accept_penalty,
                rationale=str(spec.get("rationale") or "accept exact seller quote"),
            )
        if action_type == "reject":
            remaining = max(0, max_turns - round_id)
            ev = 0.03 * remaining / max(1, max_turns) - 0.03
            if round_id <= 2:
                ev += 0.03
            return CandidateAction(
                action_type="reject",
                price=None,
                p_accept=0.0,
                p_quit=clamp(belief.interaction.quit_risk + 0.05),
                ev=ev,
                rationale=str(spec.get("rationale") or "reject and wait for more information"),
            )
        if action_type == "quit":
            return CandidateAction(
                action_type="quit",
                price=None,
                p_accept=0.0,
                p_quit=1.0,
                ev=-0.35 if round_id < max_turns else -0.05,
                rationale=str(spec.get("rationale") or "quit has low option value"),
            )
        if action_type != "offer" or price is None:
            return None
        price = self._sanitize_offer_price(price, scenario, history)
        if price is None:
            return None

        p_accept = clamp(posterior.p_accept(price))
        reward = self._reward_if_deal(price, scenario)
        round_pressure = clamp((round_id - 1) / max(1, max_turns - 1))
        exploration_weight = self._exploration_weight(round_id, max_turns, belief)
        deal_weight = self._deal_weight(round_id, max_turns, belief)
        information_value = self._information_value(price, scenario, belief, posterior)
        lowball_risk = self._lowball_risk(price, scenario, belief)
        p_quit = self._p_quit(price, scenario, belief, p_accept, lowball_risk, round_pressure)
        future_value = self._future_value(price, scenario, belief, round_id, max_turns)
        continue_prob = max(0.0, 1.0 - p_accept - p_quit)

        # Early rounds deliberately do not let low P_accept dominate. The
        # anchor can still be valuable if it preserves surplus and reduces
        # posterior uncertainty after the seller responds.
        ev = (
            deal_weight * p_accept * reward
            + exploration_weight * information_value
            + continue_prob * future_value
            - p_quit * (0.20 + 0.15 * round_pressure)
            - lowball_risk * (0.04 + 0.18 * round_pressure)
        )
        return CandidateAction(
            action_type="offer",
            price=round(float(price), 2),
            p_accept=p_accept,
            p_quit=p_quit,
            reward_if_deal=reward,
            future_value=future_value,
            ev=ev,
            rationale=str(spec.get("rationale") or "posterior-scored buyer offer"),
        )

    def _sanitize_offer_price(
        self,
        price: float,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
    ) -> Optional[float]:
        budget = float(scenario.buyer_budget)
        seller_lowest = self._lowest_seller_offer(history)
        cap = budget
        if seller_lowest is not None:
            cap = min(cap, float(seller_lowest) * 0.995)
        price = min(float(price), cap)
        last_buyer = last_price(history, "buyer")
        if last_buyer is not None:
            price = max(price, float(last_buyer) + 0.01 * budget)
        if price <= 0 or price > budget + 1e-9:
            return None
        return round(price, 2)

    def _candidate_to_plan(
        self,
        candidate: CandidateAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
    ) -> StrategicPlan:
        if candidate.action_type == "accept":
            strategic_act = "accept"
            concession = "none"
        elif candidate.action_type == "reject":
            strategic_act = "ask_info"
            concession = "none"
        elif candidate.action_type == "quit":
            strategic_act = "walk_away"
            concession = "none"
        else:
            strategic_act = "anchor_low" if last_price(history, "buyer") is None else "concede_small"
            concession = "none" if strategic_act == "anchor_low" else "small"
        return StrategicPlan(
            strategic_act=strategic_act,
            target_price=candidate.price,
            reservation_guardrail=float(scenario.buyer_budget),
            concession_size=concession,
            accept_if_at_or_below=float(scenario.buyer_budget),
            contract_priorities=["price"],
            feasibility_checks=[
                "price <= buyer budget",
                "buyer [BUY] counter must stay below seller lowest formal quote",
                "early low-acceptance anchors are allowed for information gathering",
                "final action selected from posterior-scored candidates",
            ],
            seller_feasible_price_floor=belief.reservation.p10,
            seller_feasibility_risk=clamp(1.0 - candidate.p_accept if candidate.price is not None else 1.0),
            seller_feasibility_guardrails=[
                "do not treat posterior as a hard truth when confidence is low",
                "in rounds 1-2, prioritize anchoring and information extraction",
                "in later rounds, balance expected buyer reward with seller quit risk",
                "accept only exact previous seller offers and only with meaningful surplus",
            ],
            language_style="firm",
            private_rationale=(
                f"Bayesian candidate advisor selected {candidate.action_type} at {candidate.price}; "
                f"p_accept={candidate.p_accept:.3f}, p_quit={candidate.p_quit:.3f}, ev={candidate.ev:.3f}. "
                f"{candidate.rationale}"
            ),
        )

    def _exploration_weight(self, round_id: int, max_turns: int, belief: BeliefState) -> float:
        if round_id <= 2 or belief.reservation.confidence < 0.38:
            return 0.55
        return max(0.08, 0.35 * (1.0 - clamp((round_id - 1) / max(1, max_turns - 1))))

    def _deal_weight(self, round_id: int, max_turns: int, belief: BeliefState) -> float:
        if round_id <= 2 and belief.reservation.confidence < 0.45:
            return 0.35
        return 0.55 + 0.45 * clamp((round_id - 1) / max(1, max_turns - 1))

    def _information_value(
        self,
        price: float,
        scenario: RLVRScenario,
        belief: BeliefState,
        posterior: ReservationPosterior,
    ) -> float:
        budget = max(float(scenario.buyer_budget), 1.0)
        surplus = self._reward_if_deal(price, scenario)
        uncertainty = 1.0 - clamp(belief.reservation.confidence)
        # Values around p10/p50 reveal useful reservation information; very
        # high offers are less informative because seller may simply accept.
        p10 = belief.reservation.p10 or posterior.quantile(0.10)
        p50 = belief.reservation.p50 or posterior.quantile(0.50)
        proximity = 1.0 - min(1.0, abs(float(price) - float(p10)) / budget)
        if price >= float(p50):
            proximity *= 0.75
        return clamp(0.55 * surplus + 0.45 * uncertainty * proximity)

    @staticmethod
    def _reward_if_deal(price: float, scenario: RLVRScenario) -> float:
        if price > scenario.buyer_budget:
            return -1.0
        return max(0.0, (float(scenario.buyer_budget) - float(price)) / max(float(scenario.buyer_budget), 1.0))

    def _future_value(
        self,
        price: float,
        scenario: RLVRScenario,
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> float:
        remaining = max(0, max_turns - round_id)
        if remaining <= 0:
            return 0.0
        patience = clamp(belief.interaction.patience)
        flexibility = clamp(0.35 + belief.concession.slope * 2.0)
        surplus = self._reward_if_deal(price, scenario)
        return 0.38 * surplus * max(0.30, patience) * max(0.35, flexibility) * remaining / max(1, max_turns)

    @staticmethod
    def _lowball_risk(price: float, scenario: RLVRScenario, belief: BeliefState) -> float:
        p10 = belief.reservation.p10
        if p10 is None:
            return 0.10
        gap = (float(p10) - float(price)) / max(float(scenario.buyer_budget), 1.0)
        return clamp(gap * 2.2)

    @staticmethod
    def _p_quit(
        price: float,
        scenario: RLVRScenario,
        belief: BeliefState,
        p_accept: float,
        lowball_risk: float,
        round_pressure: float,
    ) -> float:
        return clamp(
            0.50 * belief.interaction.quit_risk
            + 0.30 * lowball_risk
            + 0.15 * round_pressure * (1.0 - p_accept),
            high=0.85,
        )

    @staticmethod
    def _lowest_seller_offer(history: Sequence[Dict[str, Any]]) -> Optional[float]:
        prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        return min(prices) if prices else None

    @staticmethod
    def _early_information_round(round_id: int, max_turns: int, belief: BeliefState) -> bool:
        return round_id <= 2 or belief.reservation.confidence < 0.38

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace("$", "").replace(",", "").strip())
            except ValueError:
                return None
        return None
