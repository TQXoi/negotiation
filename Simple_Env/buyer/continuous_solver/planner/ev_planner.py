"""Expected-value planner over candidate offers under persistent belief."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from experiments.agenticpay_framework.schemas import StrategicPlan
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    RLVRScenario,
    last_price,
)

from ..belief_model.base import BeliefState, clamp
from ..belief_model.posterior import ReservationPosterior
from .base import CandidateAction, PlannerResult


class EVPlanner:
    """Enumerate buyer actions and choose the highest expected value.

    This is the first continual-resolving planner: the belief model supplies a
    reservation posterior and scalar risks; the planner uses explicit math to
    choose among candidate offers / accept / reject / quit.
    """

    def __init__(
        self,
        *,
        candidate_k: int = 5,
        quit_penalty: float = 0.25,
        future_value_weight: float = 0.55,
        accept_probability_floor: float = 0.08,
        accept_opportunity_cost_weight: float = 0.90,
    ):
        self.candidate_k = candidate_k
        self.quit_penalty = quit_penalty
        self.future_value_weight = future_value_weight
        self.accept_probability_floor = accept_probability_floor
        self.accept_opportunity_cost_weight = accept_opportunity_cost_weight

    def candidate_prices(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> List[float]:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        last_buyer = last_price(history, "buyer")
        seller_lowest = self._lowest_seller_offer(history)
        pressure = self._seller_pressure(history, belief)
        cap = self._aggressive_offer_cap(scenario, history, round_id, max_turns)
        if seller_lowest is not None:
            # Buyer counteroffers should never exceed the seller's best formal
            # quote so far. Accepting that exact quote is handled separately.
            cap = min(cap, seller_lowest * 0.995)
        floor = max(0.01, min(budget * 0.18, ref * 0.16))
        if last_buyer is not None:
            # Slow concessions: do not jump toward the budget. Keep the next
            # offer near the previous buyer offer unless late-round pressure
            # makes a larger step unavoidable.
            min_increment = budget * (0.015 if round_id <= 3 else 0.025)
            floor = max(floor, last_buyer + min_increment)
        floor = min(floor, cap)
        cap = max(floor, min(cap, budget))

        if round_id <= 1:
            prices = {
                min(cap, max(floor, budget * 0.32)),
                min(cap, max(floor, budget * 0.40)),
                min(cap, max(floor, budget * 0.48)),
                min(cap, max(floor, ref * 0.35)),
            }
        else:
            prices = {
                floor,
                min(cap, max(floor, budget * (0.42 + 0.055 * (round_id - 1)))),
                min(cap, max(floor, budget * (0.48 + 0.060 * (round_id - 1)))),
            }
        if pressure >= 2:
            # Rescue candidates: after repeated seller resistance, keep pressure
            # but stop repeating offers that are clearly below the inferred
            # feasible region. These are still below the seller's best quote.
            if belief.reservation.p10 is not None:
                prices.add(min(cap, max(floor, float(belief.reservation.p10) * 1.03)))
            if belief.reservation.p50 is not None:
                center = float(belief.reservation.p50)
                prices.add(min(cap, max(floor, center * 0.92)))
                prices.add(min(cap, max(floor, center * 0.98)))
                prices.add(min(cap, max(floor, center * 1.03)))
            if belief.reservation.mean is not None:
                prices.add(min(cap, max(floor, float(belief.reservation.mean) * 0.96)))
            if seller_lowest is not None:
                prices.add(min(cap, max(floor, seller_lowest * 0.78)))
                prices.add(min(cap, max(floor, seller_lowest * 0.84)))
                prices.add(min(cap, max(floor, seller_lowest * 0.89)))
        if belief.reservation.p10 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p10) * 0.98)))
        if belief.reservation.p50 is not None:
            center = float(belief.reservation.p50)
            prices.add(min(cap, max(floor, center * 0.88)))
            prices.add(min(cap, max(floor, center * 0.94)))
            prices.add(min(cap, max(floor, center)))
            prices.add(min(cap, max(floor, center * 1.04)))
        if belief.reservation.mean is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.mean) * 1.02)))
        if belief.reservation.p90 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p90) * 0.72)))
        if last_buyer is not None:
            prices.add(min(cap, max(floor, last_buyer + 0.025 * budget)))
        last_seller = last_price(history, "seller")
        if last_seller is not None:
            for ratio in [0.64, 0.72, 0.80, 0.86]:
                prices.add(min(cap, max(floor, last_seller * ratio)))
        # Keep a focused set around the posterior center. The seller's first
        # quote is often a high anchor, so do not automatically retain the
        # highest generated candidate when K is small.
        clean = sorted({round(max(0.01, min(budget, p)), 2) for p in prices})
        if len(clean) <= max(1, self.candidate_k):
            return clean
        selected = set()
        # Preserve one bargaining-pressure candidate below the posterior center.
        center_target = float(belief.reservation.p50 or belief.reservation.mean or clean[len(clean) // 2])
        lower_target = max(floor, center_target * (0.90 if round_id <= 3 else 0.94))
        selected.add(min(clean, key=lambda p: abs(p - lower_target)))
        # Add prices closest to posterior median/mean and a modest seller-anchor
        # discount. This tends to choose feasible offers without capitulating to
        # the seller's initial ask.
        targets = [center_target, center_target * 1.04]
        if belief.reservation.mean is not None:
            targets.append(float(belief.reservation.mean) * 0.98)
            targets.append(float(belief.reservation.mean) * 1.02)
        if last_seller is not None:
            targets.extend([last_seller * 0.78, last_seller * 0.84])
        for target in targets:
            if len(selected) >= self.candidate_k:
                break
            selected.add(min(clean, key=lambda p: abs(p - target)))
        # Fill any remaining slots from the center outward, not from the lowest
        # upward and not from the highest downward.
        for price in sorted(clean, key=lambda p: abs(p - center_target)):
            if len(selected) >= self.candidate_k:
                break
            selected.add(price)
        return sorted(selected)

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
        candidates = []
        for price in self.candidate_prices(
            scenario=scenario,
            history=history,
            belief=belief,
            round_id=round_id,
            max_turns=max_turns,
        ):
            candidates.append(self._score_offer(price, scenario, belief, posterior, round_id, max_turns))

        last_seller = last_price(history, "seller")
        if (
            last_seller is not None
            and last_seller <= scenario.buyer_budget + 1e-9
            and self._should_consider_accept(
                scenario=scenario,
                history=history,
                belief=belief,
                last_seller=last_seller,
                round_id=round_id,
                max_turns=max_turns,
                offer_candidates=candidates,
            )
        ):
            reward = self._reward_if_deal(last_seller, scenario)
            best_counter_ev = max((c.ev for c in candidates if c.action_type == "offer"), default=0.0)
            remaining = max_turns - round_id
            early_accept_penalty = 0.0
            if remaining >= 2:
                early_accept_penalty = self.accept_opportunity_cost_weight * max(0.0, best_counter_ev - reward)
                # Even when the current EV formula slightly prefers accepting,
                # keep pressure on the seller before late rounds unless the
                # seller offer is already very good.
                early_accept_penalty += 0.03 * remaining / max_turns
            candidates.append(
                CandidateAction(
                    action_type="accept",
                    price=round(float(last_seller), 2),
                    p_accept=1.0,
                    p_quit=0.0,
                    reward_if_deal=reward,
                    ev=reward - early_accept_penalty,
                    rationale="accept current budget-safe seller offer minus opportunity cost of further bargaining",
                )
            )

        # Quit is explicit but strongly discouraged before the final round when
        # a budget-safe counteroffer still exists. In RLVR, no deal gets 0, but
        # early quit destroys option value and made the initial solver brittle.
        remaining = max_turns - round_id
        quit_ev = -self.quit_penalty * (1.0 - belief.interaction.quit_risk)
        if remaining >= 2:
            quit_ev -= 0.20 + 0.05 * remaining
        best_offer_ev = max((c.ev for c in candidates if c.action_type == "offer"), default=-1.0)
        if best_offer_ev > -0.05:
            quit_ev -= 0.15
        candidates.append(
            CandidateAction(
                action_type="quit",
                price=None,
                ev=quit_ev,
                rationale="walk away only if all offers have poor expected value",
            )
        )

        selected = max(candidates, key=lambda x: x.ev)
        plan = self._candidate_to_plan(selected, scenario, belief)
        return PlannerResult(
            plan=plan,
            candidates=candidates,
            selected=selected,
            planner_type="ev_continual_resolving_planner",
            diagnostics={
                "round": round_id,
                "max_turns": max_turns,
                "belief_reservation": belief.reservation.to_dict(),
                "interaction": belief.interaction.to_dict(),
            },
        )

    def _score_offer(
        self,
        price: float,
        scenario: RLVRScenario,
        belief: BeliefState,
        posterior: ReservationPosterior,
        round_id: int,
        max_turns: int,
    ) -> CandidateAction:
        p_accept = clamp(posterior.p_accept(price))
        low_accept_risk = max(0.0, self.accept_probability_floor - p_accept)
        round_pressure = clamp(round_id / max(1, max_turns))
        reservation_gap = 0.0
        if belief.reservation.p10 is not None and price < float(belief.reservation.p10):
            reservation_gap = (float(belief.reservation.p10) - price) / max(float(scenario.buyer_budget), 1.0)
        p_quit = clamp(
            0.55 * belief.interaction.quit_risk
            + low_accept_risk
            + 0.08 * round_pressure * (1.0 - p_accept),
            + 0.45 * reservation_gap,
            high=0.80,
        )
        reward = self._reward_if_deal(price, scenario)
        future_value = self._future_value(price, scenario, belief, round_id, max_turns)
        continue_prob = max(0.0, 1.0 - p_accept - p_quit)
        lowball_penalty = 0.20 * reservation_gap * (1.0 + round_pressure)
        ev = p_accept * reward + continue_prob * future_value - p_quit * self.quit_penalty - lowball_penalty
        return CandidateAction(
            action_type="offer",
            price=round(float(price), 2),
            p_accept=p_accept,
            p_quit=p_quit,
            reward_if_deal=reward,
            future_value=future_value,
            ev=ev,
            rationale="EV = p_accept * buyer_surplus + continue_value - quit_risk_penalty",
        )

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
        concession_bonus = clamp(belief.concession.slope * 2.0)
        patience = clamp(belief.interaction.patience)
        surplus = self._reward_if_deal(price, scenario)
        option_value = 0.04 * remaining / max_turns
        return (
            self.future_value_weight
            * (surplus + option_value)
            * max(0.35, patience)
            * (0.55 + 0.45 * concession_bonus)
            * remaining
            / max_turns
        )

    @staticmethod
    def _candidate_to_plan(candidate: CandidateAction, scenario: RLVRScenario, belief: BeliefState) -> StrategicPlan:
        if candidate.action_type == "accept":
            strategic_act = "accept"
            concession = "none"
        elif candidate.action_type == "quit":
            strategic_act = "walk_away"
            concession = "none"
        else:
            strategic_act = "concede_small" if belief.concession.num_price_drops > 0 else "anchor_low"
            concession = "small" if belief.concession.num_price_drops > 0 else "none"
        return StrategicPlan(
            strategic_act=strategic_act,
            target_price=candidate.price,
            reservation_guardrail=float(scenario.buyer_budget),
            concession_size=concession,
            accept_if_at_or_below=float(scenario.buyer_budget),
            contract_priorities=["price"],
            feasibility_checks=["price <= buyer budget", "expected value selected under persistent belief"],
            seller_feasible_price_floor=belief.reservation.p10,
            seller_feasibility_risk=clamp(1.0 - candidate.p_accept if candidate.price is not None else 1.0),
            seller_feasibility_guardrails=[
                "do not treat verbal finality as certain unless behavior supports it",
                "prefer counteroffers over early quit when posterior overlap exists",
                "treat buyer budget as a limit, not a target",
                "use anchor-and-concession: start aggressively low and concede slowly",
                "never counter above the seller's lowest formal offer so far",
                "do not accept a seller offer early unless it is already a strong surplus deal",
                "preserve bargaining pressure with tactical finality language without revealing the real budget",
            ],
            language_style="firm",
            private_rationale=(
                candidate.rationale
                + " Use an aggressive anchor-and-slow-concession posture: keep the offer concrete, persuasive, below budget, and below the seller's best quote."
            ),
        )

    @staticmethod
    def _seller_offer_prices(history: Sequence[Dict[str, Any]]) -> List[float]:
        prices = []
        for turn in history:
            if turn.get("role") != "seller":
                continue
            action = turn.get("action") or {}
            if action.get("price") is not None:
                prices.append(float(action["price"]))
        return prices

    def _lowest_seller_offer(self, history: Sequence[Dict[str, Any]]) -> float | None:
        prices = self._seller_offer_prices(history)
        return min(prices) if prices else None

    @staticmethod
    def _aggressive_offer_cap(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> float:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        # Slower than the paper-aligned concession schedule. This keeps the
        # buyer from drifting toward its budget after only one or two counters.
        ratios = [0.48, 0.56, 0.66, 0.78, 0.88, 0.96]
        idx = min(max(round_id - 1, 0), len(ratios) - 1)
        # Use budget-ratio as the controlling cap. The test construction often
        # sets budget ~= 0.8 * reference price; a reference-ratio cap would keep
        # late offers below the seller's true cost and trigger seller quits.
        cap = budget * ratios[idx]
        if round_id <= 1:
            cap = min(cap, ref * 0.36)
        if round_id >= max_turns:
            cap = min(budget, max(cap, budget * 0.96))
        return max(0.01, cap)

    def _should_consider_accept(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        last_seller: float,
        round_id: int,
        max_turns: int,
        offer_candidates: Sequence[CandidateAction],
    ) -> bool:
        budget = float(scenario.buyer_budget)
        seller_ratio = float(last_seller) / max(budget, 1.0)
        reward = self._reward_if_deal(last_seller, scenario)
        best_counter_ev = max((c.ev for c in offer_candidates if c.action_type == "offer"), default=-1.0)
        remaining = max_turns - round_id
        if last_seller > budget + 1e-9:
            return False
        if remaining >= 2 and seller_ratio > 0.82:
            return False
        if remaining >= 1 and seller_ratio > 0.90:
            return False
        if reward < 0.12 and round_id < max_turns:
            return False
        if best_counter_ev > reward - 0.04 and round_id < max_turns:
            return False
        if belief.interaction.finality_prob >= 0.80 and round_id >= max_turns - 1 and reward >= 0.08:
            return True
        return round_id >= max_turns or reward >= 0.18

    @staticmethod
    def _seller_pressure(history: Sequence[Dict[str, Any]], belief: BeliefState) -> int:
        seller_rejects = 0
        seller_offers = 0
        for turn in history:
            if turn.get("role") != "seller":
                continue
            action = (turn.get("action") or {}).get("action")
            if action == "reject":
                seller_rejects += 1
            elif action in {"offer", "sell"}:
                seller_offers += 1
        return seller_rejects + max(0, seller_offers - belief.concession.num_price_drops)
