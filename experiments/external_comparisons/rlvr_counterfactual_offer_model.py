"""Counterfactual offer-response model and planner for RLVR negotiation.

The response model predicts how the seller would respond to a small set of
candidate buyer offers.  The planner then chooses an offer by expected buyer
surplus under the current concession schedule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from experiments.agenticpay_framework.components import json_from_text
from experiments.agenticpay_framework.schemas import BeliefState, StrategicPlan
from experiments.model_clients import ModelClient


@dataclass
class OfferResponseCandidate:
    price: float
    accept_prob: float
    seller_counter_price: Optional[float] = None
    counter_price_distribution: Optional[Dict[str, float]] = None
    seller_walkaway_prob: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualOfferResponseBelief:
    seller_reservation_range: Optional[List[float]]
    seller_reservation_confidence: float
    likely_acceptable_price_range: Optional[List[float]]
    walkaway_risk_next_turn: float
    candidates: List[OfferResponseCandidate] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    raw_model_output: str = ""
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data

    def to_belief_state(self) -> BeliefState:
        best_accept = max((c.accept_prob for c in self.candidates), default=0.5)
        evidence_strings = []
        for item in self.evidence:
            if isinstance(item, dict):
                quote = str(item.get("quote") or "").strip()
                supports = str(item.get("supports") or "").strip()
                evidence_strings.append(f"{quote} -> {supports}".strip(" ->"))
            else:
                evidence_strings.append(str(item))
        if not evidence_strings:
            evidence_strings = ["Counterfactual response model produced no explicit evidence."]
        return BeliefState(
            seller_reservation_range=self.seller_reservation_range,
            seller_reservation_confidence=self.seller_reservation_confidence,
            seller_reservation_estimate=_mid(self.seller_reservation_range),
            seller_flexibility_range=None,
            seller_patience_range=None,
            seller_strategy="counterfactual_offer_response_model",
            deal_risk=max(0.0, min(1.0, self.walkaway_risk_next_turn * (1.0 - 0.25 * best_accept))),
            likely_acceptable_price_range=self.likely_acceptable_price_range,
            contract_term_preferences={
                "counterfactual_offer_response_curve": [c.to_dict() for c in self.candidates],
                "response_model_used_fallback": self.used_fallback,
            },
            evidence=evidence_strings,
        )


class CounterfactualOfferResponseModel:
    def __init__(self, model: ModelClient, max_tokens: int = 1200, candidate_top_k: int = 3):
        self.model = model
        self.max_tokens = max_tokens
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.last_raw = ""

    def predict(
        self,
        *,
        scenario: Any,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        schedule: Dict[str, Any],
        candidate_prices: Optional[Sequence[float]] = None,
    ) -> CounterfactualOfferResponseBelief:
        candidate_prices = (
            self._clean_candidate_prices(candidate_prices, scenario, schedule)
            if candidate_prices is not None
            else self._candidate_prices(scenario, history, schedule)
        )
        prompt = f"""You are a counterfactual seller-response model for an RLVR buyer.

Do not generate a buyer message. Predict how the seller would respond to each candidate buyer offer.

Private buyer budget: {scenario.buyer_budget:.2f}
Reference/list price: {scenario.reference_price:.2f}
Current round: {round_id}/{max_turns}
Concession schedule:
{json.dumps(schedule, ensure_ascii=False)}

Candidate buyer offers to evaluate:
{json.dumps(candidate_prices, ensure_ascii=False)}

Conversation:
{format_history(history)}

Return only valid JSON:
{{
  "seller_reservation_range": {{"low": <number>, "high": <number>, "confidence": <0-1>}},
  "likely_acceptable_price_range": {{"low": <number>, "high": <number>}},
  "walkaway_risk_next_turn": <0-1>,
  "candidates": [
    {{
      "price": <one candidate price>,
      "accept_prob": <0-1>,
      "seller_counter_price": <number or null>,
      "counter_price_distribution": {{"low": <number>, "mid": <number>, "high": <number>}},
      "seller_walkaway_prob": <0-1>,
      "rationale": "brief evidence-based reason"
    }}
  ],
  "evidence": [
    {{"turn": <integer or null>, "quote": "short observed quote", "supports": "field supported"}}
  ]
}}

Rules:
- Seller cost/reservation is hidden; use uncertainty and evidence.
- Never assume the buyer budget equals seller reservation.
- Candidate accept probabilities should be monotonic-ish: higher prices usually have higher accept probability.
- A low accept probability is not automatically bad: low offers can be strategically useful if they elicit a counter.
- Include all candidate prices exactly once.
- If evidence is weak, keep confidence low."""
        self.last_raw = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
        try:
            parsed = json_from_text(self.last_raw)
            return self._parse(
                parsed,
                candidate_prices,
                raw=self.last_raw,
                fallback=False,
                scenario=scenario,
                history=history,
                schedule=schedule,
            )
        except Exception:
            return self._fallback(candidate_prices, scenario, history, schedule, raw=self.last_raw)

    def _candidate_prices(self, scenario: Any, history: Sequence[Dict[str, Any]], schedule: Dict[str, Any]) -> List[float]:
        budget = float(scenario.buyer_budget)
        target = float(schedule.get("target_offer", budget * 0.72))
        max_offer = float(schedule.get("max_offer", budget * 0.84))
        min_offer = float(schedule.get("min_offer", budget * 0.62))
        values = {
            min_offer,
            target,
            min(max_offer, target + budget * 0.03),
        }
        if len(history) >= 2:
            values.add(schedule.get("max_offer", budget * 0.84))
        last_seller = _last_price(history, "seller")
        if last_seller is not None:
            values.add(min(budget, float(schedule.get("max_offer", budget)), last_seller * 0.70))
            values.add(min(budget, float(schedule.get("max_offer", budget)), last_seller * 0.78))
            values.add(min(budget, float(schedule.get("max_offer", budget)), last_seller * 0.85))
        last_buyer = _last_price(history, "buyer")
        if last_buyer is not None:
            values.add(min(float(schedule.get("max_offer", budget)), last_buyer + max(0.01, budget * 0.025)))
        if schedule.get("round", 1) >= schedule.get("max_turns", 6) - 1:
            values.add(min(budget, float(schedule.get("accept_cap", budget))))
        prices = sorted(round(max(0.01, min(budget, float(v))), 2) for v in values if v is not None)
        deduped = []
        for price in prices:
            if price not in deduped:
                deduped.append(price)
        if len(deduped) <= self.candidate_top_k:
            return deduped
        anchors = [target]
        last_buyer = _last_price(history, "buyer")
        if last_buyer is not None:
            anchors.append(last_buyer + max(0.01, budget * 0.025))
        last_seller = _last_price(history, "seller")
        if last_seller is not None:
            anchors.extend([last_seller * 0.78, last_seller * 0.85])
        anchors.append(max_offer)
        selected: List[float] = []
        for anchor in anchors:
            nearest = min(deduped, key=lambda p: (abs(p - anchor), p))
            if nearest not in selected:
                selected.append(nearest)
            if len(selected) >= self.candidate_top_k:
                break
        for price in deduped:
            if len(selected) >= self.candidate_top_k:
                break
            if price not in selected:
                selected.append(price)
        return sorted(selected)

    def _clean_candidate_prices(
        self,
        candidate_prices: Sequence[float],
        scenario: Any,
        schedule: Dict[str, Any],
    ) -> List[float]:
        budget = float(scenario.buyer_budget)
        cap = min(budget, float(schedule.get("max_offer", budget)))
        cleaned = []
        for price in candidate_prices:
            try:
                value = round(max(0.01, min(cap, float(price))), 2)
            except (TypeError, ValueError):
                continue
            if value not in cleaned:
                cleaned.append(value)
        return sorted(cleaned)

    def _parse(
        self,
        parsed: Dict[str, Any],
        candidate_prices: Sequence[float],
        *,
        raw: str,
        fallback: bool,
        scenario: Any,
        history: Sequence[Dict[str, Any]],
        schedule: Dict[str, Any],
    ) -> CounterfactualOfferResponseBelief:
        reservation = _range_object(parsed.get("seller_reservation_range"))
        reservation_conf = _bounded_float(
            parsed.get("seller_reservation_range", {}).get("confidence")
            if isinstance(parsed.get("seller_reservation_range"), dict)
            else parsed.get("seller_reservation_confidence"),
            0.3,
        )
        acceptable = _range_object(parsed.get("likely_acceptable_price_range"))
        raw_candidates = parsed.get("candidates") if isinstance(parsed.get("candidates"), list) else []
        by_price: Dict[float, OfferResponseCandidate] = {}
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            price = _float_or_none(item.get("price"))
            if price is None:
                continue
            nearest = min(candidate_prices, key=lambda p: abs(p - price)) if candidate_prices else round(price, 2)
            if abs(nearest - price) > max(1.0, float(scenario.buyer_budget) * 0.025):
                continue
            by_price[round(nearest, 2)] = OfferResponseCandidate(
                price=round(nearest, 2),
                accept_prob=_bounded_float(item.get("accept_prob"), 0.5),
                seller_counter_price=_float_or_none(item.get("seller_counter_price")),
                counter_price_distribution=_counter_distribution(item.get("counter_price_distribution")),
                seller_walkaway_prob=_bounded_float(item.get("seller_walkaway_prob"), 0.1),
                rationale=str(item.get("rationale") or ""),
            )
        used_partial_fallback = len(by_price) < len(candidate_prices)
        if used_partial_fallback:
            fallback_belief = self._fallback(candidate_prices, scenario, history, schedule, raw=raw)
            for candidate in fallback_belief.candidates:
                by_price.setdefault(candidate.price, candidate)
        candidates = [by_price[round(price, 2)] for price in candidate_prices if round(price, 2) in by_price]
        evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else []
        return CounterfactualOfferResponseBelief(
            seller_reservation_range=reservation,
            seller_reservation_confidence=reservation_conf,
            likely_acceptable_price_range=acceptable,
            walkaway_risk_next_turn=_bounded_float(parsed.get("walkaway_risk_next_turn"), 0.4),
            candidates=candidates,
            evidence=evidence,
            raw_model_output=raw,
            used_fallback=fallback or used_partial_fallback,
        )

    def _fallback(
        self,
        candidate_prices: Sequence[float],
        scenario: Any,
        history: Sequence[Dict[str, Any]],
        schedule: Dict[str, Any],
        *,
        raw: str,
    ) -> CounterfactualOfferResponseBelief:
        last_seller = _last_price(history, "seller")
        if last_seller is None:
            center = min(float(scenario.buyer_budget), float(scenario.reference_price) * 0.70)
        else:
            center = min(float(scenario.buyer_budget), last_seller * 0.90)
        width = max(1.0, float(scenario.buyer_budget) * 0.08)
        candidates = []
        for price in candidate_prices:
            accept_prob = 1.0 / (1.0 + math.exp(-(price - center) / width))
            walkaway_prob = max(0.02, min(0.7, 0.35 - 0.25 * accept_prob))
            counter = None if accept_prob > 0.75 else min(float(scenario.buyer_budget), max(price, center))
            candidates.append(
                OfferResponseCandidate(
                    price=round(price, 2),
                    accept_prob=round(accept_prob, 3),
                    seller_counter_price=round(counter, 2) if counter is not None else None,
                    counter_price_distribution=(
                        {
                            "low": round(max(price, center - width), 2),
                            "mid": round(max(price, center), 2),
                            "high": round(min(float(scenario.buyer_budget), center + width), 2),
                        }
                        if counter is not None
                        else None
                    ),
                    seller_walkaway_prob=round(walkaway_prob, 3),
                    rationale="heuristic fallback from last seller offer and budget schedule",
                )
            )
        reservation = [round(max(0.01, center - width), 2), round(min(float(scenario.buyer_budget), center + width), 2)]
        return CounterfactualOfferResponseBelief(
            seller_reservation_range=reservation,
            seller_reservation_confidence=0.25,
            likely_acceptable_price_range=[round(min(candidate_prices), 2), round(max(candidate_prices), 2)] if candidate_prices else None,
            walkaway_risk_next_turn=0.35,
            candidates=candidates,
            evidence=[{"turn": None, "quote": "fallback model", "supports": "counterfactual curve"}],
            raw_model_output=raw,
            used_fallback=True,
        )


class CounterfactualOfferPlanner:
    def plan(
        self,
        *,
        scenario: Any,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        schedule: Dict[str, Any],
        response_belief: CounterfactualOfferResponseBelief,
    ) -> StrategicPlan:
        budget = float(scenario.buyer_budget)
        best = None
        best_score = -1e9
        scored = []
        for candidate in response_belief.candidates:
            if candidate.price > min(budget, float(schedule["max_offer"])) + 1e-9:
                continue
            surplus = budget - candidate.price
            counter_value = None
            if candidate.counter_price_distribution:
                counter_value = candidate.counter_price_distribution.get("mid")
            if counter_value is None:
                counter_value = candidate.seller_counter_price
            counter_surplus = max(0.0, budget - float(counter_value)) if counter_value is not None else 0.0
            counter_prob = max(0.0, 1.0 - candidate.accept_prob - candidate.seller_walkaway_prob)
            unnecessary_concession = max(0.0, candidate.price - float(schedule["target_offer"]))
            score = (
                1.35 * candidate.accept_prob * surplus
                + 0.45 * counter_prob * counter_surplus
                - 0.18 * candidate.seller_walkaway_prob * max(1.0, budget)
                - 0.12 * unnecessary_concession
            )
            if round_id <= 2:
                score -= 0.18 * unnecessary_concession
                if candidate.price >= float(schedule["max_offer"]) - 1e-9:
                    score -= 0.04 * budget
            scored.append({
                **candidate.to_dict(),
                "buyer_surplus_if_accepted": round(surplus, 4),
                "counter_surplus_mid": round(counter_surplus, 4),
                "counter_prob_proxy": round(counter_prob, 4),
                "expected_buyer_surplus": round(score, 4),
            })
            if score > best_score:
                best_score = score
                best = candidate
        if best is None:
            target = float(schedule["target_offer"])
        else:
            target = best.price
        return StrategicPlan(
            strategic_act="probe" if round_id <= 2 else ("concede_small" if round_id < max_turns - 1 else "concede_medium"),
            target_price=round(min(budget, target), 2),
            reservation_guardrail=budget,
            concession_size="small" if round_id <= 3 else "medium",
            accept_if_at_or_below=min(budget, float(schedule["accept_cap"])),
            feasibility_checks=[
                "price <= buyer budget",
                "target selected by counterfactual expected buyer surplus",
                "candidate price <= concession schedule max",
            ],
            seller_feasible_price_floor=(
                response_belief.likely_acceptable_price_range[0]
                if response_belief.likely_acceptable_price_range
                else None
            ),
            seller_feasibility_risk=response_belief.walkaway_risk_next_turn,
            seller_feasible_non_price_terms={
                "counterfactual_candidates": [c.to_dict() for c in response_belief.candidates],
                "scored_candidates": scored,
            },
            seller_feasibility_guardrails=[
                "do not exceed schedule max",
                "do not accept non-formal seller prices",
                "prefer high expected buyer surplus over raw deal probability",
                "early rounds should probe with low anchors and preserve concession room",
            ],
            language_style="friendly_firm",
            private_rationale=(
                f"Selected ${target:.2f} by counterfactual expected surplus; "
                f"best score {best_score:.3f}."
            ),
        )

    def rerank_full_framework_plan(
        self,
        *,
        scenario: Any,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        schedule: Dict[str, Any],
        response_belief: CounterfactualOfferResponseBelief,
        base_plan: StrategicPlan,
    ) -> StrategicPlan:
        """Rerank candidate offers while preserving the full framework plan.

        The counterfactual model is an advisory offer selector here. It may
        override the full-framework target only when its expected-reward score
        clearly improves or when it finds a lower offer with tolerable risk.
        """

        budget = float(scenario.buyer_budget)
        cap = min(budget, float(schedule["max_offer"]))
        base_target = _clamp_price(
            base_plan.target_price if base_plan.target_price is not None else schedule["target_offer"],
            lower=float(schedule["min_offer"]),
            upper=cap,
        )
        seller_final = _seller_finality_signal(history)
        last_seller = _last_price(history, "seller")
        late_round = round_id >= max_turns - 2
        final_round = round_id >= max_turns
        repeated_seller_floor = (
            last_seller is not None
            and _count_recent_seller_offers_near(history, last_seller, tolerance=max(0.5, 0.01 * budget)) >= 2
        )
        protected_formal_offer = (
            last_seller is not None
            and last_seller <= budget + 1e-9
            and (late_round or seller_final or repeated_seller_floor)
        )
        protected_floor = (
            max(float(schedule["min_offer"]), min(cap, last_seller * 0.98))
            if protected_formal_offer and last_seller is not None
            else None
        )

        scored = []
        best_candidate = None
        best_score = -1e9
        base_score = None

        for candidate in response_belief.candidates:
            if candidate.price > cap + 1e-9:
                continue
            surplus_norm = max(0.0, budget - candidate.price) / max(1.0, budget)
            counter_value = None
            if candidate.counter_price_distribution:
                counter_value = candidate.counter_price_distribution.get("mid")
            if counter_value is None:
                counter_value = candidate.seller_counter_price
            counter_surplus_norm = (
                max(0.0, budget - float(counter_value)) / max(1.0, budget)
                if counter_value is not None
                else 0.0
            )
            counter_prob = max(0.0, 1.0 - candidate.accept_prob - candidate.seller_walkaway_prob)
            concession_norm = max(0.0, candidate.price - base_target) / max(1.0, budget)
            formal_gap_norm = (
                max(0.0, float(last_seller) - candidate.price) / max(1.0, budget)
                if protected_formal_offer and last_seller is not None
                else 0.0
            )
            low_probe_bonus = 0.0
            if (
                candidate.price < base_target
                and candidate.accept_prob >= (0.35 if late_round else 0.18)
                and candidate.seller_walkaway_prob <= (0.30 if late_round else 0.50)
                and (protected_floor is None or candidate.price >= protected_floor - 1e-9)
            ):
                low_probe_bonus = min(0.08, (base_target - candidate.price) / max(1.0, budget))
            late_walkaway_penalty = 0.22 if final_round else (0.16 if late_round else 0.08)
            score = (
                candidate.accept_prob * surplus_norm
                + 0.22 * counter_prob * counter_surplus_norm
                + low_probe_bonus
                - late_walkaway_penalty * candidate.seller_walkaway_prob
                - 0.10 * concession_norm
                - (0.35 if protected_formal_offer else 0.0) * formal_gap_norm
            )
            if seller_final and last_seller is not None and candidate.price < last_seller * 0.95 and round_id >= 3:
                score -= 0.12
            if protected_floor is not None and candidate.price < protected_floor - 1e-9:
                score -= 0.20 + 0.40 * ((protected_floor - candidate.price) / max(1.0, budget))
            if candidate.price >= cap - 1e-9 and round_id < max_turns:
                score -= 0.04

            row = {
                **candidate.to_dict(),
                "buyer_surplus_norm": round(surplus_norm, 4),
                "counter_surplus_norm": round(counter_surplus_norm, 4),
                "counter_prob_proxy": round(counter_prob, 4),
                "concession_norm_over_base": round(concession_norm, 4),
                "formal_gap_norm": round(formal_gap_norm, 4),
                "protected_formal_offer": protected_formal_offer,
                "reranker_expected_reward_score": round(score, 6),
            }
            scored.append(row)

            if abs(candidate.price - base_target) <= max(0.5, 0.015 * budget):
                base_score = score if base_score is None else max(base_score, score)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        selected_price = base_target
        selected_reason = "kept full-framework target"
        if best_candidate is not None:
            best_price = best_candidate.price
            score_margin = best_score - (base_score if base_score is not None else -0.02)
            lower_with_tolerable_risk = (
                best_price < base_target
                and best_candidate.accept_prob >= (0.40 if late_round else 0.20)
                and best_candidate.seller_walkaway_prob <= (0.25 if late_round else 0.45)
                and (protected_floor is None or best_price >= protected_floor - 1e-9)
            )
            clear_expected_gain = score_margin >= (0.04 if late_round else 0.025)
            high_price_requires_evidence = (
                best_price <= base_target + 0.03 * budget
                or best_candidate.accept_prob >= 0.70
                or (seller_final and last_seller is not None and best_price >= last_seller * 0.95)
                or round_id >= max_turns
            )
            if lower_with_tolerable_risk or (clear_expected_gain and high_price_requires_evidence):
                selected_price = best_price
                selected_reason = (
                    "counterfactual reranker selected lower/tolerable-risk offer"
                    if lower_with_tolerable_risk
                    else "counterfactual reranker selected clear expected-reward improvement"
                )
        if protected_floor is not None and selected_price < protected_floor - 1e-9:
            protected_target = max(base_target, protected_floor)
            if last_seller is not None and seller_final and late_round:
                protected_target = max(protected_target, min(cap, last_seller))
            selected_price = min(cap, protected_target)
            selected_reason = "kept protected formal seller offer/floor instead of risky lower rerank"

        new_plan = StrategicPlan(**base_plan.to_dict())
        new_plan.target_price = round(min(budget, selected_price), 2)
        new_plan.reservation_guardrail = budget
        new_plan.accept_if_at_or_below = min(
            budget,
            base_plan.accept_if_at_or_below if base_plan.accept_if_at_or_below is not None else float(schedule["accept_cap"]),
            float(schedule["accept_cap"]),
        )
        new_plan.seller_feasibility_risk = response_belief.walkaway_risk_next_turn
        merged_terms = dict(new_plan.seller_feasible_non_price_terms or {})
        merged_terms["counterfactual_reranker"] = {
            "mode": "full_framework_plus_counterfactual_reranker",
            "base_target": round(base_target, 2),
            "selected_target": new_plan.target_price,
            "selection_reason": selected_reason,
            "base_score": round(base_score, 6) if base_score is not None else None,
            "best_score": round(best_score, 6) if best_candidate is not None else None,
            "late_round": late_round,
            "seller_final_signal": seller_final,
            "repeated_seller_floor": repeated_seller_floor,
            "protected_formal_offer": protected_formal_offer,
            "protected_floor": round(protected_floor, 2) if protected_floor is not None else None,
            "scored_candidates": scored,
        }
        new_plan.seller_feasible_non_price_terms = merged_terms
        new_plan.feasibility_checks = list(dict.fromkeys((new_plan.feasibility_checks or []) + [
            "counterfactual reranker is advisory over full-framework plan",
            "override requires expected-reward gain or lower tolerable-risk offer",
            "price <= buyer budget and schedule cap",
        ]))
        new_plan.seller_feasibility_guardrails = list(dict.fromkeys((new_plan.seller_feasibility_guardrails or []) + [
            "do not raise above full-framework target without evidence",
            "do not undercut a repeated/final budget-safe formal seller offer in late rounds",
            "prefer buyer surplus over raw deal probability",
            "fall back to full-framework target when response predictions are weak",
        ]))
        new_plan.private_rationale = (
            (new_plan.private_rationale + " " if new_plan.private_rationale else "")
            + f"Counterfactual reranker: {selected_reason}; base ${base_target:.2f}, selected ${new_plan.target_price:.2f}."
        )
        return new_plan


def format_history(history: Sequence[Dict[str, Any]]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for item in history:
        action = item.get("action") or {}
        lines.append(
            f"Round {item.get('round')} {item.get('role')}: "
            f"{action.get('action')} {action.get('price')} | {str(item.get('message', ''))[:500]}"
        )
    return "\n".join(lines)


def _last_price(history: Sequence[Dict[str, Any]], role: str) -> Optional[float]:
    for item in reversed(history):
        if item.get("role") == role:
            action = item.get("action") or {}
            price = action.get("price")
            if price is not None:
                return float(price)
    return None


def _count_recent_seller_offers_near(
    history: Sequence[Dict[str, Any]],
    price: float,
    *,
    tolerance: float,
    max_turns_back: int = 4,
) -> int:
    count = 0
    checked = 0
    for item in reversed(history):
        if item.get("role") != "seller":
            continue
        checked += 1
        action = item.get("action") or {}
        seller_price = action.get("price")
        if seller_price is not None and abs(float(seller_price) - price) <= tolerance:
            count += 1
        if checked >= max_turns_back:
            break
    return count


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _bounded_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _counter_distribution(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    parsed: Dict[str, float] = {}
    for key in ("low", "mid", "high"):
        number = _float_or_none(value.get(key))
        if number is None:
            return None
        parsed[key] = round(number, 2)
    low = min(parsed["low"], parsed["mid"], parsed["high"])
    high = max(parsed["low"], parsed["mid"], parsed["high"])
    mid = parsed["mid"]
    if mid < low or mid > high:
        mid = (low + high) / 2.0
    return {"low": round(low, 2), "mid": round(mid, 2), "high": round(high, 2)}


def _clamp_price(value: Any, *, lower: float, upper: float) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError):
        price = lower
    return round(max(lower, min(upper, price)), 2)


def _seller_finality_signal(history: Sequence[Dict[str, Any]]) -> bool:
    final_words = ("final", "lowest", "no lower", "bottom", "best i can", "take it or leave")
    for item in reversed(history):
        if item.get("role") != "seller":
            continue
        text = str(item.get("message") or "").lower()
        return any(word in text for word in final_words)
    return False


def _range_object(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        low, high = value.get("low"), value.get("high")
    elif isinstance(value, list) and len(value) == 2:
        low, high = value
    else:
        return None
    try:
        low_f, high_f = float(low), float(high)
    except (TypeError, ValueError):
        return None
    return [round(min(low_f, high_f), 2), round(max(low_f, high_f), 2)]


def _mid(value: Optional[List[float]]) -> Optional[float]:
    if not value:
        return None
    return round((float(value[0]) + float(value[1])) / 2.0, 2)
