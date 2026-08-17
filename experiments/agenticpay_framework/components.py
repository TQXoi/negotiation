"""Prompt-only components for a trainable AgenticPay buyer architecture."""

from __future__ import annotations

import json
import re
from itertools import product
from typing import Any, Dict, List, Optional

try:
    from agenticpay.agents.buyer_agent import BuyerAgent
except ImportError:  # Simple Env uses belief/planner components without AgenticPay.
    BuyerAgent = None  # type: ignore[assignment,misc]
from experiments.agenticpay_framework.schemas import (
    BeliefState,
    NaturalizationResult,
    StrategicPlan,
)
from experiments.model_clients import ModelClient


def format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No conversation yet."
    return "\n".join(
        f"[Round {msg.get('round', 0)}] {msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
        for msg in history
    )


def json_from_text(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    raw = text or ""
    for match in re.finditer(r"\{", raw):
        try:
            value, _end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object found in model output: {raw[:200]!r}")


def extract_contract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"<contract>\s*(.*?)\s*</contract>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        value = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def extract_buyer_price(text: str) -> Optional[float]:
    if not text:
        return None
    patterns = [
        r"###\s*BUYER_PRICE\s*\(\$([\d,]+\.?\d*)\)\s*###",
        r"BUYER_PRICE\s*\(\$([\d,]+\.?\d*)\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return _float_or_none(matches[-1])
    return None


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _bounded_float(value: Any, default: float) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return default
    return max(0.0, min(1.0, numeric))


def _negotiation_progress(current_state: Dict[str, Any], conversation_history: List[Dict[str, Any]]) -> float:
    round_raw = (
        current_state.get("round")
        or current_state.get("round_index")
        or current_state.get("current_round")
        or current_state.get("turn")
        or current_state.get("turn_index")
    )
    max_raw = current_state.get("max_rounds") or current_state.get("max_turns") or current_state.get("total_rounds")
    round_num = _float_or_none(round_raw)
    max_rounds = _float_or_none(max_raw)
    if round_num is None:
        seen_rounds = [
            _float_or_none(item.get("round"))
            for item in conversation_history
            if _float_or_none(item.get("round")) is not None
        ]
        round_num = max(seen_rounds) if seen_rounds else 1.0
    if max_rounds is None or max_rounds <= 0:
        max_rounds = 6.0
    return max(0.0, min(1.0, float(round_num) / float(max_rounds)))


def _number_range_or_none(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 2:
        return None
    low = _float_or_none(value[0])
    high = _float_or_none(value[1])
    if low is None or high is None:
        return None
    return [min(low, high), max(low, high)]


def _unit_interval_range_or_none(value: Any) -> Optional[List[float]]:
    if isinstance(value, (int, float, str)):
        point = _bounded_float(value, 0.5)
        return [point, point]
    raw_range = _number_range_or_none(value)
    if raw_range is None:
        return None
    return [max(0.0, min(1.0, raw_range[0])), max(0.0, min(1.0, raw_range[1]))]


def _range_midpoint(value: Optional[List[float]], default: float) -> float:
    if not value or len(value) != 2:
        return default
    return (value[0] + value[1]) / 2.0


class PromptOpponentBeliefModel:
    """Estimate seller-side hidden information from the public dialogue.

    This class is deliberately small and replaceable. A later trained belief
    model can implement the same `predict(...) -> BeliefState` interface.
    """

    def __init__(self, model: ModelClient, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.last_raw = ""

    def predict(
        self,
        *,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> BeliefState:
        prompt = f"""You are an opponent belief model for a buyer in AgenticPay negotiation.

Infer the seller-side hidden state and bargaining posture. Do not generate a buyer message.

Context:
{json.dumps(context, ensure_ascii=False, default=str)}

Current state:
{json.dumps(current_state, ensure_ascii=False, default=str)}

Conversation:
{format_history(conversation_history)}

Return only valid JSON with this schema:
{{
  "seller_reservation_range": [<low number>, <high number>] or null,
  "seller_reservation_confidence": <0-1, lower if evidence is weak>,
  "seller_flexibility_range": [<low 0-1>, <high 0-1>],
  "seller_patience_range": [<low 0-1>, <high 0-1>],
  "seller_dominance_range": [<low 0-1>, <high 0-1>],
  "seller_friendliness_range": [<low 0-1>, <high 0-1>],
  "deal_risk": <0-1, higher means timeout/no-deal risk>,
  "last_seller_offer": <number or null>,
  "likely_acceptable_price_range": [<low number>, <high number>] or null,
  "contract_term_preferences": {{"term_name": "short inferred preference"}},
  "evidence": ["brief observations grounded in the dialogue"]
}}

Important:
- The seller's reservation price/cost is private. Do not invent a precise point
  estimate unless the dialogue directly reveals it.
- Prefer an interval for seller_reservation_range, and set confidence low when
  only one or two offers have been observed.
- Do not force the seller into a single personality label. Use trait intervals:
  flexibility is willingness to concede, patience is willingness to continue,
  dominance is pressure/ultimatum behavior, and friendliness is warmth/cooperation.
- seller_reservation_range means the hidden minimum acceptable price/cost.
- likely_acceptable_price_range means transaction prices the seller may accept
  soon, which can be higher than the hidden reservation range."""
        self.last_raw = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
        try:
            parsed = json_from_text(self.last_raw)
        except Exception:
            parsed = {}
        flexibility_range = _unit_interval_range_or_none(
            parsed.get("seller_flexibility_range", parsed.get("seller_flexibility"))
        )
        patience_range = _unit_interval_range_or_none(
            parsed.get("seller_patience_range", parsed.get("seller_patience"))
        )
        return BeliefState(
            seller_reservation_range=_number_range_or_none(parsed.get("seller_reservation_range")),
            seller_reservation_confidence=_bounded_float(parsed.get("seller_reservation_confidence"), 0.3),
            seller_reservation_estimate=_float_or_none(parsed.get("seller_reservation_estimate")),
            seller_flexibility_range=flexibility_range,
            seller_patience_range=patience_range,
            seller_dominance_range=_unit_interval_range_or_none(parsed.get("seller_dominance_range")),
            seller_friendliness_range=_unit_interval_range_or_none(parsed.get("seller_friendliness_range")),
            seller_flexibility=_range_midpoint(flexibility_range, _bounded_float(parsed.get("seller_flexibility"), 0.5)),
            seller_patience=_range_midpoint(patience_range, _bounded_float(parsed.get("seller_patience"), 0.5)),
            seller_strategy=str(parsed.get("seller_strategy") or "unknown"),
            deal_risk=_bounded_float(parsed.get("deal_risk"), 0.5),
            last_seller_offer=_float_or_none(parsed.get("last_seller_offer")),
            likely_acceptable_price_range=_number_range_or_none(
                parsed.get("likely_acceptable_price_range", parsed.get("likely_acceptable_range"))
            ),
            contract_term_preferences=parsed.get("contract_term_preferences")
            if isinstance(parsed.get("contract_term_preferences"), dict)
            else {},
            evidence=parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
        )


class PromptStrategicPlanner:
    """Map state and optional belief into an abstract bargaining action."""

    def __init__(self, model: ModelClient, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.last_raw = ""

    def plan(
        self,
        *,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        buyer_max_price: Optional[float],
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        prompt = f"""You are the high-level planner for an AgenticPay buyer.

Choose a strategic action, not the final natural language message. Your job is
to protect buyer utility while still avoiding unnecessary no-deal failures.

Private buyer max price/value: {buyer_max_price}. Never reveal it.

Context:
{json.dumps(context, ensure_ascii=False, default=str)}

Current state:
{json.dumps(current_state, ensure_ascii=False, default=str)}

Belief state:
{json.dumps(belief.to_dict() if belief else None, ensure_ascii=False)}

Conversation:
{format_history(conversation_history)}

Anchor-and-concession strategy:
- Treat the buyer max price/value as a hard ceiling, not as a target.
- Early rounds should normally anchor low and gather information, not accept
  or jump to the buyer max. Opening offers should test seller flexibility.
- Concede slowly and deliberately. Prefer hold/concede_small after rejection;
  use concede_medium only when late-round no-deal risk is high.
- Do not fully trust seller claims like "final", "minimum", or "below cost";
  use them as soft signals unless repeated formal actions support them.
- In contract tasks, trade low-cost non-price terms for price concessions:
  give seller-compatible terms only when buyer utility remains positive.
- Accept only when the offer is clearly buyer-favorable or the final-round
  no-deal risk dominates the expected surplus from one more counteroffer.

Return only valid JSON:
{{
  "strategic_act": "anchor_low|hold|concede_small|concede_medium|ask_info|accept|walk_away",
  "target_price": <number or null>,
  "reservation_guardrail": <number or null>,
  "concession_size": "none|small|medium|large",
  "accept_if_at_or_below": <number or null>,
  "contract_priorities": ["price", "delivery_days", "return_policy", "packaging", "user_product_preference"],
  "required_non_price_terms": {{"term_name": "required_or_preferred_value"}},
  "avoid_non_price_terms": {{"term_name": ["bad_value_1", "bad_value_2"]}},
  "feasibility_checks": ["price <= buyer max", "buyer-side contract utility >= 0", "seller likely utility >= 0", "all required contract fields valid"],
  "seller_feasible_price_floor": <number or null, lowest price likely acceptable to seller>,
  "seller_feasibility_risk": <0-1, higher means buyer-feasible offer may still fail seller IR>,
  "seller_feasible_non_price_terms": {{"term_name": "seller-likely-feasible value or preference"}},
  "seller_feasibility_guardrails": ["brief rules for avoiding seller-side infeasible deals"],
  "language_style": "friendly|firm|urgent|patient",
  "private_rationale": "one short sentence"
}}

Rules:
- Never plan to offer above the buyer max price/value when it is known.
- Prefer lower buyer price unless deal risk is high or the seller offer is already attractive.
- Preserve buyer surplus. A higher target price lowers buyer score unless it
  materially increases the probability of a feasible deal.
- Early low anchors may be below the estimated seller floor when belief
  confidence is low; the purpose is information gathering and pressure.
- For contract tasks, plan both price and non-price terms because buyer utility is multi-issue.
- Use the buyer_preferences and field descriptions in context to avoid non-price terms that make buyer utility negative.
- If the seller proposes non-price terms that are costly to the buyer, do not accept them unless the lower price compensates.
- Use the belief state's seller_reservation_range and likely_acceptable_price_range
  as seller-feasibility guardrails. A buyer-feasible offer is still bad if it is
  very likely to give the seller negative utility.
- If the estimated seller-feasible price floor is above buyer max/value, prefer
  walk_away over repeating an impossible offer until timeout.
- If buyer max/value and seller-feasible floor overlap and deal risk is high,
  target the overlapping feasible region in later rounds rather than anchoring
  far below it forever. Do not do this on the first probe unless belief is very reliable.
- For contract tasks, include non-price terms that are likely acceptable to the
  seller when they do not violate buyer utility.
- The final plan must be feasible under the buyer's private value and plausibly
  seller-feasible under latent-belief uncertainty, not merely likely to get agreement."""
        self.last_raw = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
        try:
            parsed = json_from_text(self.last_raw)
        except Exception:
            parsed = {}
        target_price = _float_or_none(parsed.get("target_price"))
        reservation_guardrail = _float_or_none(parsed.get("reservation_guardrail")) or buyer_max_price
        seller_feasible_price_floor = _float_or_none(parsed.get("seller_feasible_price_floor"))
        inferred_seller_floor = self._infer_seller_feasible_floor(belief)
        if seller_feasible_price_floor is None:
            seller_feasible_price_floor = inferred_seller_floor
        seller_feasibility_risk = _bounded_float(
            parsed.get("seller_feasibility_risk"),
            belief.deal_risk if belief else 0.5,
        )
        progress = _negotiation_progress(current_state, conversation_history)
        strategic_act = str(parsed.get("strategic_act") or "concede_small")
        private_rationale = str(parsed.get("private_rationale") or "")

        if buyer_max_price is not None and target_price is not None:
            target_price = min(target_price, buyer_max_price)
        if buyer_max_price is not None and reservation_guardrail is not None:
            reservation_guardrail = min(reservation_guardrail, buyer_max_price)
        accept_if_at_or_below = _float_or_none(parsed.get("accept_if_at_or_below"))
        if buyer_max_price is not None and accept_if_at_or_below is not None:
            accept_if_at_or_below = min(accept_if_at_or_below, buyer_max_price)
        if seller_feasible_price_floor is not None and buyer_max_price is not None:
            if seller_feasible_price_floor > buyer_max_price + 1e-9:
                seller_feasibility_risk = max(seller_feasibility_risk, 0.9)
                if self._seller_floor_is_reliable(belief):
                    strategic_act = "walk_away"
                    target_price = buyer_max_price
                    accept_if_at_or_below = buyer_max_price
                    private_rationale = (
                        private_rationale
                        + " Seller-feasible floor appears above buyer max; avoid infeasible timeout."
                    ).strip()
            elif (
                target_price is not None
                and target_price < seller_feasible_price_floor
                and seller_feasibility_risk >= 0.55
                and (progress >= 0.55 or self._seller_floor_is_reliable(belief))
            ):
                target_price = min(seller_feasible_price_floor, buyer_max_price)
                if accept_if_at_or_below is not None:
                    accept_if_at_or_below = max(accept_if_at_or_below, target_price)
                private_rationale = (
                    private_rationale
                    + " Target moved into buyer/seller feasible overlap."
                ).strip()
        feasibility_checks = parsed.get("feasibility_checks") if isinstance(parsed.get("feasibility_checks"), list) else []
        if not feasibility_checks:
            feasibility_checks = ["price <= buyer max", "buyer-side contract utility >= 0"]
        if "seller likely utility >= 0" not in feasibility_checks:
            feasibility_checks.append("seller likely utility >= 0")
        seller_guardrails = (
            parsed.get("seller_feasibility_guardrails")
            if isinstance(parsed.get("seller_feasibility_guardrails"), list)
            else []
        )
        if seller_feasible_price_floor is not None:
            seller_guardrails.append(f"avoid final offers below likely seller-feasible floor {seller_feasible_price_floor:g}")
        if buyer_max_price is not None and seller_feasible_price_floor is not None and seller_feasible_price_floor > buyer_max_price:
            seller_guardrails.append("seller floor appears above buyer max; prefer walk_away to timeout")
        seller_non_price_terms = (
            parsed.get("seller_feasible_non_price_terms")
            if isinstance(parsed.get("seller_feasible_non_price_terms"), dict)
            else {}
        )
        if belief and belief.contract_term_preferences:
            seller_non_price_terms = {**belief.contract_term_preferences, **seller_non_price_terms}
        return StrategicPlan(
            strategic_act=strategic_act,
            target_price=target_price,
            reservation_guardrail=reservation_guardrail,
            concession_size=str(parsed.get("concession_size") or "small"),
            accept_if_at_or_below=accept_if_at_or_below,
            contract_priorities=parsed.get("contract_priorities")
            if isinstance(parsed.get("contract_priorities"), list)
            else ["price"],
            required_non_price_terms=parsed.get("required_non_price_terms")
            if isinstance(parsed.get("required_non_price_terms"), dict)
            else {},
            avoid_non_price_terms=parsed.get("avoid_non_price_terms")
            if isinstance(parsed.get("avoid_non_price_terms"), dict)
            else {},
            feasibility_checks=feasibility_checks,
            seller_feasible_price_floor=seller_feasible_price_floor,
            seller_feasibility_risk=seller_feasibility_risk,
            seller_feasible_non_price_terms=seller_non_price_terms,
            seller_feasibility_guardrails=list(dict.fromkeys(str(item) for item in seller_guardrails)),
            language_style=str(parsed.get("language_style") or "friendly"),
            private_rationale=private_rationale,
        )

    @staticmethod
    def _infer_seller_feasible_floor(belief: Optional[BeliefState]) -> Optional[float]:
        if belief is None:
            return None
        if belief.likely_acceptable_price_range:
            return belief.likely_acceptable_price_range[0]
        if belief.last_seller_offer is not None and belief.deal_risk >= 0.65:
            return belief.last_seller_offer
        if belief.seller_reservation_range and belief.seller_reservation_confidence >= 0.5:
            return belief.seller_reservation_range[0]
        return None

    @staticmethod
    def _seller_floor_is_reliable(belief: Optional[BeliefState]) -> bool:
        if belief is None:
            return False
        if belief.deal_risk >= 0.7:
            return True
        if belief.seller_reservation_confidence >= 0.55:
            return True
        if belief.last_seller_offer is not None and belief.seller_flexibility <= 0.35:
            return True
        return False


class ResponseValidator:
    """Validate and revise final buyer text before it reaches AgenticPay."""

    def __init__(self, model: ModelClient, buyer_max_price: Optional[float], max_tokens: int = 1024):
        self.model = model
        self.buyer_max_price = buyer_max_price
        self.max_tokens = max_tokens
        self.last_raw_revision = ""

    def validate_and_revise(
        self,
        *,
        response: str,
        context: Dict[str, Any],
        current_state: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        belief: Optional[BeliefState],
        plan: Optional[StrategicPlan],
        selected_seller_id: Optional[int] = None,
    ) -> NaturalizationResult:
        validation = self.validate_response(
            response=response,
            context=context,
            current_state=current_state,
            plan=plan,
            selected_seller_id=selected_seller_id,
        )
        if validation["ok"]:
            return NaturalizationResult(final_response=response, raw_response=response, validation=validation)

        repaired_contract = validation.get("recommended_contract")
        repaired_price = validation.get("recommended_price")
        if repaired_contract:
            hard_target = (
                "Use exactly this buyer-feasible contract JSON. Do not change any price, "
                "continuous term, or discrete term:\n"
                f"{json.dumps(repaired_contract, ensure_ascii=False, indent=2)}"
            )
        elif repaired_price is not None:
            hard_target = (
                f"Use exactly one buyer-feasible price tag: ### BUYER_PRICE(${repaired_price:g}) ###."
            )
        else:
            hard_target = "Repair the response while satisfying all hard constraints."

        revision_prompt = f"""You are a validator-rewriter for an AgenticPay buyer.

The previous buyer response violates feasibility constraints. Rewrite it once.
Return only the final buyer message, not analysis.

Concrete repair target:
{hard_target}

Hard constraints:
- Never offer or accept a price above buyer max/value: {self.buyer_max_price}.
- If the task uses BUYER_PRICE, include exactly one valid ### BUYER_PRICE($X) ### tag.
- If the task uses <contract>, include exactly one complete valid <contract> JSON block.
- For contract tasks, every required continuous/discrete field must be valid under the schema.
- For contract tasks, preserve every repaired non-price term exactly; all continuous
  and discrete terms are part of buyer utility.
- Avoid any non-price term combination that makes buyer-side utility negative.
- Do not reveal private buyer max, belief JSON, or the private plan.

Validation errors:
{json.dumps(validation, ensure_ascii=False)}

Context:
{json.dumps(context, ensure_ascii=False, default=str)}

Current state:
{json.dumps(current_state, ensure_ascii=False, default=str)}

Belief:
{json.dumps(belief.to_dict() if belief else None, ensure_ascii=False)}

Plan:
{json.dumps(plan.to_dict() if plan else None, ensure_ascii=False)}

Conversation:
{format_history(conversation_history)}

Previous invalid response:
{response}
"""
        revised = self.model.generate(revision_prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens).strip()
        self.last_raw_revision = revised
        revised_validation = self.validate_response(
            response=revised,
            context=context,
            current_state=current_state,
            plan=plan,
            selected_seller_id=selected_seller_id,
        )
        revised_validation["initial_validation"] = validation
        revised_validation["was_revised"] = True
        if not revised_validation["ok"]:
            fallback = self._fallback_response(repaired_contract, repaired_price)
            if fallback:
                fallback_validation = self.validate_response(
                    response=fallback,
                    context=context,
                    current_state=current_state,
                    plan=plan,
                    selected_seller_id=selected_seller_id,
                )
                fallback_validation["initial_validation"] = validation
                fallback_validation["llm_revision_validation"] = revised_validation
                fallback_validation["was_revised"] = True
                fallback_validation["used_deterministic_fallback"] = True
                return NaturalizationResult(
                    final_response=fallback,
                    raw_response=response,
                    validation=fallback_validation,
                )
        return NaturalizationResult(final_response=revised, raw_response=response, validation=revised_validation)

    def validate_response(
        self,
        *,
        response: str,
        context: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None,
        plan: Optional[StrategicPlan] = None,
        selected_seller_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        contract = extract_contract(response)
        price = self._response_price(response, contract)
        contract_config = self._contract_config(context, current_state or {}, selected_seller_id)
        buyer_max = self.buyer_max_price or _float_or_none(context.get("max_price"))
        guardrail = plan.reservation_guardrail if plan else None

        if price is None:
            warnings.append("No explicit buyer price or contract price found.")
        if buyer_max is not None and price is not None and price > buyer_max + 1e-9:
            errors.append(f"Price {price} exceeds buyer max/value {buyer_max}.")
        if guardrail is not None and price is not None and price > guardrail + 1e-9:
            errors.append(f"Price {price} exceeds planner reservation guardrail {guardrail}.")

        contract_result: Dict[str, Any] = {}
        recommended_contract: Optional[Dict[str, Any]] = None
        if contract_config:
            if contract is None:
                errors.append("Contract-mode context detected but no valid <contract> JSON was found.")
            else:
                contract_result = self._validate_contract(contract, contract_config)
                errors.extend(contract_result.get("errors", []))
                warnings.extend(contract_result.get("warnings", []))
            repaired = self._repair_contract_for_buyer(
                contract=contract,
                config=contract_config,
                buyer_max=buyer_max,
                guardrail=guardrail,
                plan=plan,
            )
            if repaired:
                recommended_contract = repaired["contract"]
                contract_result["recommended_contract"] = recommended_contract
                contract_result["recommended_buyer_utility"] = repaired.get("buyer_utility")
                contract_result["recommended_changed_terms"] = repaired.get("changed_terms", [])

        recommended_price: Optional[float] = None
        if not contract_config and buyer_max is not None:
            price_cap = buyer_max
            if guardrail is not None:
                price_cap = min(price_cap, guardrail)
            if price is None or price > price_cap + 1e-9:
                recommended_price = price_cap

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "price": price,
            "buyer_max_price": buyer_max,
            "contract": contract,
            "contract_diagnostics": contract_result,
            "recommended_contract": recommended_contract,
            "recommended_price": recommended_price,
            "was_revised": False,
        }

    @staticmethod
    def _contract_config(
        context: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None,
        selected_seller_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        current_state = current_state or {}
        seller_configs = (
            context.get("seller_contract_configs")
            or context.get("environment_info", {}).get("seller_contract_configs")
            or current_state.get("contract_configs")
        )
        if isinstance(seller_configs, dict) and seller_configs:
            seller_key = selected_seller_id
            if seller_key is None:
                seller_key = current_state.get("selected_seller") or current_state.get("buyer_selected_seller")
            if seller_key is not None:
                config = seller_configs.get(seller_key) or seller_configs.get(str(seller_key))
                if isinstance(config, dict):
                    return config
            first_config = next((value for value in seller_configs.values() if isinstance(value, dict)), {})
            return first_config if isinstance(first_config, dict) else {}
        config = (
            context.get("contract_config")
            or context.get("environment_info", {}).get("contract_config")
            or current_state.get("contract_config")
            or {}
        )
        return config if isinstance(config, dict) else {}

    @staticmethod
    def _response_price(response: str, contract: Optional[Dict[str, Any]]) -> Optional[float]:
        if contract is not None:
            price = _float_or_none(contract.get("price"))
            if price is not None:
                return price
        return extract_buyer_price(response)

    def _validate_contract(self, contract: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        continuous_terms = contract.get("continuous_terms", {})
        discrete_terms = contract.get("discrete_terms", {})
        if not isinstance(continuous_terms, dict):
            errors.append("contract.continuous_terms must be an object.")
            continuous_terms = {}
        if not isinstance(discrete_terms, dict):
            errors.append("contract.discrete_terms must be an object.")
            discrete_terms = {}

        continuous_bounds = config.get("continuous_bounds", {})
        discrete_options = config.get("discrete_options", {})
        for term in continuous_terms:
            if term not in continuous_bounds:
                errors.append(f"Unknown continuous term: {term}.")
        for term in discrete_terms:
            if term not in discrete_options:
                errors.append(f"Unknown discrete term: {term}.")
        for term, bounds in continuous_bounds.items():
            if term not in continuous_terms:
                errors.append(f"Missing required continuous term: {term}.")
                continue
            value = _float_or_none(continuous_terms.get(term))
            if value is None:
                errors.append(f"Continuous term {term} is not numeric.")
                continue
            min_v = bounds.get("min")
            max_v = bounds.get("max")
            if min_v is not None and value < float(min_v):
                errors.append(f"Continuous term {term}={value} is below min {min_v}.")
            if max_v is not None and value > float(max_v):
                errors.append(f"Continuous term {term}={value} is above max {max_v}.")
        for term, options in discrete_options.items():
            if term not in discrete_terms:
                errors.append(f"Missing required discrete term: {term}.")
                continue
            if discrete_terms.get(term) not in options:
                errors.append(f"Discrete term {term}={discrete_terms.get(term)!r} is not in {options}.")

        buyer_utility = self._buyer_contract_utility(contract, config)
        if buyer_utility is not None and buyer_utility < 0:
            errors.append(f"Buyer-side contract utility is negative ({buyer_utility:.4f}).")
        return {
            "errors": errors,
            "warnings": warnings,
            "buyer_utility": buyer_utility,
        }

    def _repair_contract_for_buyer(
        self,
        *,
        contract: Optional[Dict[str, Any]],
        config: Dict[str, Any],
        buyer_max: Optional[float],
        guardrail: Optional[float],
        plan: Optional[StrategicPlan],
    ) -> Optional[Dict[str, Any]]:
        """Choose a complete contract that is valid and buyer-utility feasible.

        The repair is buyer-side only: seller utility is private and should be
        handled as a bargaining-risk signal by belief/planning, not as a hard
        validator constraint.
        """
        buyer_prefs = config.get("buyer_preferences")
        if not isinstance(buyer_prefs, dict):
            return None

        price_cap = buyer_max
        if price_cap is None:
            price_cap = _float_or_none(config.get("max_price"))
        if guardrail is not None:
            price_cap = min(price_cap, guardrail) if price_cap is not None else guardrail
        if price_cap is None:
            price_cap = _float_or_none(contract.get("price")) if contract else plan.target_price if plan else None
        if price_cap is None:
            return None

        current_price = _float_or_none(contract.get("price")) if contract else None
        if current_price is not None:
            price_cap = min(price_cap, current_price)
        if plan and plan.target_price is not None:
            price_cap = min(price_cap, plan.target_price)

        continuous_choices = self._continuous_candidate_values(contract, config)
        discrete_choices = self._discrete_candidate_values(contract, config)
        if continuous_choices is None or discrete_choices is None:
            return None

        continuous_terms = list(continuous_choices.keys())
        discrete_terms = list(discrete_choices.keys())
        best_contract: Optional[Dict[str, Any]] = None
        best_utility: Optional[float] = None
        best_score: Optional[float] = None

        for continuous_values in product(*(continuous_choices[term] for term in continuous_terms)):
            continuous_part = dict(zip(continuous_terms, continuous_values))
            for discrete_values in product(*(discrete_choices[term] for term in discrete_terms)):
                discrete_part = dict(zip(discrete_terms, discrete_values))
                term_value = self._buyer_term_value(continuous_part, discrete_part, config)
                if term_value is None:
                    continue
                feasible_price = min(price_cap, term_value)
                if feasible_price < 0:
                    continue
                candidate = {
                    "price": round(feasible_price, 4),
                    "continuous_terms": continuous_part,
                    "discrete_terms": discrete_part,
                }
                utility = self._buyer_contract_utility(candidate, config)
                if utility is None:
                    continue
                change_penalty = self._contract_change_penalty(contract, candidate)
                # Primary goal: non-negative buyer utility. Secondary goals:
                # preserve the naturalizer's terms and keep price close to plan.
                is_feasible = 1.0 if utility >= -1e-9 else 0.0
                score = is_feasible * 1_000_000.0 + utility * 1_000.0 - change_penalty
                if best_score is None or score > best_score:
                    best_contract = candidate
                    best_utility = utility
                    best_score = score

        if best_contract is None:
            return None
        return {
            "contract": best_contract,
            "buyer_utility": best_utility,
            "changed_terms": self._changed_terms(contract, best_contract),
        }

    @staticmethod
    def _continuous_candidate_values(
        contract: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Optional[Dict[str, List[float]]]:
        bounds_map = config.get("continuous_bounds", {})
        if not isinstance(bounds_map, dict):
            return None
        current_terms = contract.get("continuous_terms", {}) if contract else {}
        current_terms = current_terms if isinstance(current_terms, dict) else {}
        choices: Dict[str, List[float]] = {}
        buyer_weights = config.get("buyer_preferences", {}).get("continuous_weights", {})
        for term, bounds in bounds_map.items():
            if not isinstance(bounds, dict):
                return None
            min_v = _float_or_none(bounds.get("min"))
            max_v = _float_or_none(bounds.get("max"))
            if min_v is None or max_v is None:
                return None
            if min_v > max_v:
                min_v, max_v = max_v, min_v
            candidates = [min_v, max_v]
            current = _float_or_none(current_terms.get(term))
            if current is not None and min_v <= current <= max_v:
                candidates.append(current)
            weight = _float_or_none(buyer_weights.get(term)) or 0.0
            candidates.insert(0, max_v if weight >= 0 else min_v)
            choices[term] = list(dict.fromkeys(round(v, 4) for v in candidates))
        return choices

    @staticmethod
    def _discrete_candidate_values(
        contract: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Optional[Dict[str, List[Any]]]:
        options_map = config.get("discrete_options", {})
        if not isinstance(options_map, dict):
            return None
        current_terms = contract.get("discrete_terms", {}) if contract else {}
        current_terms = current_terms if isinstance(current_terms, dict) else {}
        buyer_weights = config.get("buyer_preferences", {}).get("discrete_weights", {})
        choices: Dict[str, List[Any]] = {}
        for term, options in options_map.items():
            if not isinstance(options, list) or not options:
                return None
            ranked = sorted(
                options,
                key=lambda value: ResponseValidator._discrete_utility(buyer_weights, term, value),
                reverse=True,
            )
            current = current_terms.get(term)
            if current in options and current not in ranked:
                ranked.append(current)
            choices[term] = ranked
        return choices

    @staticmethod
    def _buyer_term_value(
        continuous_terms: Dict[str, Any],
        discrete_terms: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[float]:
        buyer_prefs = config.get("buyer_preferences")
        if not isinstance(buyer_prefs, dict):
            return None
        value = float(buyer_prefs.get("v_base", 0.0))
        continuous_weights = buyer_prefs.get("continuous_weights", {})
        for term, raw_value in continuous_terms.items():
            numeric_value = _float_or_none(raw_value)
            if numeric_value is not None:
                value += float(continuous_weights.get(term, 0.0)) * numeric_value
        discrete_weights = buyer_prefs.get("discrete_weights", {})
        for term, raw_value in discrete_terms.items():
            value += ResponseValidator._discrete_utility(discrete_weights, term, raw_value)
        return value

    @staticmethod
    def _discrete_utility(weights: Dict[str, Any], term: str, value: Any) -> float:
        term_weights = weights.get(term, {}) if isinstance(weights, dict) else {}
        if not isinstance(term_weights, dict):
            return 0.0
        if value in term_weights:
            return float(term_weights.get(value, 0.0))
        string_value = str(value)
        if string_value in term_weights:
            return float(term_weights.get(string_value, 0.0))
        return 0.0

    @staticmethod
    def _contract_change_penalty(
        original: Optional[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> float:
        if not original:
            return 0.0
        penalty = 0.0
        original_price = _float_or_none(original.get("price"))
        candidate_price = _float_or_none(candidate.get("price"))
        if original_price is not None and candidate_price is not None:
            penalty += abs(original_price - candidate_price) * 0.1
        for term_group in ("continuous_terms", "discrete_terms"):
            original_terms = original.get(term_group, {})
            candidate_terms = candidate.get(term_group, {})
            if not isinstance(original_terms, dict) or not isinstance(candidate_terms, dict):
                continue
            for term, candidate_value in candidate_terms.items():
                if original_terms.get(term) != candidate_value:
                    penalty += 1.0
        return penalty

    @staticmethod
    def _changed_terms(
        original: Optional[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> List[str]:
        if not original:
            return ["price", "continuous_terms", "discrete_terms"]
        changed: List[str] = []
        if _float_or_none(original.get("price")) != _float_or_none(candidate.get("price")):
            changed.append("price")
        for term_group in ("continuous_terms", "discrete_terms"):
            original_terms = original.get(term_group, {})
            candidate_terms = candidate.get(term_group, {})
            if not isinstance(original_terms, dict) or not isinstance(candidate_terms, dict):
                changed.append(term_group)
                continue
            for term, candidate_value in candidate_terms.items():
                if original_terms.get(term) != candidate_value:
                    changed.append(f"{term_group}.{term}")
        return changed

    @staticmethod
    def _fallback_response(
        repaired_contract: Optional[Dict[str, Any]],
        repaired_price: Optional[float],
    ) -> Optional[str]:
        if repaired_contract:
            return (
                "I can proceed with this buyer-feasible contract:\n"
                "<contract>\n"
                f"{json.dumps(repaired_contract, ensure_ascii=False, indent=2)}\n"
                "</contract>"
            )
        if repaired_price is not None:
            return f"I can offer ### BUYER_PRICE(${repaired_price:g}) ###."
        return None

    @staticmethod
    def _buyer_contract_utility(contract: Dict[str, Any], config: Dict[str, Any]) -> Optional[float]:
        buyer_prefs = config.get("buyer_preferences")
        if not isinstance(buyer_prefs, dict):
            return None
        price = _float_or_none(contract.get("price"))
        if price is None:
            return None
        utility = float(buyer_prefs.get("v_base", 0.0)) - price
        continuous_terms = contract.get("continuous_terms", {}) if isinstance(contract.get("continuous_terms"), dict) else {}
        discrete_terms = contract.get("discrete_terms", {}) if isinstance(contract.get("discrete_terms"), dict) else {}
        for term, value in continuous_terms.items():
            numeric_value = _float_or_none(value)
            if numeric_value is not None:
                utility += float(buyer_prefs.get("continuous_weights", {}).get(term, 0.0)) * numeric_value
        for term, value in discrete_terms.items():
            utility += ResponseValidator._discrete_utility(
                buyer_prefs.get("discrete_weights", {}),
                term,
                value,
            )
        return utility


class NativeBuyerNaturalizer:
    """Turn private belief/plan guidance into AgenticPay-compliant buyer text."""

    def __init__(
        self,
        *,
        model: ModelClient,
        name: str,
        role_description: str,
        buyer_max_price: Optional[float],
        system_prompt_suffix: Optional[str],
    ):
        if BuyerAgent is None:
            raise RuntimeError(
                "AgenticPay is required for NativeBuyerNaturalizer. "
                "Run scripts/bootstrap_upstreams.sh first."
            )
        self.base_suffix = system_prompt_suffix
        self.last_selected_seller: Optional[int] = None
        self.inner = BuyerAgent(
            model=model,
            name=name,
            role_description=role_description,
            buyer_max_price=buyer_max_price,
            system_prompt_suffix=system_prompt_suffix,
        )

    def initialize(self, context: Dict[str, Any]) -> None:
        self.inner.initialize(context)

    def generate(
        self,
        *,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        belief: Optional[BeliefState],
        plan: Optional[StrategicPlan],
    ) -> NaturalizationResult:
        self.inner.system_prompt_suffix = self._build_private_guidance(belief, plan)
        final_response = self.inner.respond(conversation_history, current_state)
        self.last_selected_seller = getattr(self.inner, "last_selected_seller", None)
        return NaturalizationResult(final_response=final_response, raw_response=final_response)

    def _build_private_guidance(
        self,
        belief: Optional[BeliefState],
        plan: Optional[StrategicPlan],
    ) -> str:
        parts = []
        if self.base_suffix:
            parts.append(self.base_suffix)
        parts.append(
            "BUYER-SIDE MODULAR FRAMEWORK GUIDANCE FOR PRIVATE USE ONLY:\n"
            "- Use this auxiliary analysis to choose the final buyer message.\n"
            "- Do not reveal the belief JSON, strategic plan, or private rationale.\n"
            "- Follow all native AgenticPay output rules exactly.\n"
            "- If using price tags, include exactly one valid BUYER_PRICE tag.\n"
            "- If using <contract> JSON, include one complete valid <contract> block with every required field.\n"
            "- Never offer or accept a price above the buyer max/value.\n"
            "- For contract tasks, satisfy buyer-side utility and avoid non-price terms that make the deal economically infeasible.\n"
            "- Also respect seller-feasibility guardrails in the strategic plan: avoid repeated offers that are very likely below the seller's feasible range, and use walk-away language when the plan says walk_away.\n"
            "- Use an anchor-and-concession bargaining style: start buyer-favorable, make small reluctant concessions, and do not reveal or spend up to the buyer max unless late-round risk requires it.\n"
            "- If the plan says anchor_low, hold, or concede_small, keep the buyer price close to the plan target; do not soften it toward the seller's ask.\n"
            "- Use firm language such as a serious quick-close offer, budget discipline, and needing the seller to move; avoid apologetic language that invites a high counter.\n"
            "- Seller claims of final/minimum/below-cost are soft pressure unless backed by repeated formal actions."
        )
        if belief is not None:
            parts.append("Opponent belief state:\n" + json.dumps(belief.to_dict(), ensure_ascii=False))
        if plan is not None:
            parts.append("High-level strategic plan:\n" + json.dumps(plan.to_dict(), ensure_ascii=False))
        return "\n\n".join(parts)
