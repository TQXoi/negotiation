"""30B verifier and final language generator for v0.3 policy actions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from experiments.agenticpay_framework.components import json_from_text
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    format_action_message,
    format_transcript,
    last_price,
    parse_action,
    validate_buyer_action,
)
from experiments.model_clients import ModelClient

from ..continuous_solver.generator.base import GenerationResult
from .schema import PricePolicyAction, VerifiedPolicyAction
from .policy_model import compact_belief_summary


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class PolicyVerifierGenerator:
    """Use a larger model to verify a small price-policy action and speak."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 1600, temperature: float = 0.8, top_p: float = 1.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.last_verify_prompt = ""
        self.last_verify_raw = ""
        self.last_message_prompt = ""
        self.last_message_raw = ""
        self.last_verified: Optional[VerifiedPolicyAction] = None

    def generate(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        policy_action: PricePolicyAction,
        belief: Any | None = None,
    ) -> GenerationResult:
        verified = self.verify(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
            policy_action=policy_action,
            belief=belief,
        )
        self.last_verified = verified
        prompt = self._message_prompt(scenario, history, round_id, max_turns, verified, belief=belief)
        self.last_message_prompt = prompt
        self.last_message_raw = self.client.generate(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        parsed = parse_action("buyer", self.last_message_raw)
        parsed, validator = validate_buyer_action(parsed, scenario, history, round_id, max_turns)
        if not parsed.valid or not self._matches_verified(parsed, verified, scenario, history):
            fallback = self._fallback_message(scenario, history, verified)
            parsed = parse_action("buyer", fallback)
            parsed, validator = validate_buyer_action(parsed, scenario, history, round_id, max_turns)
            validator = {
                **validator,
                "generator_fallback_used": True,
                "llm_raw": self.last_message_raw,
                "fallback_reason": "invalid_or_not_matching_verified_policy_action",
            }
            return GenerationResult(
                raw=fallback,
                parsed_action=parsed,
                validator={**validator, "verified_policy_action": verified.to_dict()},
                generator_type="v03_policy_verifier_generator_with_fallback",
                prompt=prompt,
            )
        return GenerationResult(
            raw=self.last_message_raw,
            parsed_action=parsed,
            validator={**validator, "verified_policy_action": verified.to_dict()},
            generator_type="v03_policy_verifier_generator",
            prompt=prompt,
        )

    def verify(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        policy_action: PricePolicyAction,
        belief: Any | None = None,
    ) -> VerifiedPolicyAction:
        prompt = self._verify_prompt(scenario, history, round_id, max_turns, policy_action, belief=belief)
        self.last_verify_prompt = prompt
        self.last_verify_raw = self.client.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=700)
        try:
            parsed = json_from_text(self.last_verify_raw)
        except Exception:
            return self._deterministic_verify(policy_action, scenario, history, "invalid_verifier_json")
        approved = bool(parsed.get("approved"))
        revised = parsed.get("revised_action") if isinstance(parsed.get("revised_action"), dict) else {}
        action = PricePolicyAction(
            action_type=str(revised.get("action_type") or policy_action.action_type),
            target_price=_float_or_none(revised.get("target_price", policy_action.target_price)),
            accept_formal_seller_offer=bool(revised.get("accept_formal_seller_offer", policy_action.accept_formal_seller_offer)),
            concession_style=str(revised.get("concession_style") or policy_action.concession_style),
            confidence=float(max(0.0, min(1.0, _float_or_none(revised.get("confidence")) or policy_action.confidence))),
            rationale=str(revised.get("rationale") or policy_action.rationale),
        )
        action = self._sanitize_action(action, scenario, history, round_id, max_turns)
        action = self._cap_upward_revision(
            original=policy_action,
            revised=action,
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
        )
        return VerifiedPolicyAction(
            approved=approved and action.to_dict() == self._sanitize_action(policy_action, scenario, history, round_id, max_turns).to_dict(),
            action=action,
            revision_reason=str(parsed.get("revision_reason") or ""),
            language_plan=str(parsed.get("language_plan") or "firm, concise, buyer-favorable"),
            raw=self.last_verify_raw,
        )

    def _verify_prompt(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        policy_action: PricePolicyAction,
        belief: Any | None = None,
    ) -> str:
        seller_offer = last_price(history, "seller")
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        compact_belief = compact_belief_summary(belief)
        return f"""You are a verifier for an RLVR buyer price policy.

Check whether the small policy action is valid and strategically sane. You may
revise the action/price, but keep the revision close unless there is a clear
problem. Do not write the final buyer message.

Private buyer budget: {scenario.buyer_budget:.2f}
Reference/list price: {scenario.reference_price:.2f}
Round: {round_id}/{max_turns}
Latest formal seller [SELL] price: {seller_offer}
Lowest formal seller [SELL] price: {min(seller_prices) if seller_prices else None}

Small policy action:
{json.dumps(policy_action.to_dict(), ensure_ascii=False)}

Opponent belief summary:
{json.dumps(compact_belief, ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Verification rules:
- Hard reject/revise any target_price above budget.
- If action_type is accept, it must copy an exact previous formal seller [SELL] offer within budget.
- If action_type is offer and there is a formal seller [SELL] price, keep target_price below the lowest formal seller quote.
- "Below the seller quote" is GOOD for a buyer counteroffer. Do not revise a
  buyer [BUY] upward merely because it is below the seller quote.
- Do not accept Talk-only prices when seller Action is [REJECT].
- Early rounds should usually approve aggressive low anchors, even if acceptance probability is low.
- The belief summary is noisy. Use it to prevent repeated obviously infeasible
  low offers, but do not convert seller reservation estimates into buyer price
  targets.
- When belief says seller floor/deal risk is high and the same low offer has
  already been rejected, a small upward revision is allowed.
- Avoid [QUIT] before the final round.
- Preserve buyer surplus. Lower deal price means higher reward. Never say a
  higher buyer offer "maximizes surplus"; it reduces surplus.
- Do not revise a small-policy offer upward by more than a small concession
  unless the original action violates a hard rule.
- A near-budget offer is only sane in late rounds after repeated formal seller
  [SELL] offers within budget. It is not sane just because the seller rejected.

Return only valid JSON:
{{
  "approved": <true or false>,
  "revised_action": {{
    "action_type": "offer|accept|reject|quit",
    "target_price": <number or null>,
    "accept_formal_seller_offer": <true or false>,
    "concession_style": "anchor_low|hold_firm|small_concession|medium_concession|late_close|walkaway",
    "confidence": <0-1>,
    "rationale": "short private reason"
  }},
  "revision_reason": "why changed or approved",
  "language_plan": "one sentence about buyer tone and persuasion"
}}
"""

    def _message_prompt(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        verified: VerifiedPolicyAction,
        belief: Any | None = None,
    ) -> str:
        action = verified.action
        compact_belief = compact_belief_summary(belief)
        return f"""You are the final buyer message generator in an RLVR negotiation.

Use the verified policy action exactly. Your job is only to produce persuasive
Thought/Talk/Action text in the paper format.

Private buyer budget: {scenario.buyer_budget:.2f}
Product codename: {scenario.codename_or_default()}
Quantity: {scenario.quantity}
Reference/list price: {scenario.reference_price:.2f}
Round: {round_id}/{max_turns}

Verified policy action:
{json.dumps(verified.to_dict(), ensure_ascii=False)}

Compact opponent belief:
{json.dumps(compact_belief, ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Language strategy:
- Be firm and buyer-favorable.
- Use anchor-and-concession language: serious cash offer, quick close, cannot justify the ask, need you to move.
- If early, pressure and probe while making the concrete offer.
- If conceding, make it sound reluctant and calculated.
- If belief suggests high seller floor or high deal risk, frame the offer as a
  serious quick-close concession without revealing budget.
- Never reveal the true budget.
- Do not mention policy, verifier, probabilities, reward, or JSON.

Action rules:
- If verified action_type is offer, output exactly Action: [BUY] ${action.target_price} ({scenario.quantity}x {scenario.codename_or_default()}).
- If verified action_type is accept, output exactly Action: [DEAL] ${action.target_price} ({scenario.quantity}x {scenario.codename_or_default()}).
- If verified action_type is reject, output Action: [REJECT].
- If verified action_type is quit, output Action: [QUIT].

Return exactly:
Thought: ...
Talk: ...
Action: ...
"""

    def _deterministic_verify(
        self,
        action: PricePolicyAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        reason: str,
    ) -> VerifiedPolicyAction:
        revised = self._sanitize_action(action, scenario, history, round_id=1_000_000, max_turns=1_000_000)
        return VerifiedPolicyAction(
            approved=False,
            action=revised,
            revision_reason=reason,
            language_plan="firm counteroffer with safe price",
            raw=self.last_verify_raw,
        )

    def _sanitize_action(
        self,
        action: PricePolicyAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> PricePolicyAction:
        seller_offer = last_price(history, "seller")
        action_type = action.action_type if action.action_type in {"offer", "accept", "reject", "quit"} else "offer"
        price = action.target_price
        if action_type == "quit" and round_id < max_turns:
            action_type = "offer"
        if action_type == "accept":
            if seller_offer is None or seller_offer > scenario.buyer_budget:
                action_type = "offer"
            else:
                price = float(seller_offer)
        if action_type == "offer":
            price = self._safe_offer_price(
                float(price) if price is not None else min(scenario.buyer_budget, scenario.reference_price * 0.45),
                scenario,
                history,
            )
        elif action_type in {"reject", "quit"}:
            price = None
        return PricePolicyAction(
            action_type=action_type,
            target_price=round(float(price), 2) if price is not None else None,
            accept_formal_seller_offer=action_type == "accept",
            concession_style=action.concession_style,
            confidence=action.confidence,
            rationale=action.rationale,
        )

    def _cap_upward_revision(
        self,
        *,
        original: PricePolicyAction,
        revised: PricePolicyAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> PricePolicyAction:
        """Prevent the verifier from erasing the trainable policy's price decision."""

        if original.action_type != "offer" or revised.action_type != "offer":
            return revised
        if original.target_price is None or revised.target_price is None:
            return revised
        original_price = float(original.target_price)
        revised_price = float(revised.target_price)
        if revised_price <= original_price:
            return revised

        budget = float(scenario.buyer_budget)
        last_buyer = last_price(history, "buyer")
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        seller_cap = min(seller_prices) - max(0.01, 0.002 * budget) if seller_prices else budget
        if round_id <= 2:
            max_increment = 0.03 * budget
        elif round_id >= max_turns:
            max_increment = 0.10 * budget
        else:
            max_increment = 0.055 * budget
        base = last_buyer if last_buyer is not None else original_price
        capped_price = min(
            revised_price,
            original_price + max_increment,
            float(base) + max_increment,
            seller_cap,
            budget,
        )
        if capped_price >= revised_price - 1e-9:
            return revised
        return PricePolicyAction(
            action_type="offer",
            target_price=round(max(0.01, capped_price), 2),
            accept_formal_seller_offer=False,
            concession_style=revised.concession_style,
            confidence=min(revised.confidence, original.confidence),
            rationale=(
                revised.rationale
                + " Verifier upward revision capped to preserve buyer surplus and keep the small policy in control."
            ).strip(),
        )

    @staticmethod
    def _safe_offer_price(price: float, scenario: RLVRScenario, history: Sequence[Dict[str, Any]]) -> float:
        cap = float(scenario.buyer_budget)
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        if seller_prices:
            cap = min(cap, min(seller_prices) - max(0.01, 0.002 * scenario.buyer_budget))
        return max(0.01, min(float(price), cap))

    def _matches_verified(
        self,
        parsed: ParsedAction,
        verified: VerifiedPolicyAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
    ) -> bool:
        action = verified.action
        if parsed.action == "buy" and action.action_type == "offer":
            return parsed.price is not None and action.target_price is not None and abs(parsed.price - action.target_price) <= 0.02
        if parsed.action == "deal" and action.action_type == "accept":
            seller_offer = last_price(history, "seller")
            return (
                seller_offer is not None
                and parsed.price is not None
                and parsed.price <= scenario.buyer_budget
                and abs(parsed.price - seller_offer) <= 0.02
            )
        if parsed.action in {"reject", "quit"}:
            return parsed.action == action.action_type
        return False

    @staticmethod
    def _fallback_message(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        verified: VerifiedPolicyAction,
    ) -> str:
        action = verified.action
        if action.action_type == "accept":
            seller_offer = last_price(history, "seller")
            if seller_offer is not None and seller_offer <= scenario.buyer_budget:
                return format_action_message(
                    "buyer",
                    "DEAL",
                    round(float(seller_offer), 2),
                    scenario,
                    "I can accept your formal offer and close now.",
                )
        if action.action_type == "reject":
            return (
                "Thought: I should not accept that price yet, but I can keep the negotiation alive.\n"
                "Talk: I cannot justify that ask. I need you to move closer before I can close.\n"
                "Action: [REJECT]"
            )
        if action.action_type == "quit":
            return (
                "Thought: There is no remaining buyer-feasible path.\n"
                "Talk: I do not think we can make this work at a sensible price.\n"
                "Action: [QUIT]"
            )
        price = action.target_price
        if price is None:
            price = min(float(scenario.buyer_budget), float(scenario.reference_price) * 0.45)
        price = PolicyVerifierGenerator._safe_offer_price(float(price), scenario, history)
        return format_action_message(
            "buyer",
            "BUY",
            round(float(price), 2),
            scenario,
            f"${price:.2f} is a serious cash offer. I need you to move meaningfully if we are closing today.",
        )
