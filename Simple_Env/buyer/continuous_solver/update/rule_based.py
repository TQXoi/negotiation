"""Rule-based belief updates from formal buyer/seller actions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    last_price,
)

from ..belief_model.base import BeliefEvidence, BeliefState, clamp
from ..belief_model.posterior import PosteriorBeliefModel


class RuleBasedBeliefUpdater:
    """Update reservation posterior and scalar risk state from observed actions."""

    def __init__(self, *, ema_alpha: float = 0.35):
        self.ema_alpha = ema_alpha

    def update(
        self,
        *,
        belief_model: PosteriorBeliefModel,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        buyer_action: Optional[ParsedAction],
        seller_action: Optional[ParsedAction],
        round_id: int,
        max_turns: int,
        candidate_prices: Iterable[float],
    ) -> BeliefState:
        state = belief_model.state
        state.round_id = round_id
        if seller_action is None:
            return belief_model.refresh_summary(candidate_prices)

        buyer_offer = buyer_action.price if buyer_action and buyer_action.price is not None else last_price(history, "buyer")
        round_pressure = clamp(round_id / max(1, max_turns))
        action = seller_action.action

        if action == "deal" and buyer_offer is not None:
            belief_model.posterior.update_accept(float(buyer_offer))
            state.interaction.quit_risk = clamp(state.interaction.quit_risk - 0.20)
            state.interaction.finality_prob = clamp(state.interaction.finality_prob + 0.20)
            state.add_evidence(
                BeliefEvidence(
                    round_id=round_id,
                    source="rule",
                    evidence_type="seller_accept",
                    text=f"Seller accepted buyer offer {buyer_offer:.2f}.",
                    effect="shift reservation posterior below accepted offer",
                    confidence=0.95,
                )
            )
        elif action in {"sell", "offer"} and seller_action.price is not None:
            self._update_seller_counter(
                belief_model=belief_model,
                state=state,
                buyer_offer=buyer_offer,
                seller_offer=float(seller_action.price),
                round_id=round_id,
                round_pressure=round_pressure,
            )
        elif action == "reject" and buyer_offer is not None:
            belief_model.posterior.update_reject(float(buyer_offer))
            state.interaction.quit_risk = clamp(state.interaction.quit_risk + 0.025 + 0.025 * round_pressure)
            state.interaction.patience = clamp(state.interaction.patience - 0.03)
            state.add_evidence(
                BeliefEvidence(
                    round_id=round_id,
                    source="rule",
                    evidence_type="seller_reject",
                    text=f"Seller rejected buyer offer {buyer_offer:.2f}.",
                    effect="shift reservation posterior above rejected offer and increase quit risk",
                    confidence=0.75,
                )
            )
        elif action == "quit":
            if buyer_offer is not None:
                belief_model.posterior.update_reject(float(buyer_offer))
            state.interaction.quit_risk = clamp(state.interaction.quit_risk + 0.35)
            state.interaction.finality_prob = clamp(state.interaction.finality_prob + 0.25)
            state.interaction.patience = clamp(state.interaction.patience - 0.35)
            state.add_evidence(
                BeliefEvidence(
                    round_id=round_id,
                    source="rule",
                    evidence_type="seller_quit",
                    text="Seller quit the negotiation.",
                    effect="raise quit risk and reinforce that previous offer was likely too low",
                    confidence=0.90,
                )
            )

        return belief_model.refresh_summary(candidate_prices)

    def _update_seller_counter(
        self,
        *,
        belief_model: PosteriorBeliefModel,
        state: BeliefState,
        buyer_offer: Optional[float],
        seller_offer: float,
        round_id: int,
        round_pressure: float,
    ) -> None:
        prior_offer = state.concession.last_seller_offer
        belief_model.posterior.update_counter(
            buyer_offer,
            seller_offer,
            round_pressure=round_pressure,
            buyer_budget=float(state.buyer_budget),
        )
        state.concession.previous_seller_offer = prior_offer
        state.concession.last_seller_offer = seller_offer

        if prior_offer is not None and seller_offer < prior_offer:
            relative_drop = (prior_offer - seller_offer) / max(prior_offer, 1.0)
            state.concession.num_price_drops += 1
            state.concession.slope = (
                (1.0 - self.ema_alpha) * state.concession.slope + self.ema_alpha * relative_drop
            )
            state.interaction.finality_prob = clamp(state.interaction.finality_prob - 0.10)
            state.interaction.quit_risk = clamp(state.interaction.quit_risk - 0.05)
            effect = "increase flexibility and lower finality because seller conceded"
            evidence_type = "seller_concession"
            confidence = 0.85
        else:
            state.interaction.finality_prob = clamp(state.interaction.finality_prob + 0.035 + 0.04 * round_pressure)
            state.interaction.quit_risk = clamp(state.interaction.quit_risk + 0.015 * round_pressure)
            effect = "increase finality because seller did not reduce price"
            evidence_type = "seller_counter_no_drop"
            confidence = 0.65

        state.add_evidence(
            BeliefEvidence(
                round_id=round_id,
                source="rule",
                evidence_type=evidence_type,
                text=f"Seller countered at {seller_offer:.2f}.",
                effect=effect,
                confidence=confidence,
            )
        )
