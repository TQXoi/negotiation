"""Hybrid rule-based plus LLM-assisted belief update."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import ParsedAction, RLVRScenario

from ..belief_model.base import BeliefState
from ..belief_model.posterior import PosteriorBeliefModel
from .llm_assisted import LLMAssistedUpdater
from .rule_based import RuleBasedBeliefUpdater


class HybridBeliefUpdater:
    """Apply rule updates first, then bounded LLM semantic deltas."""

    def __init__(
        self,
        rule_updater: RuleBasedBeliefUpdater,
        llm_updater: Optional[LLMAssistedUpdater] = None,
    ):
        self.rule_updater = rule_updater
        self.llm_updater = llm_updater
        self.last_llm_update = None

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
        state = self.rule_updater.update(
            belief_model=belief_model,
            scenario=scenario,
            history=history,
            buyer_action=buyer_action,
            seller_action=seller_action,
            round_id=round_id,
            max_turns=max_turns,
            candidate_prices=candidate_prices,
        )
        if self.llm_updater is None or seller_action is None:
            return state
        self.last_llm_update = self.llm_updater.extract(
            state=state,
            scenario=scenario,
            history=history,
            buyer_action=buyer_action,
            seller_action=seller_action,
            round_id=round_id,
            max_turns=max_turns,
        )
        return self.llm_updater.apply(state, self.last_llm_update)
