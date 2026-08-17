"""Simple Env buyers for frozen-planner identifiability experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVREvalEpisode,
    RLVRScenario,
    format_action_message,
    parse_action,
    validate_buyer_action,
)

from .buyer import ContinuousSolverBuyer
from .identifiable import (
    ActiveProbePlanner,
    ChangePointBehaviorBelief,
    default_behavior_hypotheses,
)


FROZEN_PLANNER_PATH = Path(__file__).resolve().parent / "identifiable/frozen_planner.json"


class ContinuousRuleFrozenEVBuyer(ContinuousSolverBuyer):
    """Stable comparison point whose PlannerParams are never retuned."""

    update_mode = "rule"
    planner_mode = "ev_parametric"

    def __init__(self, *args, continuous_planner_params_json: str | None = None, **kwargs):
        super().__init__(
            *args,
            continuous_planner_params_json=(
                continuous_planner_params_json or str(FROZEN_PLANNER_PATH)
            ),
            **kwargs,
        )


class ContinuousRuleRegimeMonitorBuyer(ContinuousRuleFrozenEVBuyer):
    """Frozen exploitation planner plus a public-action change-point monitor."""

    active_probe = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._behavior_key: str | None = None
        self._behavior_belief: ChangePointBehaviorBelief | None = None
        self._behavior_history_len = 0
        self._probe_planner = ActiveProbePlanner(
            information_weight=1.0,
            surplus_cost_weight=0.08,
        )

    @property
    def behavior_belief(self) -> ChangePointBehaviorBelief | None:
        return self._behavior_belief

    def _ensure_behavior_belief(self, scenario: RLVRScenario) -> None:
        if self._behavior_key == scenario.item_id and self._behavior_belief is not None:
            return
        self._behavior_key = scenario.item_id
        self._behavior_belief = ChangePointBehaviorBelief(
            default_behavior_hypotheses(float(scenario.reference_price)),
            hazard=0.03,
        )
        self._behavior_history_len = 0

    @staticmethod
    def _history_action(item: Dict[str, Any]) -> ParsedAction:
        action = item.get("action")
        if isinstance(action, dict):
            return ParsedAction(
                role=str(action.get("role") or item.get("role") or ""),
                action=str(action.get("action") or ""),
                price=action.get("price"),
                raw=str(action.get("raw") or item.get("message") or ""),
                valid=bool(action.get("valid", True)),
                item_spec=action.get("item_spec"),
            )
        return parse_action(str(item.get("role") or ""), str(item.get("message") or ""))

    def _observe_public_history(
        self,
        history: Sequence[Dict[str, Any]],
        *,
        max_turns: int,
    ) -> None:
        assert self._behavior_belief is not None
        if len(history) < self._behavior_history_len:
            # A fresh episode for the same public item: retain belief, reset the
            # episode-local cursor.
            self._behavior_history_len = 0
        for index in range(self._behavior_history_len, len(history)):
            item = history[index]
            if item.get("role") != "seller":
                continue
            seller_action = self._history_action(item)
            buyer_offer = None
            for prior_index in range(index - 1, -1, -1):
                prior = history[prior_index]
                if prior.get("role") != "buyer":
                    continue
                buyer_action = self._history_action(prior)
                if buyer_action.action == "offer" and buyer_action.price is not None:
                    buyer_offer = float(buyer_action.price)
                break
            if buyer_offer is None:
                continue
            if seller_action.action not in {"accept", "offer", "reject", "walk_away"}:
                continue
            self._behavior_belief.update(
                offer=buyer_offer,
                outcome=seller_action.action,
                round_id=int(item.get("round") or 1),
                max_turns=max_turns,
            )
        self._behavior_history_len = len(history)

    def observe_episode(self, episode: RLVREvalEpisode, *, max_turns: int) -> None:
        """Consume a terminal seller action that has no following buyer turn."""

        self._observe_public_history(episode.transcript, max_turns=max_turns)

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> tuple[ParsedAction, Dict[str, Any]]:
        self._ensure_behavior_belief(scenario)
        self._observe_public_history(history, max_turns=max_turns)
        action, trace = super().act(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
        )
        assert self._behavior_belief is not None
        probe_record = None
        if self.active_probe and round_id == 1:
            upper = min(float(scenario.buyer_budget), 0.95 * float(scenario.reference_price))
            offers = [
                round(ratio * float(scenario.reference_price), 2)
                for ratio in (0.30, 0.38, 0.46, 0.54, 0.62, 0.68, 0.74, 0.80, 0.88, 0.95)
                if ratio * float(scenario.reference_price) <= upper + 1e-9
            ]
            probe = self._probe_planner.choose_from_belief(
                belief=self._behavior_belief,
                candidate_offers=offers,
                buyer_budget=float(scenario.buyer_budget),
                round_id=round_id,
                max_turns=max_turns,
            )
            raw = format_action_message(
                "buyer",
                "BUY",
                round(probe.offer, 2),
                scenario,
                "I can make a concrete offer now. How much flexibility do you have around this number?",
            )
            parsed = parse_action("buyer", raw)
            overridden, validator = validate_buyer_action(
                parsed, scenario, history, round_id, max_turns
            )
            if overridden.valid:
                action = overridden
                probe_record = {
                    "applied": True,
                    "offer": probe.offer,
                    "information_gain_bits": probe.information_gain_bits,
                    "surplus_cost": probe.surplus_cost,
                    "score": probe.score,
                    "validator": validator,
                }
        trace["identifiable_behavior"] = {
            "oracle_truth_used_by_agent": False,
            "planner_frozen_config": str(FROZEN_PLANNER_PATH),
            "belief": self._behavior_belief.to_dict(),
            "active_probe": probe_record,
        }
        return action, trace


class ContinuousRuleActiveProbeBuyer(ContinuousRuleRegimeMonitorBuyer):
    """Frozen planner with a bounded first-round information-gain probe."""

    active_probe = True
