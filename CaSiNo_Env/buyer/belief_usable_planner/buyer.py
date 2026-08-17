from __future__ import annotations

from typing import Any, Dict, List, Tuple

from CaSiNo_Env.buyer.base import ASTRABuyer
from CaSiNo_Env.buyer.belief_usable_planner.belief_model import MathPartnerBeliefModel
from CaSiNo_Env.buyer.belief_usable_planner.generator import FinalDialogueGenerator
from CaSiNo_Env.buyer.belief_usable_planner.llm_chooser import LLMFinalChooser
from CaSiNo_Env.buyer.belief_usable_planner.planner import RatioCandidatePlanner
from CaSiNo_Env.environment.actions import NegotiationAction
from CaSiNo_Env.environment.casino import complement_allocation, score_allocation


class BeliefUsablePlannerBuyer(ASTRABuyer):
    """Belief-usable planner without LLM-only final selection.

    Flow:
      1. update compact partner belief with math,
      2. generate candidates from different self/opponent weight ratios,
      3. score candidates with the belief model,
      4. choose one candidate with a deterministic chooser,
      5. ask the LLM generator to produce the final utterance.
    """

    name = "belief_usable_planner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief_model = MathPartnerBeliefModel()
        self.planner = RatioCandidatePlanner(candidate_top_k=self.candidate_top_k)
        self.generator = FinalDialogueGenerator(
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        belief_state = self.belief_model.update(scenario, history)
        candidates = self.planner.generate_candidates(
            scenario,
            belief_state,
            self.belief_model,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        chosen = self.planner.choose_final(candidates, round_id=round_id, max_rounds=max_rounds)
        accept_candidate = self._maybe_accept_seller_offer(scenario, history, round_id=round_id, max_rounds=max_rounds)
        if accept_candidate and accept_candidate["belief_score"] >= chosen.get("belief_score", -999):
            chosen = accept_candidate
        action, raw = self.generator.generate(
            scenario=scenario,
            history=history,
            belief=belief_state.to_json(),
            chosen=chosen,
            candidates=candidates,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        return action, {
            "prompt_type": self.name,
            "belief": belief_state.to_json(),
            "candidates": candidates,
            "chosen_candidate": chosen,
            "raw": raw,
            "training_hooks": {
                "planner_sft_input": "scenario + history + compact belief + candidate table",
                "planner_sft_target": "chosen candidate id / action type / allocation",
                "belief_sft_target": "posterior calibration from future P2 responses",
            },
        }

    def _maybe_accept_seller_offer(
        self,
        scenario: Any,
        history: List[Dict[str, Any]],
        *,
        round_id: int,
        max_rounds: int,
    ) -> Dict[str, Any] | None:
        last_seller_offer = None
        for turn in reversed(history):
            action = turn.get("action", {})
            if turn.get("actor") == "seller" and action.get("type") == "offer" and action.get("allocation"):
                last_seller_offer = action.get("allocation")
                break
        if not last_seller_offer:
            return None
        p1_alloc = complement_allocation(last_seller_offer, scenario.pool)
        self_score = score_allocation(p1_alloc, scenario.p1_values)
        self_norm = self_score / max(score_allocation(scenario.pool, scenario.p1_values), 1)
        time_ratio = round_id / max(max_rounds, 1)
        accept_threshold = 0.68 - 0.16 * time_ratio
        if self_norm < accept_threshold:
            return None
        return {
            "candidate_type": "accept_last_seller_offer",
            "action_type": "accept",
            "allocation": None,
            "partner_gets": last_seller_offer,
            "self_score": round(self_score, 4),
            "self_norm": round(self_norm, 4),
            "accept_prob": 1.0,
            "walkaway_risk": 0.0,
            "info_gain": 0.0,
            "belief_score": round(self_norm + 0.04 * time_ratio, 4),
        }


class BeliefUsableLLMChooserBuyer(BeliefUsablePlannerBuyer):
    """Same belief/candidate/scoring pipeline, but final action is chosen by LLM."""

    name = "belief_usable_llm_chooser"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_chooser = LLMFinalChooser(
            self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        belief_state = self.belief_model.update(scenario, history)
        candidates = self.planner.generate_candidates(
            scenario,
            belief_state,
            self.belief_model,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        deterministic_choice = self.planner.choose_final(candidates, round_id=round_id, max_rounds=max_rounds)
        accept_candidate = self._maybe_accept_seller_offer(scenario, history, round_id=round_id, max_rounds=max_rounds)
        if accept_candidate and accept_candidate["belief_score"] >= deterministic_choice.get("belief_score", -999):
            deterministic_choice = accept_candidate
        chosen = self.llm_chooser.choose(
            scenario=scenario,
            history=history,
            belief=belief_state.to_json(),
            candidates=candidates,
            deterministic_choice=deterministic_choice,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        action, raw = self.generator.generate(
            scenario=scenario,
            history=history,
            belief=belief_state.to_json(),
            chosen=chosen,
            candidates=candidates,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        return action, {
            "prompt_type": self.name,
            "belief": belief_state.to_json(),
            "candidates": candidates,
            "deterministic_choice": deterministic_choice,
            "chosen_candidate": chosen,
            "raw": raw,
            "training_hooks": {
                "llm_chooser_sft_input": "scenario + history + compact belief + candidate table + deterministic baseline",
                "llm_chooser_sft_target": "candidate_id + rationale",
                "planner_sft_target": "chosen candidate id / action type / allocation",
            },
        }
