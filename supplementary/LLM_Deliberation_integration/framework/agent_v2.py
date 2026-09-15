from __future__ import annotations

from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction
from astra_integration.framework.belief_model import MultiPartyBeliefModel
from astra_integration.framework.generator import LockedDialogueGenerator
from astra_integration.framework.planner_v2 import CoalitionAwarePlannerV2


class BeliefPlannerGeneratorRoleV2:
    name = "belief_planner_generator_v2"

    def __init__(self, model: Any | None = None, *, candidate_top_k: int = 8, temperature: float = 0.0, max_tokens: int = 1600):
        self.belief_model = MultiPartyBeliefModel()
        self.planner = CoalitionAwarePlannerV2(candidate_top_k)
        self.generator = LockedDialogueGenerator(model, temperature, max_tokens)

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int,
        max_turns: int, final_vote: bool,
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        if turn_id == 0 and player == game.p1:
            chosen = {"candidate_type": "paper_fixed_opening", "action_type": "propose", "deal": dict(game.initial_deal), "planner_version": self.planner.version}
            action, raw = self.generator.generate(game=game, player=player, history=history, belief={}, chosen=chosen)
            return action, {"prompt_type": self.name, "belief": {}, "candidates": [chosen], "chosen_candidate": chosen, "raw": raw, "final_vote": final_vote}
        beliefs = self.belief_model.update(game, player, history)
        full_belief = {name: value.to_json() for name, value in beliefs.items()}
        compact_belief = {
            name: {"option_posteriors": value.option_posteriors(), "threshold_ratio": round(value.threshold_ratio, 4), "flexibility": round(value.flexibility, 4), "recent_evidence": value.evidence[-3:]}
            for name, value in beliefs.items()
        }
        candidates = self.planner.generate(game, player, beliefs, history, turn_id, max_turns, final_vote)
        chosen = self.planner.choose(candidates)
        action, raw = self.generator.generate(game=game, player=player, history=history, belief=compact_belief, chosen=chosen)
        return action, {
            "prompt_type": self.name, "belief": full_belief, "candidates": candidates,
            "chosen_candidate": chosen, "raw": raw, "final_vote": final_vote,
            "planner_version": self.planner.version,
        }
