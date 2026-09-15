from __future__ import annotations

from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction
from astra_integration.framework.belief_model import MultiPartyBeliefModel
from astra_integration.framework.generator import LockedDialogueGenerator
from astra_integration.framework.planner import MultiPartyCandidatePlanner


class BeliefPlannerGeneratorRole:
    name = "belief_planner_generator"

    def __init__(self, model: Any | None = None, *, candidate_top_k: int = 8, temperature: float = 0.0, max_tokens: int = 700):
        self.belief_model = MultiPartyBeliefModel()
        self.planner = MultiPartyCandidatePlanner(candidate_top_k)
        self.generator = LockedDialogueGenerator(model, temperature, max_tokens)

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int, max_turns: int, final_vote: bool
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        if turn_id == 0 and player == game.p1:
            chosen = {
                "candidate_type": "paper_fixed_opening",
                "action_type": "propose",
                "deal": dict(game.initial_deal),
            }
            action, raw = self.generator.generate(
                game=game, player=player, history=history, belief={}, chosen=chosen,
            )
            return action, {
                "prompt_type": self.name, "belief": {}, "candidates": [chosen],
                "chosen_candidate": chosen, "raw": raw, "final_vote": final_vote,
            }
        beliefs = self.belief_model.update(game, player, history)
        full_belief_json = {name: value.to_json() for name, value in beliefs.items()}
        generator_belief = {
            name: {
                "option_posteriors": value.option_posteriors(),
                "threshold_ratio": round(value.threshold_ratio, 4),
                "flexibility": round(value.flexibility, 4),
                "adversarial_probability": round(value.adversarial_probability, 4),
                "recent_evidence": value.evidence[-3:],
            }
            for name, value in beliefs.items()
        }
        candidates = self.planner.generate(game, player, beliefs, turn_id, max_turns)
        chosen = self.planner.choose(candidates)
        if final_vote and player == game.p1:
            chosen["action_type"] = "propose"
        action, raw = self.generator.generate(
            game=game,
            player=player,
            history=history,
            belief=generator_belief,
            chosen=chosen,
        )
        return action, {
            "prompt_type": self.name,
            "belief": full_belief_json,
            "candidates": candidates,
            "chosen_candidate": chosen,
            "raw": raw,
            "final_vote": final_vote,
        }
