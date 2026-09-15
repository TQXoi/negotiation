from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction, action_format_instructions, parse_action


class DirectPromptRole:
    name = "direct_prompt"

    def __init__(self, model: Any, *, temperature: float = 0.0, max_tokens: int = 700):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int, max_turns: int, final_vote: bool
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        spec = game.players[player]
        opening_rule = ""
        if turn_id == 0 and player == game.p1:
            opening_rule = f"This is the fixed opening. You must propose exactly this deal: {game.initial_deal}."
        prompt = f"""{game.global_instructions}
You represent {player}. Your confidential role information follows:
{spec.private_instructions}
Your private option scores are {spec.scores}; minimum acceptable total is {spec.threshold}. Never reveal these numbers.
Turn {turn_id}/{max_turns}. Final official proposal: {final_vote}.
Recent public history: {json.dumps(history[-6:], ensure_ascii=False)}
Negotiate for a feasible agreement while following your incentive: {spec.incentive}.
{opening_rule}
{action_format_instructions(game.issues)}"""
        raw = self.model.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens, top_p=1.0)
        return parse_action(raw, game.issues), {"prompt_type": self.name, "raw": raw, "final_vote": final_vote}
