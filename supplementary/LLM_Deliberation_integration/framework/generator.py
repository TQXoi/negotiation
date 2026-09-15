from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction, action_format_instructions, parse_action


class LockedDialogueGenerator:
    def __init__(self, model: Any | None, temperature: float = 0.0, max_tokens: int = 700):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, *, game: Any, player: str, history: List[Dict[str, Any]], belief: Dict[str, Any], chosen: Dict[str, Any]) -> Tuple[DeliberationAction, str]:
        locked_deal = chosen.get("deal")
        if self.model is None:
            action = DeliberationAction(chosen.get("action_type", "propose"), "I propose this package for consideration.", locked_deal)
            return action, json.dumps(action.to_json())
        visible = [{"actor": row["actor"], "action": row["action"]} for row in history[-6:]]
        prompt = f"""You represent {player} in a multi-party negotiation.
Recent public history: {json.dumps(visible, ensure_ascii=False)}
Compact opponent belief: {json.dumps(belief, ensure_ascii=False)}
The planner has LOCKED this action: {json.dumps(chosen, ensure_ascii=False)}
Express it briefly and persuasively. Never reveal scores or change the locked action/deal.
{action_format_instructions(game.issues)}"""
        raw = self.model.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens, top_p=1.0)
        parsed = parse_action(raw, game.issues)
        if parsed.parse_error or parsed.type != chosen.get("action_type") or parsed.deal != locked_deal:
            parsed = DeliberationAction(chosen.get("action_type", "propose"), parsed.message or "I propose this package.", locked_deal, raw, "action_lock_repair")
        return parsed, raw
