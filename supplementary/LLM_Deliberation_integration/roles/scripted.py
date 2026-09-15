from __future__ import annotations

from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction
from astra_integration.tools.deals import enumerate_deals, score_deal


class ScriptedRole:
    """Deterministic, model-free partner used for smoke tests and controls."""

    name = "scripted_threshold"

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int, max_turns: int, final_vote: bool
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        spec = game.players[player]
        last_deal = next((row["action"].get("deal") for row in reversed(history) if row["action"].get("deal")), None)
        if last_deal and score_deal(spec.scores, last_deal) >= spec.threshold:
            action = DeliberationAction("support", "I can support this complete package.", last_deal)
        else:
            deal = max(enumerate_deals(game.issues), key=lambda item: score_deal(spec.scores, item))
            action = DeliberationAction("propose", "I suggest this package as a basis for agreement.", deal)
        return action, {"prompt_type": self.name, "private_score": score_deal(spec.scores, action.deal) if action.deal else None}
