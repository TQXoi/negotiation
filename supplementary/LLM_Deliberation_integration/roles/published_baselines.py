from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from astra_integration.environment.actions import DeliberationAction
from astra_integration.tools.deals import enumerate_deals, score_deal


class RepeatedRuleBasedRole:
    """FACT29's published repeated-rule baseline, adapted to this runner.

    Each acting player starts from the latest proposal and changes issues in a
    random priority order to its own best option until its private threshold is
    reached. The RNG is local and deterministically keyed by turn/player.
    """

    name = "repeated_rule_based"

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]],
        turn_id: int, max_turns: int, final_vote: bool,
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        spec = game.players[player]
        last_deal = next(
            (row["action"].get("deal") for row in reversed(history) if row["action"].get("deal")),
            None,
        )
        if last_deal is None:
            deal = max(enumerate_deals(game.issues), key=lambda d: score_deal(spec.scores, d))
        else:
            deal = dict(last_deal)
            rng = random.Random(f"{game.game_id}:{player}:{turn_id}")
            issue_order = list(game.issues)
            rng.shuffle(issue_order)
            for issue in issue_order:
                if score_deal(spec.scores, deal) >= spec.threshold:
                    break
                current = score_deal(spec.scores, deal)
                best_option, best_score = deal[issue], current
                for option in range(1, game.issues[issue] + 1):
                    candidate = {**deal, issue: option}
                    value = score_deal(spec.scores, candidate)
                    if value > best_score:
                        best_option, best_score = option, value
                deal[issue] = best_option
        action = DeliberationAction("propose", "I propose this revised package.", deal)
        return action, {
            "prompt_type": self.name,
            "source": "Carrasco Pollo et al., FACT29",
            "private_score": score_deal(spec.scores, deal),
            "final_vote": final_vote,
        }
