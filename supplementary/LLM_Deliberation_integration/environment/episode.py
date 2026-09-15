from __future__ import annotations

import random
import time
from typing import Any, Dict, List

from astra_integration.tools.deals import score_deal


def _schedule(players: List[str], p1: str, discussion_turns: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    result: List[str] = []
    last = p1
    while len(result) < discussion_turns:
        block = list(players)
        rng.shuffle(block)
        for _ in range(30):
            if block[0] != last and block[-1] != p1:
                break
            rng.shuffle(block)
        result.extend(block)
        last = block[-1]
    return result[:discussion_turns]


def run_episode(game: Any, roles: Dict[str, Any], *, discussion_turns: int = 24, history_window: int = 6, seed: int = 0, rollout: int = 0, communication_free: bool = False) -> Dict[str, Any]:
    start = time.time()
    history: List[Dict[str, Any]] = []

    def execute(actor: str, turn: int, final_vote: bool = False) -> None:
        public_history = [
            {
                "turn": row["turn"],
                "actor": row["actor"],
                "action": {
                    "type": row["action"].get("type"),
                    "message": row["action"].get("message", ""),
                    "deal": row["action"].get("deal"),
                },
            }
            for row in history
        ]
        action, trace = roles[actor].act(
            # Roles receive the full event stream so a persistent/continuous
            # belief never forgets evidence. Prompt-based roles and generators
            # enforce their own public history window. Private traces are never
            # passed back into another role.
            game=game, player=actor, history=public_history, turn_id=turn, max_turns=discussion_turns + 2, final_vote=final_vote
        )
        history.append({"turn": turn, "actor": actor, "action": action.to_json(), "trace": trace})

    # Multi-agent paper protocol has an extra P1 opening. García et al.'s
    # communication-free implementation instead assigns the R ordinary slots
    # to P1 and then asks P1 for the final proposal (R+1 total calls).
    if communication_free:
        discussion_schedule = [game.p1] * discussion_turns
        start_index = 0
    else:
        execute(game.p1, 0)
        discussion_schedule = _schedule(list(game.players), game.p1, discussion_turns, seed)
        start_index = 1
    for index, actor in enumerate(discussion_schedule, start=start_index):
        execute(actor, index)
    final_turn = discussion_turns if communication_free else discussion_turns + 1
    execute(game.p1, final_turn, final_vote=True)

    final_deal = history[-1]["action"].get("deal")
    scores = {name: score_deal(spec.scores, final_deal) if final_deal else 0 for name, spec in game.players.items()}
    supporters = [name for name, spec in game.players.items() if final_deal and scores[name] >= spec.threshold]
    required_parties = {game.p1, game.p2}
    agreement_n_minus_1 = len(supporters) >= len(game.players) - 1 and required_parties.issubset(supporters)
    agreement_unanimous = len(supporters) == len(game.players)
    complete_deals = [row["action"].get("deal") for row in history if row["action"].get("deal")]
    any_success = False
    for deal in complete_deals:
        deal_supporters = [name for name, spec in game.players.items() if score_deal(spec.scores, deal) >= spec.threshold]
        if len(deal_supporters) >= len(game.players) - 1 and required_parties.issubset(deal_supporters):
            any_success = True
            break
    return {
        "game": game.game_id,
        "rollout": rollout,
        "seed": seed,
        "p1": game.p1,
        "p2": game.p2,
        "final_deal": final_deal,
        "scores": scores,
        "thresholds": {name: spec.threshold for name, spec in game.players.items()},
        "supporters": supporters,
        "support_count": len(supporters),
        "agreement_n_minus_1": agreement_n_minus_1,
        "agreement_unanimous": agreement_unanimous,
        "any_success": any_success,
        "parse_errors": sum(bool(row["action"].get("parse_error")) for row in history),
        "history": history,
        "elapsed_seconds": round(time.time() - start, 3),
        "communication_free": communication_free,
    }
