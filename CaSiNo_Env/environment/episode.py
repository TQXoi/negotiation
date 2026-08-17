from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from CaSiNo_Env.environment.actions import NegotiationAction
from CaSiNo_Env.environment.casino import (
    CasinoScenario,
    complement_allocation,
    normalize_allocation,
    score_allocation,
    scenario_to_json,
)


def visible_history(history: List[Dict[str, Any]]) -> str:
    lines = []
    for turn in history:
        actor = turn["actor"]
        action = turn["action"]
        message = action.get("message") or ""
        allocation = action.get("allocation")
        alloc_text = f" allocation={allocation}" if allocation else ""
        lines.append(f"{actor}: {action.get('type')} - {message}{alloc_text}")
    return "\n".join(lines) if lines else "No dialogue yet."


def final_scores_from_offer(
    scenario: CasinoScenario,
    proposer: str,
    proposer_allocation: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    proposer_allocation = normalize_allocation(proposer_allocation)
    other_allocation = complement_allocation(proposer_allocation, scenario.pool)
    if proposer == "buyer":
        p1_allocation, p2_allocation = proposer_allocation, other_allocation
    else:
        p1_allocation, p2_allocation = other_allocation, proposer_allocation
    p1_score = score_allocation(p1_allocation, scenario.p1_values)
    p2_score = score_allocation(p2_allocation, scenario.p2_values)
    return p1_allocation, p2_allocation, p1_score, p2_score


def run_episode(
    scenario: CasinoScenario,
    buyer: Any,
    seller: Any,
    *,
    max_rounds: int = 12,
    rollout: int = 0,
) -> Dict[str, Any]:
    start = time.time()
    history: List[Dict[str, Any]] = []
    last_offer: Optional[Dict[str, Any]] = None
    status = "walk_away"
    terminal_reason = "max_rounds"
    p1_allocation = None
    p2_allocation = None
    p1_score = 0
    p2_score = 0

    for round_id in range(1, max_rounds + 1):
        buyer_action, buyer_trace = buyer.act(scenario=scenario, history=history, round_id=round_id, max_rounds=max_rounds)
        history.append({"round": round_id, "actor": "buyer", "action": buyer_action.to_json(), "trace": buyer_trace})
        if buyer_action.type == "walk_away":
            terminal_reason = "buyer_walk_away"
            break
        if buyer_action.type == "accept" and last_offer and last_offer["actor"] == "seller":
            p1_allocation, p2_allocation, p1_score, p2_score = final_scores_from_offer(
                scenario, "seller", last_offer["allocation"]
            )
            status = "agreement"
            terminal_reason = "buyer_accept"
            break
        if buyer_action.type == "offer" and buyer_action.allocation is not None:
            last_offer = {"actor": "buyer", "allocation": buyer_action.allocation}

        seller_action, seller_trace = seller.act(scenario=scenario, history=history, round_id=round_id, max_rounds=max_rounds)
        history.append({"round": round_id, "actor": "seller", "action": seller_action.to_json(), "trace": seller_trace})
        if seller_action.type == "walk_away":
            terminal_reason = "seller_walk_away"
            break
        if seller_action.type == "accept" and last_offer and last_offer["actor"] == "buyer":
            p1_allocation, p2_allocation, p1_score, p2_score = final_scores_from_offer(
                scenario, "buyer", last_offer["allocation"]
            )
            status = "agreement"
            terminal_reason = "seller_accept"
            break
        if seller_action.type == "offer" and seller_action.allocation is not None:
            last_offer = {"actor": "seller", "allocation": seller_action.allocation}

    return {
        "buyer": getattr(buyer, "name", buyer.__class__.__name__),
        "seller": getattr(seller, "name", seller.__class__.__name__),
        "scenario": scenario.scenario_id,
        "dialogue_id": scenario.dialogue_id,
        "rollout": rollout,
        "status": status,
        "terminal_reason": terminal_reason,
        "rounds": max((turn["round"] for turn in history), default=0),
        "p1_score": p1_score,
        "p2_score": p2_score,
        "p1_allocation": p1_allocation,
        "p2_allocation": p2_allocation,
        "p1_max_score": scenario.max_score,
        "score_ratio_p1": round(p1_score / scenario.max_score, 6) if scenario.max_score else 0.0,
        "history": history,
        "scenario_context": scenario_to_json(scenario),
        "elapsed_seconds": round(time.time() - start, 3),
    }
