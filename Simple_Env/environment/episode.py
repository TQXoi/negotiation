import json
from dataclasses import asdict
from typing import Any, Dict

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import run_episode as _run_episode

from .types import RLVREvalEpisode, RLVRScenario


def run_episode(*args: Any, **kwargs: Any) -> RLVREvalEpisode:
    return _run_episode(*args, **kwargs)


def episode_result_record(episode: RLVREvalEpisode, seller_type: str, seller_model: str) -> Dict[str, Any]:
    scenario = episode.scenario
    return {
        "episode_result": {
            "buyer": episode.variant,
            "seller": seller_type,
            "seller_model": seller_model,
            "scenario": scenario.get("item_id"),
            "title": scenario.get("title"),
            "rollout": episode.rollout,
            "buyer_budget": scenario.get("buyer_budget"),
            "seller_cost": scenario.get("seller_cost"),
            "reference_price": scenario.get("reference_price"),
            "mutual_interest": episode.mi,
            "status": episode.status,
            "terminal_reason": episode.terminal_reason,
            "final_price": episode.final_price,
            "reward": episode.reward,
            "buyer_bargained_ratio": episode.buyer_bargained_ratio,
            "buyer_overshoot": episode.buyer_overshoot,
            "buyer_format_violation": episode.buyer_format_violation,
            "rounds": episode.rounds,
            "elapsed_seconds": episode.elapsed_seconds,
        }
    }


def print_episode_result(episode: RLVREvalEpisode, seller_type: str, seller_model: str) -> None:
    print(json.dumps(episode_result_record(episode, seller_type, seller_model), ensure_ascii=False), flush=True)


def episode_to_json(episode: RLVREvalEpisode) -> str:
    return json.dumps(asdict(episode), ensure_ascii=False)
