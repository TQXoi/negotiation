from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
from typing import Dict, List, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    load_existing_episodes,
    load_final_failed_episode_keys,
    summarize,
)

from .types import RLVREvalEpisode, RLVRScenario


def build_config(args) -> Dict:
    return {
        "script": "Simple_Env/eval.py",
        "paper": "Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards",
        "paper_alignment": {
            "test_instances": args.num_test_instances,
            "rollouts_per_instance": args.rollouts_per_instance,
            "max_turns": args.max_turns,
            "buyer_temperature": args.buyer_temperature,
            "seller_temperature": args.seller_temperature,
            "max_tokens": args.max_tokens,
            "seller_model": args.seller_model,
        },
        "simple_env": {
            "buyer_variants": args.buyer_variants,
            "seller_type": args.seller_type,
            "seller_persona": args.seller_persona,
            "candidate_offer_k": getattr(args, "candidate_offer_k", None),
            "prints_episode_result_after_each_episode": True,
        },
        "args": vars(args),
    }


def write_static_outputs(args, scenarios: Sequence[RLVRScenario]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(build_config(args), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "scenarios.jsonl").open("w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(asdict(scenario), ensure_ascii=False) + "\n")


def write_summary(args, episodes_by_variant: Dict[str, List[RLVREvalEpisode]]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": build_config(args),
        "by_variant": {variant: summarize(episodes) for variant, episodes in episodes_by_variant.items()},
        "paper_table1_reference": {
            "trained_qwen3_30b_a3b": {
                "reward": 0.7664,
                "deal_rate": 0.9199,
                "buyer_bargained_ratio": 0.8385,
                "price_overshoot_rate": 0.0010,
            }
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def append_episode(args, episode: RLVREvalEpisode) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{episode.variant}_episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
        f.flush()


def record_failed_episode(
    args,
    *,
    variant: str,
    scenario: RLVRScenario,
    rollout: int,
    exc: BaseException,
    attempt: int | None = None,
    final: bool = True,
) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "buyer": variant,
        "seller": args.seller_type,
        "variant": variant,
        "scenario": asdict(scenario),
        "rollout": rollout,
        "attempt": attempt,
        "final": final,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }
    with (out / "failed_episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


__all__ = [
    "append_episode",
    "build_config",
    "load_existing_episodes",
    "load_final_failed_episode_keys",
    "record_failed_episode",
    "write_static_outputs",
    "write_summary",
]
