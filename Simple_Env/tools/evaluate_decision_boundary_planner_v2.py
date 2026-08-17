#!/usr/bin/env python3
"""Paired hidden-profile evaluation for fine-tuned Planner V2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.belief import StrategicReadinessMixtureBeliefUpdater
from framework.planner import BehavioralFrontierPlanner
from framework.trainable_planner import TrainableDecisionBoundaryPlanner
from Simple_Env.buyer.universal_framework import UniversalFrameworkV59Buyer
from Simple_Env.environment import run_episode
from Simple_Env.buyer.continuous_solver.identifiable import UtilityProfile
from Simple_Env.tools.train_universal_planner_cem import (
    NullPublicLanguageClient,
    ScriptedSeller,
    bootstrap_profile_ci,
    profile_suite,
    scenario_for,
)


def make_buyer(planner: Any, calibration_json: Path) -> UniversalFrameworkV59Buyer:
    client = NullPublicLanguageClient()
    buyer = UniversalFrameworkV59Buyer(client=client, max_tokens=400, temperature=0.0, top_p=1.0)
    buyer.engine.structured_updater = StrategicReadinessMixtureBeliefUpdater.from_calibration_json(
        calibration_json
    )
    buyer.engine.planner = planner
    return buyer


def evaluate_system(
    system: str,
    planner_factory,
    *,
    calibration_json: Path,
    rollouts: int,
    seed: int,
    profiles: list[UtilityProfile],
) -> list[dict[str, Any]]:
    records = []
    for profile_index, profile in enumerate(profiles):
        scenario = scenario_for(profile)
        for rollout in range(rollouts):
            common_seed = seed + profile_index * 1_000_003 + rollout * 10_007
            episode = run_episode(
                variant=system,
                scenario=scenario,
                rollout=rollout,
                buyer=make_buyer(planner_factory(), calibration_json),
                seller=ScriptedSeller(profile, 100.0, common_seed),
                max_turns=6,
            )
            actions = [
                (turn["action"]["action"], turn["action"].get("price"))
                for turn in episode.transcript if turn["role"] == "buyer"
            ]
            records.append({
                "system": system,
                "profile_id": profile.profile_id,
                "rollout": rollout,
                "reward": float(episode.reward),
                "deal": episode.status == "deal",
                "terminal_reason": episode.terminal_reason,
                "buyer_actions": actions,
                "episode": asdict(episode),
            })
    return records


def evaluation_profiles(split: str) -> list[UtilityProfile]:
    if split != "final":
        return profile_suite(split)
    ratios = (0.325, 0.525, 0.775, 0.900)
    policies = (
        ("adaptive", 0.82, 0.06, 0.30),
        ("stubborn", 0.52, 0.24, 0.82),
        ("patient_strategic", 0.90, 0.03, 0.65),
    )
    return [
        UtilityProfile(
            profile_id=f"final_{name}_{ratio:.3f}", cost_ratio=ratio,
            counter_propensity=counter, quit_bias=quit, finality=finality,
        )
        for ratio in ratios
        for name, counter, quit, finality in policies
    ]


def summarize(trained: list[dict[str, Any]], base: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    left = {(r["profile_id"], r["rollout"]): r for r in trained}
    right = {(r["profile_id"], r["rollout"]): r for r in base}
    reward_by_profile: dict[str, list[float]] = {}
    agreement_by_profile: dict[str, list[float]] = {}
    flips = first_flips = 0
    for key in sorted(set(left) & set(right)):
        profile, _ = key
        l, r = left[key], right[key]
        reward_by_profile.setdefault(profile, []).append(l["reward"] - r["reward"])
        agreement_by_profile.setdefault(profile, []).append(float(l["deal"]) - float(r["deal"]))
        flips += l["buyer_actions"] != r["buyer_actions"]
        first_flips += l["buyer_actions"][:1] != r["buyer_actions"][:1]
    reward = {key: statistics.mean(values) for key, values in reward_by_profile.items()}
    agreement = {key: statistics.mean(values) for key, values in agreement_by_profile.items()}
    paired = len(set(left) & set(right))
    result = {
        "profile_clusters": len(reward),
        "paired_episodes": paired,
        "trained_mean_reward": statistics.mean(r["reward"] for r in trained),
        "base_mean_reward": statistics.mean(r["reward"] for r in base),
        "mean_trained_reward_delta": statistics.mean(reward.values()),
        "reward_profile_bootstrap_95_ci": bootstrap_profile_ci(reward, samples, seed),
        "trained_agreement_rate": statistics.mean(float(r["deal"]) for r in trained),
        "base_agreement_rate": statistics.mean(float(r["deal"]) for r in base),
        "mean_trained_agreement_delta": statistics.mean(agreement.values()),
        "agreement_profile_bootstrap_95_ci": bootstrap_profile_ci(agreement, samples, seed + 1),
        "action_sequence_flip_rate": flips / max(1, paired),
        "first_action_flip_rate": first_flips / max(1, paired),
        "profile_reward_deltas": reward,
    }
    result["planner_gate_passed"] = (
        result["action_sequence_flip_rate"] > 0.0
        and result["mean_trained_reward_delta"] > 0.0
        and result["reward_profile_bootstrap_95_ci"][0] > 0.0
        and result["agreement_profile_bootstrap_95_ci"][0] >= -0.05
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--profile-split", choices=("dev", "test", "final"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--require-direct-evidence-for-override", action="store_true")
    parser.add_argument("--minimum-override-advantage", type=float, default=0.0)
    parser.add_argument("--minimum-accept-override-advantage", type=float)
    parser.add_argument("--minimum-continue-over-accept-advantage", type=float)
    parser.add_argument("--max-initial-proxy-jump", type=float)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = evaluation_profiles(args.profile_split)
    trained = evaluate_system(
        "planner_v2_finetuned", lambda: TrainableDecisionBoundaryPlanner(
            args.checkpoint,
            require_direct_evidence_for_override=args.require_direct_evidence_for_override,
            minimum_override_advantage=args.minimum_override_advantage,
            minimum_accept_override_advantage=args.minimum_accept_override_advantage,
            minimum_continue_over_accept_advantage=args.minimum_continue_over_accept_advantage,
            max_initial_proxy_jump=args.max_initial_proxy_jump,
        ),
        calibration_json=args.calibration_json, rollouts=args.rollouts, seed=args.seed,
        profiles=profiles,
    )
    base = evaluate_system(
        "planner_v5_9_base", BehavioralFrontierPlanner,
        calibration_json=args.calibration_json, rollouts=args.rollouts, seed=args.seed,
        profiles=profiles,
    )
    comparison = summarize(trained, base, args.bootstrap_samples, args.seed)
    (args.output_dir / "paired_episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trained + base),
        encoding="utf-8",
    )
    (args.output_dir / "integrated_metrics.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
