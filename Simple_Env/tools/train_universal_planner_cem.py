#!/usr/bin/env python3
"""CEM policy search for the normalized belief-usable frontier planner.

The language generator and calibrated belief are frozen.  Only interpretable,
environment-independent planner weights are optimized in a scripted hidden-
profile simulator, then evaluated on held-out utility/policy profile clusters.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    format_action_message,
    last_price,
    parse_action,
)
from framework.belief import StrategicReadinessMixtureBeliefUpdater
from framework.planner import BehavioralFrontierPlanner, BehavioralFrontierPlannerConfig
from Simple_Env.buyer.continuous_solver.identifiable import UtilityProfile
from Simple_Env.buyer.universal_framework import UniversalFrameworkV59Buyer
from Simple_Env.environment import run_episode
from Simple_Env.environment.types import RLVRScenario


PARAM_BOUNDS = {
    "lookahead_discount": (0.30, 0.98),
    "maximum_lookahead_bonus": (0.04, 0.40),
    "feasibility_floor": (0.02, 0.50),
    "uncertain_accept_option_weight": (0.00, 0.20),
    "frontier_gate_rate": (0.20, 1.60),
    "frontier_acceptance_discount": (0.20, 0.98),
    "frontier_score_penalty": (0.04, 0.65),
    "frontier_margin": (0.00, 0.08),
}


class NullPublicLanguageClient:
    def generate(self, prompt: str, **_: Any) -> str:
        if "extract bounded opponent-preference evidence" in prompt:
            return json.dumps({
                "reservation_ratio_range": None,
                "issue_signals": [],
                "patience": None,
                "quit_risk": None,
                "evidence": [],
            })
        return "This is a concrete proposal; please indicate whether there is room to move."


class ScriptedSeller:
    def __init__(self, profile: UtilityProfile, reference_price: float, seed: int):
        self.model = profile.behavior(reference_price)
        self.rng = random.Random(seed)

    def act(self, *, scenario, history, round_id, max_turns):
        buyer_offer = float(last_price(history, "buyer") or 0.0)
        distribution = self.model.response_distribution(
            buyer_offer, round_id=round_id, max_turns=max_turns
        )
        accept = distribution["accept"] if buyer_offer > scenario.seller_cost + 1e-9 else 0.0
        counter = distribution["counter"]
        quit_probability = distribution["quit"]
        total = accept + counter + quit_probability
        accept, counter = accept / total, counter / total
        draw = self.rng.random()
        if draw < accept:
            raw = format_action_message("seller", "DEAL", buyer_offer, scenario, "I accept.")
        elif draw < accept + counter:
            progress = round_id / max(1, max_turns)
            target = scenario.seller_cost + (scenario.reference_price - scenario.seller_cost) * max(
                0.08, 0.62 * (1.0 - progress)
            )
            price = min(
                scenario.reference_price,
                max(scenario.seller_cost + 0.01, buyer_offer + 0.01, target),
            )
            raw = format_action_message(
                "seller", "SELL", round(price, 2), scenario,
                "I cannot accept that level, but I can make this counteroffer.",
            )
        else:
            raw = "Thought: stop\nTalk: I cannot continue on these terms.\nAction: [QUIT]"
        return parse_action("seller", raw), raw


def profile_suite(split: str) -> list[UtilityProfile]:
    if split == "train":
        ratios = (0.35, 0.50, 0.65, 0.80)
        policies = (
            ("patient", 0.92, 0.02, 0.20),
            ("neutral", 0.78, 0.08, 0.25),
            ("firm", 0.48, 0.28, 0.75),
        )
    elif split == "dev":
        ratios = (0.425, 0.575, 0.725, 0.875)
        policies = (
            ("flexible", 0.86, 0.05, 0.12),
            ("hardball", 0.64, 0.14, 0.58),
            ("impatient", 0.36, 0.38, 0.88),
        )
    elif split == "test":
        ratios = (0.275, 0.475, 0.675, 0.925)
        policies = (
            ("very_patient", 0.96, 0.01, 0.08),
            ("strategic", 0.70, 0.10, 0.68),
            ("very_firm", 0.30, 0.46, 0.94),
        )
    else:
        raise ValueError(split)
    return [
        UtilityProfile(
            profile_id=f"{split}_{policy_name}_{ratio:.3f}",
            cost_ratio=ratio,
            counter_propensity=counter,
            quit_bias=quit_bias,
            finality=finality,
        )
        for ratio in ratios
        for policy_name, counter, quit_bias, finality in policies
    ]


def scenario_for(profile: UtilityProfile) -> RLVRScenario:
    return RLVRScenario(
        item_id=f"planner_{profile.profile_id}",
        title="Normalized hidden-profile planner training item",
        buyer_budget=95.0,
        seller_cost=100.0 * profile.cost_ratio,
        reference_price=100.0,
        category="normalized_profile",
        codename="normalized_item",
        description="No benchmark-specific lexical cue.",
        features="Mechanism training only.",
        quantity=1,
    )


def config_from_params(params: dict[str, float]) -> BehavioralFrontierPlannerConfig:
    base = asdict(BehavioralFrontierPlannerConfig())
    base.update({key: float(value) for key, value in params.items() if key in PARAM_BOUNDS})
    return BehavioralFrontierPlannerConfig(**base)


def make_buyer(
    config: BehavioralFrontierPlannerConfig,
    calibration_json: Path,
) -> UniversalFrameworkV59Buyer:
    client = NullPublicLanguageClient()
    buyer = UniversalFrameworkV59Buyer(
        client=client, max_tokens=400, temperature=0.0, top_p=1.0
    )
    buyer.engine.structured_updater = (
        StrategicReadinessMixtureBeliefUpdater.from_calibration_json(calibration_json)
    )
    buyer.engine.planner = BehavioralFrontierPlanner(config)
    return buyer


def evaluate(
    params: dict[str, float],
    *,
    profiles: Iterable[UtilityProfile],
    rollouts: int,
    seed: int,
    calibration_json: Path,
    keep_episodes: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    config = config_from_params(params)
    records = []
    for profile_index, profile in enumerate(profiles):
        scenario = scenario_for(profile)
        for rollout in range(rollouts):
            common_seed = seed + profile_index * 1_000_003 + rollout * 10_007
            episode = run_episode(
                variant="planner_cem",
                scenario=scenario,
                rollout=rollout,
                buyer=make_buyer(config, calibration_json),
                seller=ScriptedSeller(profile, 100.0, common_seed),
                max_turns=6,
            )
            records.append({
                "profile_id": profile.profile_id,
                "rollout": rollout,
                "reward": float(episode.reward),
                "deal": episode.status == "deal",
                "terminal_reason": episode.terminal_reason,
                **({"episode": asdict(episode)} if keep_episodes else {}),
            })
    return statistics.mean(row["reward"] for row in records), records


def sample_candidate(
    mean: dict[str, float], std: dict[str, float], rng: random.Random
) -> dict[str, float]:
    return {
        key: max(PARAM_BOUNDS[key][0], min(PARAM_BOUNDS[key][1], rng.gauss(mean[key], std[key])))
        for key in PARAM_BOUNDS
    }


def bootstrap_profile_ci(deltas: dict[str, float], samples: int, seed: int) -> list[float]:
    values = list(deltas.values())
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def summarize_pair(
    trained: list[dict[str, Any]], current: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    left = {(row["profile_id"], row["rollout"]): row for row in trained}
    right = {(row["profile_id"], row["rollout"]): row for row in current}
    by_profile: dict[str, list[float]] = {}
    agreement_by_profile: dict[str, list[float]] = {}
    action_sequence_flips = 0
    for key in sorted(set(left) & set(right)):
        profile, _ = key
        by_profile.setdefault(profile, []).append(left[key]["reward"] - right[key]["reward"])
        agreement_by_profile.setdefault(profile, []).append(float(left[key]["deal"]) - float(right[key]["deal"]))
        left_actions = [
            (turn["action"]["action"], turn["action"].get("price"))
            for turn in left[key].get("episode", {}).get("transcript", [])
            if turn.get("role") == "buyer"
        ]
        right_actions = [
            (turn["action"]["action"], turn["action"].get("price"))
            for turn in right[key].get("episode", {}).get("transcript", [])
            if turn.get("role") == "buyer"
        ]
        action_sequence_flips += left_actions != right_actions
    deltas = {key: statistics.mean(value) for key, value in by_profile.items()}
    agreement = {key: statistics.mean(value) for key, value in agreement_by_profile.items()}
    return {
        "profile_clusters": len(deltas),
        "paired_episodes": len(set(left) & set(right)),
        "mean_trained_reward_delta": statistics.mean(deltas.values()),
        "reward_profile_bootstrap_95_ci": bootstrap_profile_ci(deltas, samples, seed),
        "mean_trained_agreement_delta": statistics.mean(agreement.values()),
        "agreement_profile_bootstrap_95_ci": bootstrap_profile_ci(agreement, samples, seed + 1),
        "action_sequence_flip_rate": action_sequence_flips / max(1, len(set(left) & set(right))),
        "profile_reward_deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=14)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--train-rollouts", type=int, default=5)
    parser.add_argument("--dev-rollouts", type=int, default=12)
    parser.add_argument("--test-rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    args = parser.parse_args()
    if not args.calibration_json.exists():
        raise FileNotFoundError(args.calibration_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    defaults = asdict(BehavioralFrontierPlannerConfig())
    mean = {key: float(defaults[key]) for key in PARAM_BOUNDS}
    std = {key: (high - low) * 0.25 for key, (low, high) in PARAM_BOUNDS.items()}
    history = []
    best = dict(mean)
    best_dev = -math.inf
    train_profiles, dev_profiles = profile_suite("train"), profile_suite("dev")
    for iteration in range(args.iterations):
        candidates = [dict(mean)] + [sample_candidate(mean, std, rng) for _ in range(args.population - 1)]
        scored = []
        for candidate_id, candidate in enumerate(candidates):
            train_reward, _ = evaluate(
                candidate, profiles=train_profiles, rollouts=args.train_rollouts,
                seed=args.seed, calibration_json=args.calibration_json,
            )
            scored.append((train_reward, candidate))
            history.append({"iteration": iteration, "candidate": candidate_id, "train_reward": train_reward, "params": candidate})
        elites = sorted(scored, key=lambda item: item[0], reverse=True)[:args.elite]
        mean = {key: statistics.mean(item[1][key] for item in elites) for key in PARAM_BOUNDS}
        std = {
            key: max((PARAM_BOUNDS[key][1] - PARAM_BOUNDS[key][0]) * 0.025,
                     statistics.pstdev([item[1][key] for item in elites]))
            for key in PARAM_BOUNDS
        }
        dev_reward, _ = evaluate(
            mean, profiles=dev_profiles, rollouts=args.dev_rollouts,
            seed=args.seed + 91, calibration_json=args.calibration_json,
        )
        history.append({"iteration": iteration, "candidate": "elite_mean", "dev_reward": dev_reward, "params": mean})
        if dev_reward > best_dev:
            best_dev, best = dev_reward, dict(mean)
        print(json.dumps({"iteration": iteration, "best_train": elites[0][0], "elite_mean_dev": dev_reward, "best_dev": best_dev}), flush=True)

    current_params = {key: float(defaults[key]) for key in PARAM_BOUNDS}
    test_profiles = profile_suite("test")
    trained_reward, trained_rows = evaluate(
        best, profiles=test_profiles, rollouts=args.test_rollouts,
        seed=args.seed + 191, calibration_json=args.calibration_json, keep_episodes=True,
    )
    current_reward, current_rows = evaluate(
        current_params, profiles=test_profiles, rollouts=args.test_rollouts,
        seed=args.seed + 191, calibration_json=args.calibration_json, keep_episodes=True,
    )
    comparison = summarize_pair(trained_rows, current_rows, args.bootstrap_samples, args.seed)
    comparison["trained_mean_reward"] = trained_reward
    comparison["current_mean_reward"] = current_reward
    # A point on the same piecewise-constant policy surface is not a trained
    # improvement.  In particular, [0, 0] must not pass the deployment gate.
    comparison["planner_gate_passed"] = (
        comparison["action_sequence_flip_rate"] > 0.0
        and comparison["mean_trained_reward_delta"] > 0.0
        and comparison["reward_profile_bootstrap_95_ci"][0] > 0.0
        and comparison["agreement_profile_bootstrap_95_ci"][0] >= -0.05
    )
    artifact = {
        "schema_version": "universal_behavioral_frontier_planner_cem_v1",
        "belief_calibration_json": str(args.calibration_json.resolve()),
        "optimized_params": best,
        "full_config": asdict(config_from_params(best)),
        "training": {
            "method": "cross-entropy method / derivative-free policy search",
            "language_generator_frozen": True,
            "belief_frozen": True,
            "train_profiles": [profile.to_dict() for profile in train_profiles],
            "dev_profiles": [profile.to_dict() for profile in dev_profiles],
        },
        "heldout_test": comparison,
        "history": history,
    }
    (args.output_dir / "trained_planner_config.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "test_episodes.jsonl").write_text(
        "".join(json.dumps({"system": "trained", **row}, ensure_ascii=False) + "\n" for row in trained_rows)
        + "".join(json.dumps({"system": "current", **row}, ensure_ascii=False) + "\n" for row in current_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "comparison": comparison, "best_params": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
