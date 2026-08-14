#!/usr/bin/env python3
"""One-step V7 replay on frozen V5 trajectories without model calls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena_integration.decision_calibrated_agent import (  # noqa: E402
    FactorizedPreferencePolicyBeliefPlannerAgent,
    _clip,
    _normalize,
)


SETTING = {
    "buyer": ("Player BLUE", "buyer_seller", "buyer reward is private WTP 63 minus deal price", "ZUP: 1000", False, 1),
    "seller": ("Player RED", "buyer_seller", "seller reward is deal price minus private production cost 43", "X: 1", False, 0),
    "resource_first": ("Player RED", "resource_exchange", "maximize net resource value with private values X=0.5 and Y=2.5", "X: 25, Y: 5", True, 0),
    "resource_second": ("Player BLUE", "resource_exchange", "maximize net resource value with private values X=2.5 and Y=0.5", "X: 5, Y: 25", False, 1),
}
MODES = (
    "frozen", "wrong_confident", "shuffled", "oracle",
    "policy_uniform", "policy_shuffled",
)


def latest(path: Path, keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[tuple(row[key] for key in keys)] = row
    return rows


def apply_semantic(agent, item: dict[str, Any] | None) -> None:
    if not item or item.get("parse_error"):
        return
    reliability = float(item.get("bounded_reliability", 0.0))
    concession = _clip(float(item.get("concession_signal", 0.0)), -1.0, 1.0)
    firmness = _clip(float(item.get("firmness", 0.5)), 0.0, 1.0)
    agent.semantic_acceptance_shift = _clip(
        0.65 * agent.semantic_acceptance_shift
        + reliability * (concession - (firmness - 0.5)),
        -0.35,
        0.35,
    )
    agent._particle_response_cache = {}
    claim = item.get("explicit_reservation_claim")
    if claim is not None and agent.game_kind == "buyer_seller":
        claim = float(claim)
        policy_size = len(agent.policy_particles)
        likelihood = [
            (1.0 - reliability)
            + reliability * math.exp(-0.5 * ((theta - claim) / 8.0) ** 2)
            for theta in agent.buy_values
        ]
        agent._set_joint_probs(
            _normalize(
                [
                    probability * likelihood[index // policy_size]
                    for index, probability in enumerate(agent._joint_probs())
                ]
            )
        )
    agent.semantic_evidence.append(dict(item))


def signature(agent, mode, learned_theta, learned_resource, joint, event):
    agent._set_joint_probs(list(joint))
    agent._set_decision_belief(mode, learned_theta, learned_resource)
    candidates = agent._build_candidates()
    action, planner = agent._choose(candidates, event)
    result = {
        "signature": agent._action_signature(action),
        "posterior": agent._posterior_json(),
        "planner": planner,
        "action": action,
    }
    agent._set_joint_probs(list(joint))
    return result


def replay_setting(root: Path, setting: str) -> dict[str, Any]:
    source = root / setting / "framework_v5"
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    episodes = latest(source / "episodes.jsonl", ("run", "episode"))
    name, kind, objective, resources, starts, focal = SETTING[setting]
    count = Counter()
    examples = []
    preference_path = []
    policy_path = []

    for run in range(1, int(config["runs"]) + 1):
        trace_path = source / "model_traces" / f"run_{run:02d}" / name.replace(" ", "_") / "framework_v5_decisions.jsonl"
        traces = latest(trace_path, ("episode", "decision"))
        agent = FactorizedPreferencePolicyBeliefPlannerAgent(
            agent_name=name, model="offline", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode=str(config["protocol_mode"]),
            game_turn_limit=int(config["max_turns"]),
            candidate_count=int(config["candidate_count"]),
            seed=int(config["seed"]) + run * 1009 + focal,
        )
        for episode in range(1, int(config["episodes_per_run"]) + 1):
            agent.prepare_episode(
                episode_index=episode, total_episodes=int(config["episodes_per_run"]),
                game_kind=kind, private_objective=objective, starts_episode=starts,
            )
            agent.init_agent(
                f"<my resources> {resources} </my resources><my goals> {objective} </my goals>",
                f"You are {name}.",
            )
            rows = [row for (ep, _), row in traces.items() if int(ep) == episode]
            rows.sort(key=lambda row: int(row["decision"]))
            for trace in rows:
                count["decisions"] += 1
                agent.decision_index = int(trace["decision"])
                event = trace.get("event")
                pre_joint = list(agent._joint_probs())
                if event:
                    agent._record_response_calibration(event)
                    agent._update_from_event(event)
                    apply_semantic(agent, (trace.get("semantic_evidence") or [None])[-1])
                learned_joint = list(agent._joint_probs())
                learned_theta = list(agent.buy_probs)
                learned_resource = list(agent.resource_probs)
                agent._set_joint_probs(pre_joint)
                pre_theta = list(agent.buy_probs)
                pre_resource = list(agent.resource_probs)
                agent._set_joint_probs(learned_joint)

                continuous = signature(
                    agent, "continuous", learned_theta, learned_resource,
                    learned_joint, event,
                )
                old_signature = agent._action_signature(trace["chosen_action"])
                if continuous["signature"] != old_signature:
                    count["v7_vs_v5_action_flips"] += 1
                if continuous["action"].get("type") == "PROPOSE":
                    count["v7_proposals"] += 1
                    if float(continuous["action"].get("self_utility", 0.0)) < 0:
                        count["v7_infeasible_proposals"] += 1

                interventions = {}
                # Full interventions are evaluated once per episode. Continuous
                # V7 is still replayed at every logged decision. This preserves
                # all roles/episodes while avoiding seven repeated combinatorial
                # frontier aggregations at every within-episode counteroffer.
                if int(trace["decision"]) == 1:
                    count["intervention_decisions"] += 1
                    for mode in MODES:
                        mode_joint = pre_joint if mode == "frozen" else learned_joint
                        result = signature(
                            agent, "continuous" if mode == "frozen" else mode,
                            pre_theta if mode == "frozen" else learned_theta,
                            pre_resource if mode == "frozen" else learned_resource,
                            mode_joint, event,
                        )
                        interventions[mode] = result["signature"]
                        if result["signature"] != continuous["signature"]:
                            count[f"flip_{mode}"] += 1

                posterior = continuous["posterior"]
                preference_path.append(
                    float(posterior.get("mean", posterior.get("x_weight_mean")))
                )
                policy_path.append(dict(posterior["response_policy"]))
                if len(examples) < 20 and any(
                    value != continuous["signature"] for value in interventions.values()
                ):
                    examples.append(
                        {
                            "run": run, "episode": episode,
                            "decision": trace["decision"],
                            "v5": old_signature,
                            "v7": continuous["signature"],
                            "interventions": interventions,
                            "posterior": posterior,
                            "chosen_belief_use": continuous["action"].get(
                                "belief_usable_planning"
                            ),
                        }
                    )

                agent._set_joint_probs(learned_joint)
                agent._commit_chosen_action(trace["chosen_action"])
                agent.conversation.append(
                    {"role": "assistant", "content": trace.get("locked_response", "")}
                )
                agent.posterior_history.append(posterior)

            result = episodes[(run, episode)]
            agent.record_episode(
                transcript=[], own_reward=float(result["focal_reward"]),
                agreement=bool(result["agreement"]), deal=result.get("deal"),
            )

    return {
        "setting": setting,
        "source": str(source),
        **dict(count),
        "action_flip_rates": {
            mode: count[f"flip_{mode}"] / max(1, count["intervention_decisions"])
            for mode in MODES
        },
        "final_preference_estimate": preference_path[-1] if preference_path else None,
        "final_response_policy": policy_path[-1] if policy_path else None,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root", type=Path,
        default=Path("arena_runs/formal_belief_matrix_qwen30b_5x20_20260812"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "framework_v7_factorized_preference_policy",
        "model_calls": 0,
        "source_posterior_validation": (
            "Iteration 007 exact replay validated all 553 V5 decision posterior summaries "
            "with zero mismatch before this V7 shadow replay."
        ),
        "causal_limit": (
            "One-step actions are evaluated on fixed V5 trajectories; no counterfactual "
            "reward is claimed after an action flip."
        ),
        "settings": [],
    }
    for setting in SETTING:
        row = replay_setting(args.formal_root, setting)
        report["settings"].append(row)
        print(json.dumps({
            "setting_complete": setting,
            "decisions": row.get("decisions"),
            "intervention_decisions": row.get("intervention_decisions"),
        }, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "settings": [
            {key: row.get(key) for key in (
                "setting", "decisions", "v7_vs_v5_action_flips",
                "v7_proposals", "v7_infeasible_proposals",
            )}
            for row in report["settings"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
