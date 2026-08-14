#!/usr/bin/env python3
"""One-step V8 shadow actions on fixed V7 confirmatory trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena_integration.decision_calibrated_agent import (  # noqa: E402
    FactorizedPreferencePolicyBeliefPlannerAgent,
    ReliabilityGatedFactorizedBeliefPlannerAgent,
    _clip,
    _normalize,
)


SETTING = {
    "buyer": ("Player BLUE", "buyer_seller", "buyer reward is private WTP 63 minus deal price", "ZUP: 1000", False, 1),
    "seller": ("Player RED", "buyer_seller", "seller reward is deal price minus private production cost 43", "X: 1", False, 0),
    "resource_first": ("Player RED", "resource_exchange", "maximize net resource value with private values X=0.5 and Y=2.5", "X: 25, Y: 5", True, 0),
    "resource_second": ("Player BLUE", "resource_exchange", "maximize net resource value with private values X=2.5 and Y=0.5", "X: 5, Y: 25", False, 1),
}


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


def choose(agent, event, mode, learned_buy, learned_resource, learned_joint):
    agent._set_joint_probs(list(learned_joint))
    if mode == "v8":
        agent._set_decision_belief("continuous", learned_buy, learned_resource)
    elif mode == "wrong_safe":
        agent._set_decision_belief("wrong_confident", learned_buy, learned_resource)
    elif mode == "v7_ungated":
        FactorizedPreferencePolicyBeliefPlannerAgent._set_decision_belief(
            agent, "continuous", learned_buy, learned_resource
        )
    elif mode == "anchor":
        anchor_theta, anchor_policy = agent._marginals_from_joint(
            agent.episode_anchor_joint_probs
        )
        agent._set_joint_probs(agent._factorized_joint(anchor_theta, anchor_policy))
    else:
        raise ValueError(mode)
    candidates = agent._build_candidates()
    action, planner = agent._choose(candidates, event)
    return {
        "signature": agent._action_signature(action),
        "action": action,
        "planner": planner,
        "gate": dict(agent._last_reliability_gate or {}),
    }


def replay_setting(root: Path, setting: str) -> dict[str, Any]:
    source = root / setting / "framework_v7"
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    episodes = latest(source / "episodes.jsonl", ("run", "episode"))
    name, kind, objective, resources, starts, focal = SETTING[setting]
    count = Counter()
    theta_trust = []
    policy_trust = []
    examples = []

    for run in range(1, int(config["runs"]) + 1):
        trace_path = (
            source / "model_traces" / f"run_{run:02d}"
            / name.replace(" ", "_") / "framework_v7_decisions.jsonl"
        )
        traces = latest(trace_path, ("episode", "decision"))
        agent = ReliabilityGatedFactorizedBeliefPlannerAgent(
            agent_name=name, model="offline", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode=str(config["protocol_mode"]),
            game_turn_limit=int(config["max_turns"]),
            candidate_count=int(config["candidate_count"]),
            seed=int(config["seed"]) + run * 1009 + focal,
        )
        for episode in range(1, int(config["episodes_per_run"]) + 1):
            agent.prepare_episode(
                episode_index=episode,
                total_episodes=int(config["episodes_per_run"]),
                game_kind=kind,
                private_objective=objective,
                starts_episode=starts,
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
                if event:
                    agent._record_response_calibration(event)
                    agent._update_from_event(event)
                    apply_semantic(
                        agent, (trace.get("semantic_evidence") or [None])[-1]
                    )
                learned_joint = list(agent._joint_probs())
                learned_buy = list(agent.buy_probs)
                learned_resource = list(agent.resource_probs)

                v8 = choose(
                    agent, event, "v8", learned_buy, learned_resource, learned_joint
                )
                v7 = choose(
                    agent, event, "v7_ungated", learned_buy, learned_resource,
                    learned_joint,
                )
                anchor = choose(
                    agent, event, "anchor", learned_buy, learned_resource,
                    learned_joint,
                )
                wrong = choose(
                    agent, event, "wrong_safe", learned_buy, learned_resource,
                    learned_joint,
                )
                if v8["signature"] != v7["signature"]:
                    count["v8_vs_v7_flips"] += 1
                if v8["signature"] != anchor["signature"]:
                    count["v8_vs_anchor_flips"] += 1
                if wrong["signature"] != v8["signature"]:
                    count["wrong_safe_flips"] += 1
                if v8["signature"] == anchor["signature"]:
                    count["v8_anchor_matches"] += 1
                gate = v8["gate"]
                theta_trust.append(float(gate.get("theta_trust", 0.0)))
                policy_trust.append(float(gate.get("policy_trust", 0.0)))
                if len(examples) < 16 and v8["signature"] != v7["signature"]:
                    examples.append(
                        {
                            "run": run,
                            "episode": episode,
                            "decision": trace["decision"],
                            "v7": v7["signature"],
                            "v8": v8["signature"],
                            "anchor": anchor["signature"],
                            "gate": gate,
                        }
                    )

                # Fixed-trajectory replay: follow the actual V7 public action,
                # never an unexecuted V8 counterfactual.
                agent._set_joint_probs(learned_joint)
                agent._commit_chosen_action(trace["chosen_action"])
                agent.conversation.append(
                    {"role": "assistant", "content": trace.get("locked_response", "")}
                )
                agent.posterior_history.append(trace.get("posterior", {}))

            result = episodes[(run, episode)]
            agent._set_joint_probs(learned_joint if rows else agent._joint_probs())
            agent.record_episode(
                transcript=[],
                own_reward=float(result["focal_reward"]),
                agreement=bool(result["agreement"]),
                deal=result.get("deal"),
            )

    decisions = max(1, count["decisions"])
    return {
        "setting": setting,
        "source": str(source),
        **dict(count),
        "v8_vs_v7_flip_rate": count["v8_vs_v7_flips"] / decisions,
        "v8_vs_anchor_flip_rate": count["v8_vs_anchor_flips"] / decisions,
        "v8_anchor_match_rate": count["v8_anchor_matches"] / decisions,
        "wrong_safe_flip_rate": count["wrong_safe_flips"] / decisions,
        "mean_theta_trust": mean(theta_trust) if theta_trust else None,
        "mean_policy_trust": mean(policy_trust) if policy_trust else None,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "framework_v8_reliability_gated_factorized_belief",
        "model_calls": 0,
        "causal_limit": (
            "One-step actions are computed on fixed V7 trajectories. No "
            "counterfactual realized reward is claimed after an action flip."
        ),
        "settings": [],
    }
    for setting in SETTING:
        row = replay_setting(args.confirmatory_root, setting)
        report["settings"].append(row)
        print(json.dumps({
            key: row.get(key) for key in (
                "setting", "decisions", "v8_vs_v7_flip_rate",
                "v8_vs_anchor_flip_rate", "mean_theta_trust",
                "mean_policy_trust",
            )
        }, ensure_ascii=False), flush=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
