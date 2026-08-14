#!/usr/bin/env python3
"""One-step counterfactual replay of V6 on completed V5 trajectories.

This script never calls a language model.  It reconstructs the V5 particle
posterior from recorded structured events and recorded semantic-extractor
outputs, verifies that reconstruction against each saved posterior summary,
then asks V6 what it would do at the same decision.  Because future opponent
responses would change after an action flip, the output is explicitly a
one-step/off-policy diagnostic rather than a counterfactual reward estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena_integration.decision_calibrated_agent import (
    DecisionRelevantInformationPlannerAgent,
    _clip,
    _normalize,
)


SETTING = {
    "buyer": {
        "agent_name": "Player BLUE",
        "game_kind": "buyer_seller",
        "objective": "buyer reward is private WTP 63 minus deal price",
        "resources": "ZUP: 1000",
        "starts": False,
        "focal_index": 1,
    },
    "seller": {
        "agent_name": "Player RED",
        "game_kind": "buyer_seller",
        "objective": "seller reward is deal price minus private production cost 43",
        "resources": "X: 1",
        "starts": False,
        "focal_index": 0,
    },
    "resource_first": {
        "agent_name": "Player RED",
        "game_kind": "resource_exchange",
        "objective": "maximize net resource value with private values X=0.5 and Y=2.5",
        "resources": "X: 25, Y: 5",
        "starts": True,
        "focal_index": 0,
    },
    "resource_second": {
        "agent_name": "Player BLUE",
        "game_kind": "resource_exchange",
        "objective": "maximize net resource value with private values X=2.5 and Y=0.5",
        "resources": "X: 5, Y: 25",
        "starts": False,
        "focal_index": 1,
    },
}


def latest_jsonl(path: Path, keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict]:
    rows: dict[tuple[Any, ...], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[tuple(row[key] for key in keys)] = row
    return rows


def apply_recorded_semantic(agent, item: dict[str, Any] | None) -> None:
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
    claim = item.get("explicit_reservation_claim")
    if claim is not None and agent.game_kind == "buyer_seller":
        claim = float(claim)
        likelihood = [
            (1.0 - reliability)
            + reliability * math.exp(-0.5 * ((reservation - claim) / 8.0) ** 2)
            for reservation in agent.buy_values
        ]
        agent.buy_probs = _normalize(
            [prob * weight for prob, weight in zip(agent.buy_probs, likelihood)]
        )
    agent.semantic_evidence.append(dict(item))
    agent.evidence.append(
        {
            "event": "PUBLIC_LANGUAGE",
            "source": "recorded_semantic_extractor_replay",
            "reliability": reliability,
            "claim": claim,
        }
    )


def summary_error(agent, expected: dict[str, Any]) -> float:
    actual = agent._posterior_json()
    fields = (
        ("mean", "q10", "q90", "normalized_entropy")
        if agent.game_kind == "buyer_seller"
        else ("x_weight_mean", "x_weight_q10", "x_weight_q90", "normalized_entropy")
    )
    return max(abs(float(actual[name]) - float(expected[name])) for name in fields)


def replay_cell(root: Path, setting: str, tolerance: float) -> dict[str, Any]:
    source = root / setting / "framework_v5"
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    episodes = latest_jsonl(source / "episodes.jsonl", ("run", "episode"))
    spec = SETTING[setting]
    counters: Counter[str] = Counter()
    errors: list[float] = []
    examples: list[dict[str, Any]] = []

    for run in range(1, int(config["runs"]) + 1):
        trace_path = (
            source
            / "model_traces"
            / f"run_{run:02d}"
            / spec["agent_name"].replace(" ", "_")
            / "framework_v5_decisions.jsonl"
        )
        traces = latest_jsonl(trace_path, ("episode", "decision"))
        agent = DecisionRelevantInformationPlannerAgent(
            agent_name=spec["agent_name"],
            model="offline-replay",
            base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY",
            temperature=float(config["temperature"]),
            max_tokens=int(config["max_tokens"]),
            seed=int(config["seed"]) + run * 1009 + int(spec["focal_index"]),
            trace_dir=None,
            history_window=int(config["history_window"]),
            context_char_limit=int(config["context_char_limit"]),
            protocol_mode=str(config["protocol_mode"]),
            protocol_repair_attempts=int(config["protocol_repair_attempts"]),
            game_turn_limit=int(config["max_turns"]),
            candidate_count=int(config["candidate_count"]),
            belief_mode="continuous",
            language_realizer=False,
            semantic_belief=False,
        )
        for episode in range(1, int(config["episodes_per_run"]) + 1):
            agent.prepare_episode(
                episode_index=episode,
                total_episodes=int(config["episodes_per_run"]),
                game_kind=spec["game_kind"],
                private_objective=spec["objective"],
                starts_episode=spec["starts"],
            )
            agent.init_agent(
                f"<my resources> {spec['resources']} </my resources>"
                f"<my goals> {spec['objective']} </my goals>",
                f"You are {spec['agent_name']}.",
            )
            episode_traces = [
                row for (ep, _), row in traces.items() if int(ep) == episode
            ]
            episode_traces.sort(key=lambda row: int(row["decision"]))
            for trace in episode_traces:
                counters["decisions"] += 1
                agent.decision_index = int(trace["decision"])
                event = trace.get("event")
                if event:
                    agent._record_response_calibration(event)
                    agent._update_from_event(event)
                    semantic = (trace.get("semantic_evidence") or [None])[-1]
                    apply_recorded_semantic(agent, semantic)
                error = summary_error(agent, trace["learned_posterior"])
                errors.append(error)
                validated = error <= tolerance
                counters["posterior_validated" if validated else "posterior_mismatch"] += 1

                candidates = agent._build_candidates()
                v6_action, v6_planner = agent._choose(candidates, event)
                old_action = trace["chosen_action"]
                old_signature = agent._action_signature(old_action)
                v6_signature = agent._action_signature(v6_action)
                if old_signature != v6_signature:
                    counters["v6_action_flips"] += 1
                old_probe = bool(
                    old_action.get("kind") == "probe"
                    and trace.get("planner", {}).get("reason")
                    == "cross_episode_information_value_justifies_probe"
                )
                if old_probe:
                    counters["v5_cross_episode_probes"] += 1
                if old_probe and v6_action.get("type") == "ACCEPT":
                    counters["v6_blocked_old_probes"] += 1
                    if float(v6_action.get("self_utility", 0.0)) > 0:
                        counters["v6_restored_positive_accepts"] += 1
                if v6_action.get("type") == "PROPOSE":
                    counters["v6_proposals"] += 1
                    if float(v6_action.get("self_utility", 0.0)) < 0:
                        counters["v6_infeasible_proposals"] += 1
                if v6_planner.get("decision_relevant_voi_evaluated"):
                    counters["v6_voi_evaluations"] += 1
                if v6_planner.get("reason") == "decision_relevant_voi_justifies_probe":
                    counters["v6_justified_probes"] += 1
                if validated and old_probe and len(examples) < 12:
                    examples.append(
                        {
                            "run": run,
                            "episode": episode,
                            "decision": trace["decision"],
                            "old": old_signature,
                            "v6": v6_signature,
                            "accept_value": trace.get("planner", {}).get(
                                "accept_observed_value"
                            ),
                            "v4_information_value": trace.get("planner", {}).get(
                                "cross_episode_information_value"
                            ),
                            "v6_information_value": v6_planner.get(
                                "probe_information_value"
                            ),
                            "v6_voi": v6_planner.get("decision_relevant_voi"),
                        }
                    )

                # Continue on the logged trajectory, not on the counterfactual V6 action.
                agent._commit_chosen_action(old_action)
                agent.conversation.append(
                    {"role": "assistant", "content": trace.get("locked_response", "")}
                )
                agent.posterior_history.append(trace["learned_posterior"])

            row = episodes[(run, episode)]
            agent.record_episode(
                transcript=[],
                own_reward=float(row["focal_reward"]),
                agreement=bool(row["agreement"]),
                deal=row.get("deal"),
            )

    return {
        "setting": setting,
        "source": str(source),
        "replay_type": "one_step_off_policy_fixed_v5_trajectory",
        "posterior_tolerance": tolerance,
        "max_posterior_summary_error": max(errors, default=None),
        "mean_posterior_summary_error": sum(errors) / len(errors) if errors else None,
        **dict(counters),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("arena_runs/formal_belief_matrix_qwen30b_5x20_20260812"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posterior-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    report = {
        "method": "framework_v6_decision_relevant_voi",
        "model_calls": 0,
        "causal_limit": (
            "Actions are one-step counterfactuals on fixed V5 trajectories; reward after an "
            "action flip is not identifiable without an online rollout."
        ),
        "settings": [
            replay_cell(args.formal_root, setting, args.posterior_tolerance)
            for setting in SETTING
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "settings": [
            {
                "setting": row["setting"],
                "decisions": row.get("decisions", 0),
                "posterior_mismatch": row.get("posterior_mismatch", 0),
                "old_probes": row.get("v5_cross_episode_probes", 0),
                "blocked": row.get("v6_blocked_old_probes", 0),
                "v6_probes": row.get("v6_justified_probes", 0),
            }
            for row in report["settings"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
