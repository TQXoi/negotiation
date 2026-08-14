"""Aggregate direct/Opponent-Simulation/framework repeated-game results."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True)
    parser.add_argument("--opponent-simulation", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def read_run(path):
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    latest = {}
    for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["run"], row["episode"])] = row
    return config, latest


def bootstrap_ci(values, samples, seed):
    if not values:
        return [None, None]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def metrics(rows, expected_episodes):
    valid = [row for row in rows.values() if not row.get("error")]
    total = len(rows)
    reward_sum = sum(float(row.get("focal_reward", 0.0)) for row in valid)
    agreement_sum = sum(float(row.get("agreement", False)) for row in valid)
    return {
        "episodes_expected": expected_episodes,
        "episodes_recorded": total,
        "completion_rate": total / expected_episodes if expected_episodes else 0.0,
        "valid_episodes": len(valid),
        "errors": total - len(valid),
        "format_success_rate": len(valid) / total if total else 0.0,
        # Errors receive zero agreement and zero reward.  These are the primary
        # comparison metrics; conditional-valid metrics are diagnostics only.
        "all_episode_agreement_rate": agreement_sum / total if total else 0.0,
        "all_episode_mean_focal_reward": reward_sum / total if total else None,
        "valid_only_agreement_rate": mean(float(row["agreement"]) for row in valid) if valid else 0.0,
        "valid_only_mean_focal_reward": mean(row["focal_reward"] for row in valid) if valid else None,
        "valid_only_mean_joint_reward": mean(row["joint_reward"] for row in valid) if valid else None,
        "valid_only_mean_turns": mean(row["turns"] for row in valid) if valid else None,
        "valid_only_mean_focal_calls": mean(row["focal_model_calls"] for row in valid) if valid else None,
    }


def paired_run_deltas(baseline, method):
    grouped = {}
    for key in sorted(set(baseline) & set(method)):
        left, right = baseline[key], method[key]
        # A protocol/API failure is part of deployable policy value and receives
        # zero.  Skipping failures caused the old formal directory to look much
        # stronger despite only 3.5--62% of episodes being valid.
        left_reward = 0.0 if left.get("error") else float(left.get("focal_reward", 0.0))
        right_reward = 0.0 if right.get("error") else float(right.get("focal_reward", 0.0))
        grouped.setdefault(key[0], []).append(right_reward - left_reward)
    return [mean(values) for _, values in sorted(grouped.items())]


def paired_coverage(baseline, method, expected_episodes):
    common = set(baseline) & set(method)
    return {
        "paired_episodes": len(common),
        "paired_coverage_rate": len(common) / expected_episodes if expected_episodes else 0.0,
    }


def main():
    args = parse_args()
    configs = {}
    rows = {}
    for label, path in {
        "direct": args.direct,
        "opponent_simulation": args.opponent_simulation,
        "framework": args.framework,
    }.items():
        configs[label], rows[label] = read_run(path)
    identity_fields = [
        "setting", "model", "runs", "episodes_per_run", "max_turns",
        "candidate_count", "seed", "protocol_mode", "opponent_policy",
        "resolved_opponent_model", "resolved_opponent_base_url",
    ]
    mismatches = {
        field: {label: config.get(field) for label, config in configs.items()}
        for field in identity_fields
        if len({config.get(field) for config in configs.values()}) != 1
    }
    if mismatches:
        raise ValueError(f"Comparison is not protocol matched: {mismatches}")
    expected_episodes = int(configs["direct"]["runs"]) * int(
        configs["direct"]["episodes_per_run"]
    )
    report = {
        "protocol": {field: configs["direct"].get(field) for field in identity_fields},
        "methods": {
            label: {
                "configured_method": configs[label].get("method"),
                **metrics(value, expected_episodes),
            }
            for label, value in rows.items()
        },
        "paired_improvement_over_direct": {},
        "paper_qwen3_reference_delta": {
            "buyer": "+10.04 ± 2.03",
            "seller": "+18.54 ± 2.46",
            "resource_first": "+8.95 ± 6.23",
            "resource_second": "+29.65 ± 0.33",
        },
        "interpretation_warning": (
            "The paper reference is an improvement over its own Gemini-opponent direct baseline. "
            "Only compare numerically after matching acting/opponent model snapshots and prompts. "
            "Primary local metrics include failed episodes as zero; valid-only metrics are diagnostic."
        ),
    }
    for label in ("opponent_simulation", "framework"):
        deltas = paired_run_deltas(rows["direct"], rows[label])
        report["paired_improvement_over_direct"][label] = {
            "mean_run_level_delta": mean(deltas) if deltas else None,
            "bootstrap_95_ci": bootstrap_ci(deltas, args.bootstrap_samples, args.seed),
            "run_level_deltas": deltas,
            **paired_coverage(rows["direct"], rows[label], expected_episodes),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
