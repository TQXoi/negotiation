#!/usr/bin/env python3
"""Analyze V7 core smoke and preference/policy mechanism traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

SETTINGS = ("buyer", "seller", "resource_first", "resource_second")
METHODS = ("framework_v5", "framework_v6", "framework_v7")


def latest(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(int(row["run"]), int(row["episode"]))] = row
    return [rows[key] for key in sorted(rows)]


def run_rewards(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["run"]), []).append(float(row["focal_reward"]))
    return {run: statistics.mean(values) for run, values in grouped.items()}


def trace_mechanism(root, setting):
    paths = list((root / setting / "framework_v7" / "model_traces").glob(
        "run_*/*/framework_v7_decisions.jsonl"
    ))
    rows = {}
    for path in paths:
        run = int(path.parts[-3].split("_")[-1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(run, int(row["episode"]), int(row["decision"]))] = row
    kinds = Counter(str(row["chosen_action"].get("kind")) for row in rows.values())
    belief_rows = [row for row in rows.values() if row.get("posterior", {}).get("response_policy")]
    regret_rows = [
        row["chosen_action"]["belief_usable_planning"]
        for row in rows.values()
        if row.get("chosen_action", {}).get("belief_usable_planning")
    ]
    first = [row for row in rows.values() if int(row["decision"]) == 1]
    flip_modes = (
        "frozen", "wrong_confident", "shuffled", "oracle",
        "policy_uniform", "policy_shuffled",
    )
    return {
        "decisions": len(rows),
        "first_decisions": len(first),
        "chosen_kinds": dict(kinds),
        "first_decision_action_flip_rate": {
            mode: (
                sum(bool(row["action_flip_diagnostics"]["flips"].get(mode)) for row in first)
                / len(first) if first else None
            )
            for mode in flip_modes
        },
        "mean_policy_entropy": (
            statistics.mean(
                float(row["posterior"]["response_policy"]["normalized_entropy"])
                for row in belief_rows
            ) if belief_rows else None
        ),
        "policy_surprise_resets": sum(
            int(row["posterior"]["response_policy"].get("surprise_resets", 0))
            for row in belief_rows
        ),
        "mean_chosen_normalized_regret": (
            statistics.mean(float(row["normalized_particle_regret"]) for row in regret_rows)
            if regret_rows else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cells = {}
    comparisons = []
    for setting in SETTINGS:
        cells[setting] = {}
        for method in METHODS:
            path = args.root / setting / method / "episodes.jsonl"
            rows = latest(path) if path.exists() else []
            valid = [row for row in rows if not row.get("error")]
            summary_path = args.root / setting / method / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            cells[setting][method] = {
                "episodes": len(rows),
                "errors": len(rows) - len(valid),
                "reward": statistics.mean(float(row["focal_reward"]) for row in valid) if valid else None,
                "agreement": statistics.mean(float(row["agreement"]) for row in valid) if valid else None,
                "joint": statistics.mean(float(row["joint_reward"]) for row in valid) if valid else None,
                "brier": summary.get("belief_accept_brier"),
                "ece": summary.get("belief_accept_ece_5bin"),
                "belief_mae": summary.get("mean_belief_abs_error"),
                "action_flip_rate": summary.get("action_flip_rate"),
                "run_rewards": run_rewards(valid),
            }
        cells[setting]["framework_v7"]["mechanism"] = trace_mechanism(args.root, setting)
        v7 = cells[setting]["framework_v7"]["run_rewards"]
        for baseline in ("framework_v5", "framework_v6"):
            old = cells[setting][baseline]["run_rewards"]
            common = sorted(set(v7) & set(old))
            deltas = [v7[run] - old[run] for run in common]
            comparisons.append({
                "setting": setting,
                "contrast": f"v7_minus_{baseline}",
                "run_deltas": deltas,
                "mean_delta": statistics.mean(deltas) if deltas else None,
            })
    complete = all(
        cells[s][m]["episodes"] == 40 and cells[s][m]["errors"] == 0
        for s in SETTINGS for m in METHODS
    )
    payload = {
        "complete": complete,
        "caution": "Two-run smoke is a gate, not confirmatory evidence.",
        "cells": cells,
        "comparisons": comparisons,
    }
    (args.root / "smoke_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"complete": complete, "root": str(args.root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
