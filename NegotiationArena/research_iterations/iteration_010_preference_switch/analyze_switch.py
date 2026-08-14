#!/usr/bin/env python3
"""Paired analysis for the preregistered private-preference switch test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

SETTINGS = ["resource_first", "resource_second"]
METHODS = ["framework_v7", "framework_v8", "framework_v8_frozen"]


def latest_rows(path: Path):
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[(int(row["run"]), int(row["episode"]))] = row
    return latest


def cell(root: Path, setting: str, method: str):
    path = root / setting / method / "episodes.jsonl"
    if not path.exists():
        return None
    rows = latest_rows(path)
    if len(rows) != 100 or any(row.get("error") for row in rows.values()):
        return None
    result = {}
    for run in range(1, 6):
        items = [rows[(run, episode)] for episode in range(1, 21)]
        def avg(key, subset):
            return mean(float(row[key]) for row in subset)
        def diag(key, subset):
            vals = [(row.get("focal_diagnostics") or {}).get(key) for row in subset]
            vals = [float(value) for value in vals if value is not None]
            return mean(vals) if vals else None
        result[run] = {
            "pre_reward": avg("focal_reward", items[:10]),
            "post_reward": avg("focal_reward", items[10:]),
            "early_post_reward": avg("focal_reward", items[10:15]),
            "late_post_reward": avg("focal_reward", items[15:]),
            "pre_agreement": avg("agreement", items[:10]),
            "post_agreement": avg("agreement", items[10:]),
            "pre_joint": avg("joint_reward", items[:10]),
            "post_joint": avg("joint_reward", items[10:]),
            "pre_belief_mae": diag("belief_abs_error", items[:10]),
            "post_belief_mae": diag("belief_abs_error", items[10:]),
            "early_post_belief_mae": diag("belief_abs_error", items[10:15]),
            "late_post_belief_mae": diag("belief_abs_error", items[15:]),
            "pre_estimate": diag("belief_estimate", items[:10]),
            "post_estimate": diag("belief_estimate", items[10:]),
        }
    return result


def interval(values):
    if not values:
        return None
    center = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    half = 2.776 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": center, "ci95_t": [center - half, center + half], "values": values}


def build(switch_root: Path, control_root: Path):
    report = {"switch_root": str(switch_root), "control_root": str(control_root), "cells": {}}
    for setting in SETTINGS:
        report["cells"][setting] = {}
        for method in METHODS:
            switched = cell(switch_root, setting, method)
            control = cell(control_root, setting, method)
            if not switched or not control:
                continue
            did = [
                (switched[r]["post_reward"] - switched[r]["pre_reward"])
                - (control[r]["post_reward"] - control[r]["pre_reward"])
                for r in range(1, 6)
            ]
            row = {
                "switch_runs": switched,
                "control_runs": control,
                "switch_pre_reward": mean(switched[r]["pre_reward"] for r in switched),
                "switch_post_reward": mean(switched[r]["post_reward"] for r in switched),
                "control_pre_reward": mean(control[r]["pre_reward"] for r in control),
                "control_post_reward": mean(control[r]["post_reward"] for r in control),
                "reward_did": interval(did),
                "early_post_reward": mean(switched[r]["early_post_reward"] for r in switched),
                "late_post_reward": mean(switched[r]["late_post_reward"] for r in switched),
                "post_agreement": mean(switched[r]["post_agreement"] for r in switched),
                "post_joint": mean(switched[r]["post_joint"] for r in switched),
                "pre_belief_mae": mean(switched[r]["pre_belief_mae"] for r in switched),
                "post_belief_mae": mean(switched[r]["post_belief_mae"] for r in switched),
                "early_post_belief_mae": mean(switched[r]["early_post_belief_mae"] for r in switched),
                "late_post_belief_mae": mean(switched[r]["late_post_belief_mae"] for r in switched),
                "pre_estimate": mean(switched[r]["pre_estimate"] for r in switched),
                "post_estimate": mean(switched[r]["post_estimate"] for r in switched),
            }
            report["cells"][setting][method] = row
    report["complete_cells"] = sum(len(v) for v in report["cells"].values())
    return report


def f(value):
    return "—" if value is None else f"{float(value):.3f}"


def markdown(report):
    lines = [
        "# Private-preference switch analysis", "",
        f"Complete paired cells: {report['complete_cells']}/6", "",
        "Reward DiD = (switch post − switch pre) − (stationary-control late − early), paired by run ID.", "",
    ]
    for setting in SETTINGS:
        lines += [
            f"## {setting}", "",
            "| Method | Switch pre | Switch post | Control pre | Control post | Reward DiD [95% t CI] | Post agr. | Post joint | Pre MAE | Post MAE | Early/Late post reward | Early/Late post MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            row = report["cells"][setting].get(method)
            if not row:
                continue
            did = row["reward_did"]
            lines.append(
                f"| {method} | {f(row['switch_pre_reward'])} | {f(row['switch_post_reward'])} | "
                f"{f(row['control_pre_reward'])} | {f(row['control_post_reward'])} | "
                f"{f(did['mean'])} [{f(did['ci95_t'][0])}, {f(did['ci95_t'][1])}] | "
                f"{f(row['post_agreement'])} | {f(row['post_joint'])} | "
                f"{f(row['pre_belief_mae'])} | {f(row['post_belief_mae'])} | "
                f"{f(row['early_post_reward'])}/{f(row['late_post_reward'])} | "
                f"{f(row['early_post_belief_mae'])}/{f(row['late_post_belief_mae'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("switch_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    report = build(args.switch_root, args.control_root)
    rendered = markdown(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
