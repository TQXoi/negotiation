#!/usr/bin/env python3
"""Summarize the preregistered V4/V5/V6 smoke and its mechanism traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SETTINGS = ("buyer", "seller", "resource_first", "resource_second")
METHODS = ("framework_v4", "framework_v5", "framework_v6")


def latest_rows(path: Path) -> list[dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(int(row["run"]), int(row["episode"]))] = row
    return [latest[key] for key in sorted(latest)]


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows) if rows else None


def run_rewards(rows):
    by_run = {}
    for row in rows:
        by_run.setdefault(int(row["run"]), []).append(float(row["focal_reward"]))
    return {run: statistics.mean(values) for run, values in by_run.items()}


def mechanism(root: Path, setting: str) -> dict:
    traces = list((root / setting / "framework_v6" / "model_traces").glob(
        "run_*/*/framework_v6_decisions.jsonl"
    ))
    latest = {}
    for path in traces:
        run = int(path.parts[-3].split("_")[-1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[(run, int(row["episode"]), int(row["decision"]))] = row
    evaluated = []
    justified = []
    blocked = []
    for row in latest.values():
        planner = row.get("planner") or {}
        if planner.get("decision_relevant_voi_evaluated"):
            evaluated.append(planner)
        if planner.get("reason") == "decision_relevant_voi_justifies_probe":
            justified.append(planner)
        if planner.get("probe_blocked_by_bounded_opportunity_cost"):
            blocked.append(planner)
    return {
        "decisions": len(latest),
        "voi_evaluations": len(evaluated),
        "justified_probes": len(justified),
        "blocked_probes": len(blocked),
        "mean_action_flip_probability_when_evaluated": (
            statistics.mean(
                float(row["decision_relevant_voi"]["action_flip_probability"])
                for row in evaluated
            )
            if evaluated else None
        ),
        "mean_probe_information_value_when_evaluated": (
            statistics.mean(float(row["probe_information_value"]) for row in evaluated)
            if evaluated else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cells = {}
    comparisons = []
    for setting in SETTINGS:
        cells[setting] = {}
        for method in METHODS:
            path = args.root / setting / method / "episodes.jsonl"
            rows = latest_rows(path) if path.exists() else []
            valid = [row for row in rows if not row.get("error")]
            cells[setting][method] = {
                "episodes": len(rows),
                "errors": len(rows) - len(valid),
                "mean_focal_reward": mean(valid, "focal_reward"),
                "agreement_rate": mean(valid, "agreement"),
                "mean_joint_reward": mean(valid, "joint_reward"),
                "run_rewards": run_rewards(valid),
            }
        cells[setting]["framework_v6"]["mechanism"] = mechanism(args.root, setting)
        v6 = cells[setting]["framework_v6"]["run_rewards"]
        for baseline in ("framework_v4", "framework_v5"):
            other = cells[setting][baseline]["run_rewards"]
            common = sorted(set(v6) & set(other))
            deltas = [v6[run] - other[run] for run in common]
            comparisons.append(
                {
                    "setting": setting,
                    "contrast": f"framework_v6_minus_{baseline}",
                    "paired_runs": len(common),
                    "mean_delta": statistics.mean(deltas) if deltas else None,
                    "run_deltas": deltas,
                }
            )
    complete = all(
        cells[s][m]["episodes"] == 40 and cells[s][m]["errors"] == 0
        for s in SETTINGS for m in METHODS
    )
    payload = {
        "complete": complete,
        "caution": "Two-run smoke is a mechanism/engineering gate, not confirmatory evidence.",
        "cells": cells,
        "paired_comparisons": comparisons,
    }
    (args.root / "smoke_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V6 preregistered smoke", "",
        f"- Complete: **{complete}**", "",
        "| Setting | Method | Episodes/errors | Reward | Agreement | Joint |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for method in METHODS:
            row = cells[setting][method]
            lines.append(
                f"| {setting} | {method} | {row['episodes']}/{row['errors']} | "
                f"{row['mean_focal_reward'] if row['mean_focal_reward'] is not None else 'NA'} | "
                f"{row['agreement_rate'] if row['agreement_rate'] is not None else 'NA'} | "
                f"{row['mean_joint_reward'] if row['mean_joint_reward'] is not None else 'NA'} |"
            )
    lines.extend(["", "## Paired run deltas", ""])
    for row in comparisons:
        lines.append(
            f"- {row['setting']} {row['contrast']}: {row['mean_delta']} "
            f"({row['run_deltas']})"
        )
    lines.extend(["", "## V6 mechanism", ""])
    for setting in SETTINGS:
        lines.append(
            f"- {setting}: "
            + json.dumps(cells[setting]["framework_v6"]["mechanism"], ensure_ascii=False)
        )
    lines.extend([
        "", "> This 2-run smoke is not used for a paper-level significance claim. "
        "Its purpose is to reject broken or mechanism-inactive variants before 5×20 confirmation.",
    ])
    (args.root / "SMOKE_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "output": str(args.root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
