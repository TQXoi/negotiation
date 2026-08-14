#!/usr/bin/env python3
"""Create an auditable V7 confirmatory matrix report from persisted summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METHODS = [
    "framework_v5",
    "framework_v7",
    "framework_v7_frozen",
    "framework_v7_wrong_confident",
    "framework_v7_shuffled",
    "framework_v7_oracle",
    "framework_v7_policy_uniform",
    "framework_v7_policy_shuffled",
]
SETTINGS = ["buyer", "seller", "resource_first", "resource_second"]
SHORT = {
    "framework_v5": "V5",
    "framework_v7": "V7",
    "framework_v7_frozen": "frozen",
    "framework_v7_wrong_confident": "wrong",
    "framework_v7_shuffled": "shuffled",
    "framework_v7_oracle": "oracle",
    "framework_v7_policy_uniform": "policy-uniform",
    "framework_v7_policy_shuffled": "policy-shuffled",
}


def load_summary(root: Path, setting: str, method: str) -> dict[str, Any] | None:
    path = root / setting / method / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("episodes", 0)) != 100:
        return None
    return data


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def paired_run_deltas(reference: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    ref = {int(row["run"]): row for row in reference.get("run_metrics", [])}
    var = {int(row["run"]): row for row in variant.get("run_metrics", [])}
    common = sorted(ref.keys() & var.keys())
    deltas = [
        float(var[key]["mean_focal_reward"]) - float(ref[key]["mean_focal_reward"])
        for key in common
    ]
    if not deltas:
        return {"runs": 0, "deltas": [], "mean": None, "sd": None, "ci95_t": None}
    delta_sd = stdev(deltas) if len(deltas) > 1 else 0.0
    # t(4)=2.776. The confirmatory design preregisters five paired runs.
    half_width = 2.776 * delta_sd / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    delta_mean = mean(deltas)
    return {
        "runs": len(deltas),
        "deltas": deltas,
        "mean": delta_mean,
        "sd": delta_sd,
        "ci95_t": [delta_mean - half_width, delta_mean + half_width],
        "positive_runs": sum(delta > 0 for delta in deltas),
    }


def build(root: Path) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for setting in SETTINGS:
        available = {
            method: summary
            for method in METHODS
            if (summary := load_summary(root, setting, method)) is not None
        }
        summaries[setting] = available
        comparisons[setting] = {}
        if "framework_v7" in available:
            for method, summary in available.items():
                if method == "framework_v7":
                    continue
                comparisons[setting][method] = paired_run_deltas(
                    available["framework_v7"], summary
                )
    return {
        "root": str(root),
        "complete_cells": sum(len(rows) for rows in summaries.values()),
        "total_cells": len(SETTINGS) * len(METHODS),
        "summaries": summaries,
        "paired_reward_delta_vs_v7": comparisons,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NegotiationArena V7 confirmatory matrix",
        "",
        f"Complete cells: {report['complete_cells']}/{report['total_cells']}",
        "",
        "All rewards are focal-agent rewards. `Δ vs V7` is variant minus V7, paired by the five run IDs; the interval is a descriptive t interval over five runs.",
        "",
    ]
    for setting in SETTINGS:
        lines.extend(
            [
                f"## {setting}",
                "",
                "| Method | Reward | Agreement | Joint | Brier | ECE | Pref. MAE | q10–q90 coverage | Errors |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        available = report["summaries"][setting]
        for method in METHODS:
            if method not in available:
                continue
            row = available[method]
            lines.append(
                "| {method} | {reward} | {agreement} | {joint} | {brier} | {ece} | {mae} | {coverage} | {errors} |".format(
                    method=SHORT[method],
                    reward=fmt(row.get("mean_focal_reward")),
                    agreement=fmt(row.get("agreement_rate")),
                    joint=fmt(row.get("mean_joint_reward")),
                    brier=fmt(row.get("belief_accept_brier")),
                    ece=fmt(row.get("belief_accept_ece_5bin")),
                    mae=fmt(row.get("mean_belief_abs_error")),
                    coverage=fmt(row.get("belief_q10_q90_coverage")),
                    errors=fmt(row.get("errors")),
                )
            )
        lines.extend(
            [
                "",
                "| Variant | Δ reward vs V7 | 95% t interval | Positive paired runs | Per-run deltas |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for method in METHODS:
            if method == "framework_v7":
                continue
            comparison = report["paired_reward_delta_vs_v7"][setting].get(method)
            if not comparison:
                continue
            interval = comparison["ci95_t"]
            lines.append(
                f"| {SHORT[method]} | {fmt(comparison['mean'])} | "
                f"[{fmt(interval[0])}, {fmt(interval[1])}] | "
                f"{comparison['positive_runs']}/{comparison['runs']} | "
                f"{', '.join(fmt(value) for value in comparison['deltas'])} |"
            )
        v7 = available.get("framework_v7")
        if v7:
            flips = v7.get("action_flip_rate") or {}
            lines.extend(
                [
                    "",
                    "V7 counterfactual action-flip rates: "
                    + ", ".join(f"{key}={fmt(value)}" for key, value in flips.items())
                    + ".",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    report = build(args.root)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    rendered = markdown(report)
    if args.markdown_out:
        args.markdown_out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
