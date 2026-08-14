"""Aggregate completed formal-matrix cells without third-party dependencies."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def mean_ci(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    center = statistics.mean(values)
    if len(values) < 2:
        return {"mean": center, "ci95": None, "n": len(values)}
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": center, "ci95": [center - half, center + half], "n": len(values)}


def main():
    root = Path(sys.argv[1])
    cells = []
    for path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        setting = path.parents[1].name
        method = path.parent.name
        runs = summary.get("run_metrics", [])
        cells.append(
            {
                "setting": setting,
                "method": method,
                "episodes": summary.get("episodes"),
                "valid_episodes": summary.get("valid_episodes"),
                "errors": summary.get("errors"),
                "reward": mean_ci(row.get("mean_focal_reward") for row in runs),
                "agreement": mean_ci(row.get("agreement_rate") for row in runs),
                "pre_switch_reward": mean_ci(row.get("pre_switch_reward") for row in runs),
                "post_switch_reward": mean_ci(row.get("post_switch_reward") for row in runs),
                "brier": summary.get("belief_accept_brier"),
                "nll": summary.get("belief_accept_nll"),
                "ece": summary.get("belief_accept_ece_5bin"),
                "belief_abs_error": summary.get("mean_belief_abs_error"),
                "coverage": summary.get("belief_q10_q90_coverage"),
                "action_flip_rate": summary.get("action_flip_rate"),
                "summary_path": str(path),
            }
        )
    payload = {"root": str(root), "completed_cells": len(cells), "cells": cells}
    (root / "formal_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Formal belief matrix (automatically generated)", "",
        "| Setting | Method | Valid/Episodes | Reward mean [95% CI] | Agreement | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        reward = cell["reward"]
        reward_text = "NA" if reward["mean"] is None else f'{reward["mean"]:.3f}'
        if reward["ci95"]:
            reward_text += f' [{reward["ci95"][0]:.3f}, {reward["ci95"][1]:.3f}]'
        agreement = cell["agreement"]["mean"]
        lines.append(
            f'| {cell["setting"]} | {cell["method"]} | '
            f'{cell["valid_episodes"]}/{cell["episodes"]} | {reward_text} | '
            f'{agreement:.3f} | '
            f'{cell["brier"] if cell["brier"] is not None else "NA"} | '
            f'{cell["ece"] if cell["ece"] is not None else "NA"} |'
        )
    (root / "formal_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completed_cells": len(cells), "root": str(root)}))


if __name__ == "__main__":
    main()
