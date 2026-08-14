#!/usr/bin/env python3
"""Persist progress and finalize the formal 5x20 experiment matrices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MAIN_METHODS = (
    "direct",
    "opponent_simulation_paper",
    "framework_v3_3",
    "framework_v4",
    "framework_v4_frozen",
    "framework_v4_wrong_confident",
    "framework_v4_shuffled",
    "framework_v4_oracle",
    "framework_v5",
)
SWITCH_METHODS = (
    "opponent_simulation_paper",
    "framework_v4",
    "framework_v4_frozen",
    "framework_v4_wrong_confident",
    "framework_v4_shuffled",
)
SETTINGS = ("buyer", "seller", "resource_first", "resource_second")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-root",
        default="arena_runs/formal_belief_matrix_qwen30b_5x20_20260812",
    )
    parser.add_argument(
        "--switch-root",
        default="arena_runs/opponent_switch_matrix_qwen30b_5x20_20260812",
    )
    parser.add_argument(
        "--state-dir", default="arena_runs/formal_background_20260812"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--episodes-per-run", type=int, default=20)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def latest_rows(path: Path):
    latest = {}
    malformed = 0
    if not path.exists():
        return latest, malformed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            latest[(int(row["run"]), int(row["episode"]))] = row
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed += 1
    return latest, malformed


def matrix_status(root: Path, methods, expected_per_cell: int):
    cells = []
    for setting in SETTINGS:
        for method in methods:
            latest, malformed = latest_rows(root / setting / method / "episodes.jsonl")
            errors = sum(bool(row.get("error")) for row in latest.values())
            valid = sum(not row.get("error") for row in latest.values())
            cells.append(
                {
                    "setting": setting,
                    "method": method,
                    "latest_records": len(latest),
                    "valid": valid,
                    "errors": errors,
                    "malformed_jsonl_lines": malformed,
                    "expected": expected_per_cell,
                    "complete": len(latest) == expected_per_cell and errors == 0,
                }
            )
    return {
        "root": str(root),
        "expected_cells": len(cells),
        "complete_cells": sum(row["complete"] for row in cells),
        "expected_latest_episodes": len(cells) * expected_per_cell,
        "valid_latest_episodes": sum(row["valid"] for row in cells),
        "error_latest_episodes": sum(row["errors"] for row in cells),
        "cells": cells,
    }


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_pids(state_dir: Path):
    path = state_dir / "runner_pids.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def render_progress(payload):
    lines = [
        "# NegotiationArena formal experiment progress",
        "",
        f'- Updated (UTC): {payload["updated_at"]}',
        f'- State: **{payload["state"]}**',
        f'- Model endpoint: {payload["model_endpoint"]}',
        "",
        "| Matrix | Complete cells | Valid latest episodes | Current errors |",
        "|---|---:|---:|---:|",
    ]
    for name in ("main", "opponent_switch"):
        matrix = payload[name]
        lines.append(
            f'| {name} | {matrix["complete_cells"]}/{matrix["expected_cells"]} | '
            f'{matrix["valid_latest_episodes"]}/{matrix["expected_latest_episodes"]} | '
            f'{matrix["error_latest_episodes"]} |'
        )
    incomplete = []
    for name in ("main", "opponent_switch"):
        for row in payload[name]["cells"]:
            if not row["complete"]:
                incomplete.append(
                    f'- {name}/{row["setting"]}/{row["method"]}: '
                    f'{row["valid"]}/{row["expected"]} valid, {row["errors"]} errors'
                )
    if incomplete:
        lines.extend(["", "## Incomplete cells", "", *incomplete])
    return "\n".join(lines) + "\n"


def run_aggregator(repo: Path, root: Path):
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "summarize_formal_belief_matrix.py"),
            str(root),
        ],
        cwd=repo,
        check=True,
    )


def write_completion_report(state_dir: Path, main_root: Path, switch_root: Path):
    lines = [
        "# NegotiationArena 5x20 formal matrices: automatic completion report",
        "",
        "Both matrices passed the completion gate: every expected cell has 100 latest",
        "episode records and zero current errors. Older failed attempts remain in each",
        "`episodes.jsonl` as an append-only audit trail; aggregate files use the latest",
        "record for each `(run, episode)`.",
        "",
        "Generated comparison tables:",
        "",
        f"- `{main_root / 'formal_comparison.md'}`",
        f"- `{switch_root / 'formal_comparison.md'}`",
        f"- `{state_dir / 'NEGOTIATIONARENA_FORMAL_RESULTS_CN.md'}`",
        "",
        "The scientific interpretation step should compare direct and Opponent Simulation",
        "against V4/V5, then use frozen, wrong-confident, shuffled, oracle, calibration,",
        "action-flip, and pre/post-switch metrics to test the belief-causality story.",
    ]
    (state_dir / "AUTO_COMPLETION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (state_dir / "COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def main():
    args = arguments()
    repo = Path(__file__).resolve().parents[1]
    main_root = (repo / args.main_root).resolve()
    switch_root = (repo / args.switch_root).resolve()
    state_dir = (repo / args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    expected = args.runs * args.episodes_per_run

    while True:
        main_matrix = matrix_status(main_root, MAIN_METHODS, expected)
        switch_matrix = matrix_status(switch_root, SWITCH_METHODS, expected)
        complete = (
            main_matrix["complete_cells"] == main_matrix["expected_cells"]
            and switch_matrix["complete_cells"] == switch_matrix["expected_cells"]
        )
        pids = load_pids(state_dir)
        runner_alive = {
            key: pid_alive(value)
            for key, value in pids.items()
            if key in {"main", "opponent_switch"}
        }
        if complete:
            state = "complete"
        elif not runner_alive:
            state = "not_started"
        elif runner_alive and not any(runner_alive.values()):
            state = "stalled"
        else:
            state = "running"
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "model_endpoint": "http://127.0.0.1:8002/v1",
            "runner_alive": runner_alive,
            "main": main_matrix,
            "opponent_switch": switch_matrix,
        }
        (state_dir / "progress.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (state_dir / "progress.md").write_text(render_progress(payload), encoding="utf-8")
        with (state_dir / "monitor.log").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "updated_at": payload["updated_at"],
                        "state": state,
                        "main_cells": main_matrix["complete_cells"],
                        "main_valid": main_matrix["valid_latest_episodes"],
                        "switch_cells": switch_matrix["complete_cells"],
                        "switch_valid": switch_matrix["valid_latest_episodes"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if complete:
            run_aggregator(repo, main_root)
            run_aggregator(repo, switch_root)
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "analyze_formal_results_cn.py"),
                    "--main-root", str(main_root),
                    "--switch-root", str(switch_root),
                    "--output-dir", str(state_dir),
                    "--runs", str(args.runs),
                    "--episodes-per-run", str(args.episodes_per_run),
                ],
                cwd=repo,
                check=True,
            )
            write_completion_report(state_dir, main_root, switch_root)
            return
        if not args.watch or state == "stalled":
            return
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
