#!/usr/bin/env python3
"""Write resumable progress for the V7 5×20 confirmatory matrix."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

METHODS = (
    "framework_v5", "framework_v7", "framework_v7_frozen",
    "framework_v7_wrong_confident", "framework_v7_shuffled",
    "framework_v7_oracle", "framework_v7_policy_uniform",
    "framework_v7_policy_shuffled",
)
SETTINGS = ("buyer", "seller", "resource_first", "resource_second")


def inspect(root: Path, target: int):
    cells = []
    for setting in SETTINGS:
        for method in METHODS:
            path = root / setting / method / "episodes.jsonl"
            latest = {}
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        latest[(int(row["run"]), int(row["episode"]))] = row
            errors = sum(bool(row.get("error")) for row in latest.values())
            cells.append({
                "setting": setting, "method": method,
                "latest": len(latest), "target": target, "errors": errors,
                "complete": len(latest) == target and errors == 0,
            })
    return cells


def write(root: Path, target: int):
    cells = inspect(root, target)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "complete_cells": sum(row["complete"] for row in cells),
        "total_cells": len(cells),
        "latest_episodes": sum(row["latest"] for row in cells),
        "target_episodes": target * len(cells),
        "current_errors": sum(row["errors"] for row in cells),
        "cells": cells,
    }
    (root / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V7 confirmatory progress", "",
        f"- Updated UTC: {payload['updated_utc']}",
        f"- Cells: **{payload['complete_cells']}/{payload['total_cells']}**",
        f"- Latest episodes: **{payload['latest_episodes']}/{payload['target_episodes']}**",
        f"- Current errors: **{payload['current_errors']}**", "",
        "| Setting | Method | Latest/target | Errors |",
        "|---|---|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['setting']} | {row['method']} | "
            f"{row['latest']}/{row['target']} | {row['errors']} |"
        )
    (root / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    while True:
        payload = write(args.root, args.target)
        print(json.dumps({key: payload[key] for key in (
            "complete_cells", "total_cells", "latest_episodes",
            "target_episodes", "current_errors",
        )}), flush=True)
        if not args.watch or payload["complete_cells"] == payload["total_cells"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
