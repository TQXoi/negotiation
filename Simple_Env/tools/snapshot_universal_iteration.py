#!/usr/bin/env python3
"""Create an append-only code/config snapshot for one research iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACKED = (
    "negotiation/framework/schemas.py",
    "negotiation/framework/belief.py",
    "negotiation/framework/events.py",
    "negotiation/framework/planner.py",
    "negotiation/framework/engine.py",
    "Simple_Env/buyer/universal_framework.py",
    "Simple_Env/buyer/factory.py",
    "Simple_Env/eval.py",
    "Simple_Env/scripts/run_universal_framework_v1.sh",
    "Simple_Env/scripts/analyze_universal_framework_runs.py",
    "Simple_Env/tools/build_verified_event_dataset.py",
    "Simple_Env/tools/collect_interventional_verified_events.py",
    "Simple_Env/tools/train_verified_belief_calibrator.py",
    "Simple_Env/tools/train_universal_planner_cem.py",
    "Simple_Env/tests/test_analyze_universal_framework_runs.py",
    "Simple_Env/tests/test_verified_events.py",
    "Simple_Env/tests/test_interventional_event_collector.py",
    "negotiation/tests/test_universal_framework.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--result-dir", action="append", default=[])
    args = parser.parse_args()

    destination = args.iteration_dir.resolve() / "code_snapshots" / args.phase
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite immutable snapshot: {destination}")
    destination.mkdir(parents=True)
    files = []
    for relative in TRACKED:
        source = ROOT / relative
        if not source.exists():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": digest})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "hypothesis": args.hypothesis,
        "command": args.command,
        "result_dirs": args.result_dir,
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
