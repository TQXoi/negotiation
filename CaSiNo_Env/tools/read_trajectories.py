from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def compact_turn(turn: Dict[str, Any]) -> str:
    action = turn.get("action", {})
    trace = turn.get("trace", {})
    lines = [
        f"R{turn.get('round')} {turn.get('actor')}: {action.get('type')}",
        f"  message: {action.get('message')}",
    ]
    if action.get("allocation") is not None:
        lines.append(f"  allocation: {action.get('allocation')}")
    if trace.get("belief"):
        lines.append(f"  belief: {json.dumps(trace['belief'], ensure_ascii=False)}")
    if trace.get("candidates"):
        lines.append(f"  top_candidates: {json.dumps(trace['candidates'][:3], ensure_ascii=False)}")
    if action.get("parse_error"):
        lines.append(f"  parse_error: {action.get('parse_error')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CaSiNo_Env trajectories for inspection.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    paths = [run_dir / f"{args.variant}_episodes.jsonl"] if args.variant else sorted(run_dir.glob("*_episodes.jsonl"))
    printed = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if args.scenario and row.get("scenario") != args.scenario:
                    continue
                print("=" * 88)
                print(
                    f"variant={row.get('buyer')} scenario={row.get('scenario')} rollout={row.get('rollout')} "
                    f"status={row.get('status')} p1={row.get('p1_score')} p2={row.get('p2_score')}"
                )
                ctx = row.get("scenario_context", {})
                print(f"P1 values={ctx.get('p1_values')} P2 values={ctx.get('p2_values')}")
                for turn in row.get("history", []):
                    print(compact_turn(turn))
                printed += 1
                if printed >= args.limit:
                    return


if __name__ == "__main__":
    main()
