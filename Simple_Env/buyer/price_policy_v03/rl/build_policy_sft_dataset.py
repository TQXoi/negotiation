#!/usr/bin/env python3
"""Build reward-filtered SFT data for the v0.3 small price policy.

This is an RL-lite/RFT stage: run the policy in the environment, keep actions
from high-reward episodes, and train the small model to imitate those compact
JSON decisions. It is intentionally cheap and can be iterated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", default="small_policy_v03")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "simple" / "small_policy_v03_rft_data"))
    parser.add_argument("--min-reward", type=float, default=0.45)
    parser.add_argument("--keep-deals-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-turns-per-episode", type=int, default=6)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def iter_episodes(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_to_examples(ep: Dict[str, Any], max_turns_per_episode: int) -> List[Dict[str, Any]]:
    examples = []
    reward = float(ep.get("reward") or 0.0)
    scenario = ep.get("scenario") or {}
    for trace in (ep.get("framework_trace") or [])[:max_turns_per_episode]:
        policy = trace.get("small_policy") or {}
        prompt = policy.get("prompt")
        action = policy.get("parsed_action")
        if not prompt or not isinstance(action, dict):
            continue
        # Train the policy on its sanitized action. The verifier revision is
        # recorded as metadata; later DPO/advantage training can use it.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ]
        examples.append(
            {
                "messages": messages,
                "metadata": {
                    "scenario": scenario.get("item_id"),
                    "rollout": ep.get("rollout"),
                    "episode_reward": reward,
                    "status": ep.get("status"),
                    "terminal_reason": ep.get("terminal_reason"),
                    "round": trace.get("round"),
                    "verified_action": (trace.get("verifier") or {}).get("action"),
                    "verifier_approved": (trace.get("verifier") or {}).get("approved"),
                },
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    episode_path = run_dir / f"{args.variant}_episodes.jsonl"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing episode file: {episode_path}")
    examples: List[Dict[str, Any]] = []
    kept_eps = 0
    total_eps = 0
    for ep in iter_episodes(episode_path):
        total_eps += 1
        reward = float(ep.get("reward") or 0.0)
        if args.keep_deals_only and ep.get("status") != "deal":
            continue
        if reward < args.min_reward:
            continue
        kept_eps += 1
        examples.extend(row_to_examples(ep, args.max_turns_per_episode))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_n = max(1, int(round(len(examples) * max(0.0, min(0.5, args.val_frac))))) if len(examples) >= 2 else 0
    val = examples[:val_n]
    train = examples[val_n:]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val or train[:1])
    summary = {
        "run_dir": str(run_dir),
        "variant": args.variant,
        "total_episodes": total_eps,
        "kept_episodes": kept_eps,
        "train_examples": len(train),
        "val_examples": len(val or train[:1]),
        "min_reward": args.min_reward,
        "keep_deals_only": args.keep_deals_only,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
