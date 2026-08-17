#!/usr/bin/env python3
"""Clean, deduplicate, and split RLVR SFT datasets.

Input files are produced by ``Simple_Env/tools/build_sft_dataset.py`` and use
chat-style JSONL records with a ``messages`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_key(row: Dict[str, Any]) -> str:
    messages = row.get("messages") or []
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_chars(row: Dict[str, Any]) -> int:
    return sum(len(str(msg.get("content") or "")) for msg in row.get("messages") or [])


def valid_messages(row: Dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    if messages[-1].get("role") != "assistant":
        return False
    return all(isinstance(msg.get("content"), str) and msg.get("content").strip() for msg in messages)


def clean_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    min_reward: float,
    max_chars: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seen: set[str] = set()
    kept: List[Dict[str, Any]] = []
    stats = {
        "seen": 0,
        "kept": 0,
        "dropped_invalid": 0,
        "dropped_low_reward": 0,
        "dropped_too_long": 0,
        "dropped_duplicate": 0,
    }
    for row in rows:
        stats["seen"] += 1
        if not valid_messages(row):
            stats["dropped_invalid"] += 1
            continue
        reward = float(row.get("reward") or 0.0)
        if reward < min_reward:
            stats["dropped_low_reward"] += 1
            continue
        if max_chars > 0 and estimate_chars(row) > max_chars:
            stats["dropped_too_long"] += 1
            continue
        key = stable_key(row)
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        cleaned = {
            "messages": row["messages"],
            "metadata": {
                "type": row.get("type"),
                "source_variant": row.get("source_variant"),
                "scenario_id": row.get("scenario_id"),
                "rollout": row.get("rollout"),
                "round": row.get("round"),
                "reward": reward,
            },
        }
        kept.append(cleaned)
    rng = random.Random(seed)
    rng.shuffle(kept)
    stats["kept"] = len(kept)
    return kept, stats


def split_rows(rows: List[Dict[str, Any]], val_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_val = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    return rows[n_val:], rows[:n_val]


def clean_one(
    *,
    input_file: Path,
    output_dir: Path,
    name: str,
    min_reward: float,
    max_chars: int,
    val_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    rows, stats = clean_rows(load_jsonl(input_file), min_reward=min_reward, max_chars=max_chars, seed=seed)
    train, val = split_rows(rows, val_ratio)
    stats["train"] = write_jsonl(output_dir / name / "train.jsonl", train)
    stats["val"] = write_jsonl(output_dir / name / "val.jsonl", val)
    stats["input_file"] = str(input_file)
    stats["output_train"] = str(output_dir / name / "train.jsonl")
    stats["output_val"] = str(output_dir / name / "val.jsonl")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing planner_sft.jsonl and generator_sft.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-min-reward", type=float, default=0.75)
    parser.add_argument("--generator-min-reward", type=float, default=0.65)
    parser.add_argument("--max-chars", type=int, default=60000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    stats = {
        "planner": clean_one(
            input_file=input_dir / "planner_sft.jsonl",
            output_dir=output_dir,
            name="planner",
            min_reward=args.planner_min_reward,
            max_chars=args.max_chars,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
        "generator": clean_one(
            input_file=input_dir / "generator_sft.jsonl",
            output_dir=output_dir,
            name="generator",
            min_reward=args.generator_min_reward,
            max_chars=args.max_chars,
            val_ratio=args.val_ratio,
            seed=args.seed + 1,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
