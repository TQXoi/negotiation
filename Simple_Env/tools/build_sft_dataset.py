#!/usr/bin/env python3
"""Build SFT data for RLVR planner and natural-language generator.

The input is one or more Simple_Env run directories. The tool filters completed
episodes, then emits two chat-format JSONL files:

- planner_sft.jsonl: belief + history -> strategic plan JSON
- generator_sft.jsonl: belief + plan + history -> Thought/Talk/Action message

This is deliberately conservative: by default it keeps only deal trajectories
with positive reward, no overshoot, and no format violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def episode_files(run_dirs: List[Path], variants: Optional[set[str]]) -> Iterable[Path]:
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*_episodes.jsonl")):
            variant = path.name.removesuffix("_episodes.jsonl")
            if variants and variant not in variants:
                continue
            yield path


def compact_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    action = turn.get("action") or {}
    return {
        "round": turn.get("round"),
        "role": turn.get("role"),
        "message": turn.get("message"),
        "action": {
            "type": action.get("action"),
            "price": action.get("price"),
            "valid": action.get("valid"),
            "item_spec": action.get("item_spec"),
        },
    }


def buyer_turn_indices(transcript: List[Dict[str, Any]]) -> List[int]:
    return [idx for idx, turn in enumerate(transcript) if turn.get("role") == "buyer"]


def accept_episode(
    row: Dict[str, Any],
    *,
    min_reward: float,
    require_deal: bool,
    max_first_turn_offer_ratio: Optional[float],
) -> bool:
    if require_deal and row.get("status") != "deal":
        return False
    if float(row.get("reward") or 0.0) < min_reward:
        return False
    if row.get("buyer_overshoot") or row.get("buyer_format_violation"):
        return False
    if max_first_turn_offer_ratio is not None:
        first = row.get("first_turn_offer_ratio")
        if first is not None and float(first) > max_first_turn_offer_ratio:
            return False
    return True


def make_planner_example(
    row: Dict[str, Any],
    trace: Dict[str, Any],
    history_before: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scenario = row.get("scenario") or {}
    user = {
        "task": "RLVR buyer strategic planner SFT",
        "instruction": (
            "Given the buyer's private scenario, dialogue history, and opponent belief, "
            "produce a buyer-side strategic plan. The plan should preserve buyer surplus, "
            "respect the hard budget ceiling, avoid premature quit, and choose a concrete "
            "target price or accept decision."
        ),
        "scenario": {
            "item_id": scenario.get("item_id"),
            "title": scenario.get("title"),
            "buyer_budget": scenario.get("buyer_budget"),
            "reference_price": scenario.get("reference_price"),
            "codename": scenario.get("codename"),
            "quantity": scenario.get("quantity"),
        },
        "history_before": history_before,
        "belief": trace.get("belief"),
        "concession_schedule": extract_schedule(trace),
    }
    assistant = trace.get("plan") or {}
    return {
        "type": "planner",
        "source_variant": row.get("variant"),
        "scenario_id": scenario.get("item_id"),
        "rollout": row.get("rollout"),
        "round": trace.get("round"),
        "reward": row.get("reward"),
        "messages": [
            {"role": "system", "content": "You are a buyer-side strategic planner for RLVR negotiation."},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }


def make_generator_example(
    row: Dict[str, Any],
    trace: Dict[str, Any],
    history_before: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    raw = trace.get("buyer_raw")
    action = trace.get("buyer_action") or {}
    validator = trace.get("validator") or {}
    if not raw or not action.get("valid") or validator.get("repaired"):
        return None
    scenario = row.get("scenario") or {}
    user = {
        "task": "RLVR buyer natural-language generator SFT",
        "instruction": (
            "Generate the next buyer response in exactly Thought/Talk/Action format. "
            "Use the plan and belief as advisory signals, stay within budget, keep bargaining, "
            "and use persuasive concise language paired with a concrete action."
        ),
        "scenario": {
            "item_id": scenario.get("item_id"),
            "title": scenario.get("title"),
            "buyer_budget": scenario.get("buyer_budget"),
            "reference_price": scenario.get("reference_price"),
            "codename": scenario.get("codename"),
            "quantity": scenario.get("quantity"),
        },
        "history_before": history_before,
        "belief": trace.get("belief"),
        "plan": trace.get("plan"),
        "concession_schedule": extract_schedule(trace),
    }
    return {
        "type": "generator",
        "source_variant": row.get("variant"),
        "scenario_id": scenario.get("item_id"),
        "rollout": row.get("rollout"),
        "round": trace.get("round"),
        "reward": row.get("reward"),
        "messages": [
            {"role": "system", "content": "You are the buyer response generator in an RLVR negotiation."},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            {"role": "assistant", "content": raw.strip()},
        ],
    }


def extract_schedule(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = trace.get("buyer_prompt") or ""
    marker = "Private concession schedule. Follow it unless accepting a lower formal seller [SELL] offer:"
    if marker not in prompt:
        return None
    tail = prompt.split(marker, 1)[1].strip()
    first_line = tail.splitlines()[0].strip()
    try:
        return json.loads(first_line)
    except Exception:
        return None


def build_examples(
    row: Dict[str, Any],
    *,
    include_planner: bool,
    include_generator: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    transcript = row.get("transcript") or []
    traces = row.get("framework_trace") or []
    buyer_indices = buyer_turn_indices(transcript)
    planner_examples: List[Dict[str, Any]] = []
    generator_examples: List[Dict[str, Any]] = []
    for idx, trace in enumerate(traces):
        if idx >= len(buyer_indices):
            break
        if not trace.get("belief") or not trace.get("plan"):
            continue
        history_before = [compact_turn(turn) for turn in transcript[: buyer_indices[idx]]]
        if include_planner:
            planner_examples.append(make_planner_example(row, trace, history_before))
        if include_generator:
            generator = make_generator_example(row, trace, history_before)
            if generator is not None:
                generator_examples.append(generator)
    return planner_examples, generator_examples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Simple_Env run directory. Repeatable.")
    parser.add_argument("--variants", default="full_framework,counterfactual_response_belief,typed_belief")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-reward", type=float, default=0.65)
    parser.add_argument("--require-deal", action="store_true", default=True)
    parser.add_argument("--allow-walkaway", dest="require_deal", action="store_false")
    parser.add_argument("--max-first-turn-offer-ratio", type=float, default=None)
    parser.add_argument("--planner", action="store_true", default=True)
    parser.add_argument("--no-planner", dest="planner", action="store_false")
    parser.add_argument("--generator", action="store_true", default=True)
    parser.add_argument("--no-generator", dest="generator", action="store_false")
    args = parser.parse_args()

    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    planner_rows: List[Dict[str, Any]] = []
    generator_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "episodes_seen": 0,
        "episodes_kept": 0,
        "planner_examples": 0,
        "generator_examples": 0,
        "by_variant": {},
        "filters": {
            "variants": sorted(variants),
            "min_reward": args.min_reward,
            "require_deal": args.require_deal,
            "max_first_turn_offer_ratio": args.max_first_turn_offer_ratio,
        },
    }

    for path in episode_files([Path(item) for item in args.run_dir], variants):
        variant = path.name.removesuffix("_episodes.jsonl")
        stats["by_variant"].setdefault(variant, {"seen": 0, "kept": 0, "planner_examples": 0, "generator_examples": 0})
        for row in load_jsonl(path):
            stats["episodes_seen"] += 1
            stats["by_variant"][variant]["seen"] += 1
            if not accept_episode(
                row,
                min_reward=args.min_reward,
                require_deal=args.require_deal,
                max_first_turn_offer_ratio=args.max_first_turn_offer_ratio,
            ):
                continue
            p_rows, g_rows = build_examples(row, include_planner=args.planner, include_generator=args.generator)
            if not p_rows and not g_rows:
                continue
            stats["episodes_kept"] += 1
            stats["by_variant"][variant]["kept"] += 1
            stats["by_variant"][variant]["planner_examples"] += len(p_rows)
            stats["by_variant"][variant]["generator_examples"] += len(g_rows)
            planner_rows.extend(p_rows)
            generator_rows.extend(g_rows)

    out = Path(args.output_dir)
    write_jsonl(out / "planner_sft.jsonl", planner_rows)
    write_jsonl(out / "generator_sft.jsonl", generator_rows)
    stats["planner_examples"] = len(planner_rows)
    stats["generator_examples"] = len(generator_rows)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
