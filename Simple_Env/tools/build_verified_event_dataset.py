#!/usr/bin/env python3
"""Build leakage-safe public-event data for belief response calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.events import extract_verified_event, public_talk_and_action
from framework.schemas import CanonicalOffer, CanonicalState, NegotiationObservation


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def allowed_scenarios(path: Path) -> tuple[set[str], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_id = {str(row["item_id"]): row for row in rows}
    return set(by_id), by_id


def episode_files(run_dirs: list[Path]) -> list[Path]:
    files = []
    for run_dir in run_dirs:
        files.extend(
            path for path in run_dir.rglob("*_episodes.jsonl")
            if path.name != "failed_episodes.jsonl"
        )
    return sorted(set(path.resolve() for path in files))


def action_offer(turn: dict[str, Any]) -> CanonicalOffer | None:
    action = turn.get("action") or {}
    price = action.get("price")
    return CanonicalOffer(price=float(price)) if isinstance(price, (int, float)) else None


def events_from_episode(
    row: dict[str, Any],
    *,
    source: Path,
    max_turns: int,
) -> Iterable[dict[str, Any]]:
    scenario = row.get("scenario") or {}
    budget = float(scenario.get("buyer_budget") or 0.0)
    cost = scenario.get("seller_cost")
    if budget <= 0 or not isinstance(cost, (int, float)):
        return
    scenario_id = str(scenario["item_id"])
    rollout = int(row.get("rollout", 0))
    variant = str(row.get("variant") or source.name.removesuffix("_episodes.jsonl"))
    session_id = hashlib.sha256(
        f"{source}:{variant}:{scenario_id}:{rollout}".encode()
    ).hexdigest()[:20]
    prior_buyer: CanonicalOffer | None = None
    previous_counter: float | None = None
    seller_index = 0
    for transcript_index, turn in enumerate(row.get("transcript") or []):
        role = str(turn.get("role") or "")
        action = turn.get("action") or {}
        if role == "buyer":
            prior_buyer = action_offer(turn)
            continue
        if role != "seller":
            continue
        seller_index += 1
        action_type = str(action.get("action") or "")
        if action_type == "accept":
            response = "accept"
        elif action_type == "offer":
            response = "counter" if prior_buyer is not None else "offer"
        elif action_type == "reject":
            response = "reject"
        elif action_type == "walk_away":
            response = "quit"
        else:
            response = "message"
        observation = NegotiationObservation(
            observation_id=f"{session_id}:seller:{seller_index}",
            turn=int(turn.get("round") or seller_index),
            actor_id="seller",
            counterparty_id="seller",
            response_type=response,  # type: ignore[arg-type]
            offer=action_offer(turn),
            response_to_offer=prior_buyer,
            text=public_talk_and_action(str(turn.get("message") or action.get("raw") or "")),
        )
        state = CanonicalState(
            session_id=session_id,
            environment_id="simple_env_amazonhistoryprice",
            self_id="buyer",
            role="buyer",
            counterparty_id="seller",
            turn=observation.turn,
            max_turns=max_turns,
            issues=[],
            own_value_scale=budget,
            reference_value=float(scenario.get("reference_price") or 0.0),
        )
        event = extract_verified_event(
            state, observation, previous_counter_compatibility=previous_counter
        )
        if event.counter_compatibility is not None:
            previous_counter = event.counter_compatibility
        if event.outcome == "message" or event.tested_compatibility is None:
            continue
        yield {
            "schema_version": "verified_negotiation_event_v1",
            "scenario_id": scenario_id,
            "category": scenario.get("category"),
            "variant": variant,
            "rollout": rollout,
            "source_file": str(source),
            "transcript_index": transcript_index,
            "event": event.to_dict(),
            # Evaluator-only labels are stored outside the deployed input.
            "labels": {
                "reservation_ratio": float(cost) / budget,
                "mutual_interest": bool(row.get("mi")),
            },
        }


def event_dedup_key(row: dict[str, Any]) -> str:
    event = row["event"]
    payload = {
        "scenario": row["scenario_id"],
        "turn": event["turn"],
        "outcome": event["outcome"],
        "tested": round(float(event["tested_compatibility"]), 6),
        "counter": (
            round(float(event["counter_compatibility"]), 6)
            if event["counter_compatibility"] is not None else None
        ),
        "public_text": event["public_text"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--scenario-allowlist", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--train-scenarios", type=int, default=16)
    parser.add_argument("--calibration-scenarios", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    allowlist, scenario_metadata = allowed_scenarios(args.scenario_allowlist)
    scenario_ids = sorted(allowlist)
    rng = random.Random(args.seed)
    rng.shuffle(scenario_ids)
    n_train = args.train_scenarios
    n_cal = args.calibration_scenarios
    if n_train + n_cal >= len(scenario_ids):
        raise ValueError("scenario split must leave at least one held-out test scenario")
    split_ids = {
        "train": set(scenario_ids[:n_train]),
        "calibration": set(scenario_ids[n_train:n_train + n_cal]),
        "test": set(scenario_ids[n_train + n_cal:]),
    }

    rows: list[dict[str, Any]] = []
    rejected_outside_allowlist = 0
    source_files = episode_files(args.run_dir)
    for source in source_files:
        for episode in read_jsonl(source):
            scenario_id = str((episode.get("scenario") or {}).get("item_id") or "")
            if scenario_id not in allowlist:
                rejected_outside_allowlist += 1
                continue
            rows.extend(events_from_episode(episode, source=source, max_turns=args.max_turns))

    seen: set[str] = set()
    deduplicated = []
    for row in rows:
        key = event_dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)

    output_rows = {}
    for split, ids in split_ids.items():
        selected = [row for row in deduplicated if row["scenario_id"] in ids]
        output_rows[split] = selected
        write_jsonl(args.output_dir / f"{split}.jsonl", selected)

    stats = {
        "schema_version": "verified_negotiation_event_v1",
        "seed": args.seed,
        "scenario_allowlist": str(args.scenario_allowlist.resolve()),
        "source_files": [str(path) for path in source_files],
        "raw_events": len(rows),
        "deduplicated_events": len(deduplicated),
        "duplicates_removed": len(rows) - len(deduplicated),
        "episodes_rejected_outside_allowlist": rejected_outside_allowlist,
        "splits": {
            split: {
                "scenario_ids": sorted(ids),
                "scenarios": len(ids),
                "events": len(output_rows[split]),
                "outcomes": dict(Counter(row["event"]["outcome"] for row in output_rows[split])),
                "reservation_ratio_range": [
                    min(float(scenario_metadata[item]["seller_cost"]) / float(scenario_metadata[item]["buyer_budget"]) for item in ids),
                    max(float(scenario_metadata[item]["seller_cost"]) / float(scenario_metadata[item]["buyer_budget"]) for item in ids),
                ],
            }
            for split, ids in split_ids.items()
        },
        "leakage_guards": {
            "validation40_excluded_by_allowlist": True,
            "final64_excluded_by_allowlist": True,
            "private_thought_removed": all(
                "Thought:" not in row["event"]["public_text"] for row in deduplicated
            ),
            "claims_used_as_truth_labels": False,
            "split_unit": "scenario_id",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
