#!/usr/bin/env python3
"""Collect controlled one-response seller events on an external scenario set.

Offer ratio and nominal negotiation progress are independently randomized by
scenario.  Each job exposes one budget-safe buyer offer and records only the
seller's public response; evaluator cost is stored separately as a label.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario
from experiments.model_clients import make_model_client
from framework.events import extract_verified_event, public_talk_and_action
from framework.schemas import CanonicalOffer, CanonicalState, NegotiationObservation
from Simple_Env.seller import make_seller


def read_scenarios(path: Path, limit: int) -> list[RLVRScenario]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result.append(RLVRScenario(
            item_id=str(row.get("item_id") or row.get("id")),
            title=str(row.get("title") or row.get("name") or "AmazonHistoryPrice item"),
            buyer_budget=float(row["buyer_budget"]),
            seller_cost=float(row["seller_cost"]),
            reference_price=float(row["reference_price"]),
            category=str(row.get("category") or "provided"),
            codename=str(row.get("codename") or f"item_{len(result)}"),
            description=str(row.get("description") or ""),
            features=str(row.get("features") or ""),
            quantity=int(row.get("quantity", 1)),
        ))
        if len(result) >= limit:
            break
    if len(result) != limit:
        raise ValueError(f"Requested {limit} scenarios, loaded {len(result)} from {path}")
    return result


def split_scenarios(ids: list[str], seed: int, train_n: int, calibration_n: int) -> dict[str, set[str]]:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    if train_n + calibration_n >= len(shuffled):
        raise ValueError("split must leave held-out test scenarios")
    return {
        "train": set(shuffled[:train_n]),
        "calibration": set(shuffled[train_n:train_n + calibration_n]),
        "test": set(shuffled[train_n + calibration_n:]),
    }


def job_key(scenario_id: str, probe_id: int, rollout: int) -> str:
    return f"{scenario_id}:{probe_id}:{rollout}"


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["collection_key"])
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def controlled_rounds(scenario_id: str, n: int, seed: int) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{scenario_id}".encode()).digest()
    local = random.Random(int.from_bytes(digest[:8], "big"))
    rounds = list(range(1, 7))
    local.shuffle(rounds)
    return [rounds[index % len(rounds)] for index in range(n)]


def collect_one(
    *,
    scenario: RLVRScenario,
    probe_id: int,
    offer_ratio: float,
    round_id: int,
    rollout: int,
    seller_client: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    seller = make_seller("default", seller_client, max_tokens, temperature, top_p, "neutral")
    price = round(min(scenario.buyer_budget, offer_ratio * scenario.buyer_budget), 2)
    buyer_raw = (
        "Thought: This is a controlled public offer.\n"
        "Talk: I can close promptly if this price works for you.\n"
        f"Action: [BUY] ${price:.2f} ({scenario.quantity}x {scenario.codename_or_default()})"
    )
    buyer_action = {
        "role": "buyer", "action": "offer", "price": price, "raw": buyer_raw,
        "valid": True, "item_spec": f"{scenario.quantity}x {scenario.codename_or_default()}",
    }
    history = [{"round": round_id, "role": "buyer", "message": buyer_raw, "action": buyer_action}]
    parsed, raw = seller.act(
        scenario=scenario, history=history, round_id=round_id, max_turns=6
    )
    if parsed.action == "accept":
        response = "accept"
    elif parsed.action == "offer":
        response = "counter"
    elif parsed.action == "reject":
        response = "reject"
    elif parsed.action == "walk_away":
        response = "quit"
    else:
        response = "message"
    session_id = hashlib.sha256(
        f"external2025:{scenario.item_id}:{probe_id}:{rollout}".encode()
    ).hexdigest()[:20]
    observation = NegotiationObservation(
        observation_id=f"{session_id}:seller:1",
        turn=round_id,
        actor_id="seller",
        counterparty_id="seller",
        response_type=response,  # type: ignore[arg-type]
        offer=CanonicalOffer(price=float(parsed.price)) if parsed.price is not None else None,
        response_to_offer=CanonicalOffer(price=price),
        text=public_talk_and_action(raw),
    )
    state = CanonicalState(
        session_id=session_id,
        environment_id="simple_env_amazonhistoryprice_interventional",
        self_id="buyer",
        role="buyer",
        counterparty_id="seller",
        turn=round_id,
        max_turns=6,
        issues=[],
        own_value_scale=scenario.buyer_budget,
        reference_value=scenario.reference_price,
    )
    event = extract_verified_event(state, observation)
    return {
        "schema_version": "verified_negotiation_event_v1",
        "collection_key": job_key(scenario.item_id, probe_id, rollout),
        "scenario_id": scenario.item_id,
        "category": scenario.category,
        "variant": "controlled_interventional_probe",
        "rollout": rollout,
        "source_file": "amazonhistoryprice_test128_seed2025_budget0.8.jsonl",
        "transcript_index": 1,
        "probe": {
            "probe_id": probe_id,
            "offer_ratio": offer_ratio,
            "nominal_round": round_id,
        },
        "event": event.to_dict(),
        "labels": {
            "reservation_ratio": scenario.seller_cost / scenario.buyer_budget,
            "mutual_interest": scenario.buyer_budget > scenario.seller_cost,
        },
    }


def write_split_outputs(raw_path: Path, output_dir: Path, splits: dict[str, set[str]]) -> dict[str, Any]:
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = {}
    for split, ids in splits.items():
        selected = [row for row in rows if row["scenario_id"] in ids]
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
        stats[split] = {"scenarios": len(ids), "events": len(selected), "scenario_ids": sorted(ids)}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai@http://127.0.0.1:8002/v1:Qwen3-30B-A3B-Instruct-2507-base")
    parser.add_argument("--scenarios-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--instances", type=int, default=128)
    parser.add_argument("--train-scenarios", type=int, default=80)
    parser.add_argument("--calibration-scenarios", type=int, default=24)
    parser.add_argument("--probe-ratios", default="0.30,0.45,0.60,0.75,0.90,0.98")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT", "1800")
    scenarios = read_scenarios(args.scenarios_jsonl, args.instances)
    splits = split_scenarios(
        [scenario.item_id for scenario in scenarios], args.seed,
        args.train_scenarios, args.calibration_scenarios,
    )
    ratios = [float(item) for item in args.probe_ratios.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "all_events.jsonl"
    done = completed_keys(raw_path) if args.resume else set()
    jobs = []
    for scenario in scenarios:
        rounds = controlled_rounds(scenario.item_id, len(ratios), args.seed)
        for probe_id, (ratio, round_id) in enumerate(zip(ratios, rounds)):
            for rollout in range(args.rollouts):
                if job_key(scenario.item_id, probe_id, rollout) not in done:
                    jobs.append((scenario, probe_id, ratio, round_id, rollout))
    client = make_model_client(args.model, enable_thinking=False)
    lock = threading.Lock()
    errors = []

    def run(job: tuple[Any, ...]) -> dict[str, Any]:
        scenario, probe_id, ratio, round_id, rollout = job
        return collect_one(
            scenario=scenario, probe_id=probe_id, offer_ratio=ratio,
            round_id=round_id, rollout=rollout, seller_client=client,
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        completed = len(done)
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                job = futures.pop(future)
                try:
                    row = future.result()
                    with lock, raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                    completed += 1
                    print(json.dumps({"completed": completed, "expected": args.instances * len(ratios) * args.rollouts, "key": row["collection_key"]}), flush=True)
                except Exception as exc:
                    scenario, probe_id, _ratio, _round, rollout = job
                    errors.append({"key": job_key(scenario.item_id, probe_id, rollout), "error": f"{type(exc).__name__}: {exc}"})
    if errors:
        (args.output_dir / "errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Collection finished with {len(errors)} errors; rerun with --resume")
    # A successful resumed collection supersedes errors from an earlier transient
    # endpoint failure.  Keeping that stale file would make a healthy dataset look
    # failed to downstream monitors and human reviewers.
    (args.output_dir / "errors.json").unlink(missing_ok=True)
    split_stats = write_split_outputs(raw_path, args.output_dir, splits)
    manifest = {
        "schema_version": "interventional_verified_events_v1",
        "model": args.model,
        "scenarios_jsonl": str(args.scenarios_jsonl.resolve()),
        "seed": args.seed,
        "probe_ratios": ratios,
        "rollouts": args.rollouts,
        "events": args.instances * len(ratios) * args.rollouts,
        "splits": split_stats,
        "design": "offer ratio and nominal progress independently permuted within scenario",
        "leakage_guards": {
            "source_is_external_seed2025": True,
            "private_thought_removed": True,
            "claim_truth_used_as_label": False,
            "split_unit": "scenario_id",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
