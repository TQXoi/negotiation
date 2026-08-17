from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.model_clients import make_model_client

from CaSiNo_Env.buyer.factory import make_buyer
from CaSiNo_Env.environment.casino import load_casino_split
from CaSiNo_Env.environment.episode import run_episode
from CaSiNo_Env.environment.io import append_jsonl, read_jsonl, write_json
from CaSiNo_Env.environment.metrics import summarize_episodes
from CaSiNo_Env.seller.factory import make_seller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular CaSiNo fixed-partner evaluations.")
    parser.add_argument(
        "--casino-split-json",
        default=str(ROOT / "CaSiNo_Env" / "data" / "casino" / "casino_test.json"),
    )
    parser.add_argument("--require-casino-split-json", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--buyer-model", default="openai:Qwen3-30B-A3B-Instruct-2507-base")
    parser.add_argument("--seller-model", default=None)
    parser.add_argument("--buyer-variants", default="direct_prompt,full_framework")
    parser.add_argument("--seller-type", default="llm_partner")
    parser.add_argument("--partner-personality", default="base")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--rollouts-per-instance", type=int, default=1)
    parser.add_argument("--n-round", type=int, default=12)
    parser.add_argument("--buyer-temperature", type=float, default=0.7)
    parser.add_argument("--seller-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "casino" / "smoke"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed_keys(path: Path) -> Set[Tuple[str, int]]:
    rows = read_jsonl(path)
    return {(str(row.get("scenario")), int(row.get("rollout", 0))) for row in rows}


def main() -> None:
    args = parse_args()
    split_path = Path(args.casino_split_json)
    if args.require_casino_split_json and not split_path.exists():
        raise FileNotFoundError(split_path)
    scenarios = load_casino_split(split_path, split=args.split, limit=args.num_instances)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))
    with (output_dir / "scenarios.jsonl").open("w") as f:
        for scenario in scenarios:
            f.write(json.dumps({"scenario": scenario.scenario_id, "dialogue_id": scenario.dialogue_id}, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "num_scenarios": len(scenarios), "output_dir": str(output_dir)}, ensure_ascii=False))
        return

    buyer_model = make_model_client(args.buyer_model)
    seller_model = make_model_client(args.seller_model or args.buyer_model)
    variants = [item.strip() for item in args.buyer_variants.split(",") if item.strip()]
    all_rows: List[Dict[str, Any]] = []

    for variant in variants:
        episode_path = output_dir / f"{variant}_episodes.jsonl"
        done = completed_keys(episode_path) if args.resume else set()
        if done:
            print(json.dumps({"buyer": variant, "resumed_episodes": len(done)}, ensure_ascii=False))
        for scenario in scenarios:
            for rollout in range(args.rollouts_per_instance):
                key = (scenario.scenario_id, rollout)
                if key in done:
                    continue
                buyer = make_buyer(
                    variant,
                    buyer_model,
                    temperature=args.buyer_temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    candidate_top_k=args.candidate_top_k,
                )
                seller = make_seller(
                    args.seller_type,
                    seller_model,
                    personality=args.partner_personality,
                    temperature=args.seller_temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                row = run_episode(scenario, buyer, seller, max_rounds=args.n_round, rollout=rollout)
                append_jsonl(episode_path, row)
                all_rows.append(row)
                print(json.dumps({
                    "episode_result": {
                        "buyer": row["buyer"],
                        "seller": row["seller"],
                        "partner_personality": args.partner_personality,
                        "scenario": row["scenario"],
                        "rollout": rollout,
                        "status": row["status"],
                        "terminal_reason": row["terminal_reason"],
                        "p1_score": row["p1_score"],
                        "p2_score": row["p2_score"],
                        "score_ratio_p1": row["score_ratio_p1"],
                        "rounds": row["rounds"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                }, ensure_ascii=False))
                if args.summary_every > 0 and len(all_rows) % args.summary_every == 0:
                    print(json.dumps({"partial_summary": summarize_episodes(all_rows)}, ensure_ascii=False))

    summary = {}
    for variant in variants:
        rows = list(read_jsonl(output_dir / f"{variant}_episodes.jsonl"))
        summary[variant] = summarize_episodes(rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
