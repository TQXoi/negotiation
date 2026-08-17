#!/usr/bin/env python3
"""Read AgenticPay_Env trajectories with framework belief/plan diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def compact_json(value: Any, max_chars: int = 1400) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated>"


def print_score_formula(row: Dict[str, Any]) -> None:
    buyer_max = row.get("buyer_max_price")
    seller_min = row.get("seller_min_price")
    final_price = row.get("agreed_price")
    if final_price is None:
        final_price = row.get("buyer_price") or row.get("seller_price")
    gamma = row.get("gamma") or 0.99
    rounds = row.get("total_rounds") or len(row.get("rounds") or []) or 1
    print("\n[BuyerScore Formula]")
    print("If feasible deal and valid price range:")
    print("  Z = buyer_max_price - seller_min_price")
    print("  u_b = (buyer_max_price - final_price) / Z")
    print("  discount = gamma^(rounds - 1)")
    print("  BuyerScore = discount * (Db + Wb * u_b + Eb)")
    print("Default single28 weights in AgenticPay code: Db=10, Wb=80, Eb=10, Fb=15, gamma=0.99")
    print("If no feasible valid deal:")
    print("  BuyerScore = -Fb * (1 - discount)")
    print(
        "Observed: "
        f"buyer_max={fmt(buyer_max)}, seller_min={fmt(seller_min)}, final_price={fmt(final_price)}, "
        f"rounds={fmt(rounds)}, gamma={fmt(gamma)}, buyer_score={fmt(row.get('buyer_score'))}"
    )
    if all(isinstance(x, (int, float)) for x in [buyer_max, seller_min, final_price]):
        z = float(buyer_max) - float(seller_min)
        if z > 0:
            ub = (float(buyer_max) - float(final_price)) / z
            discount = float(gamma) ** max(0, int(rounds) - 1)
            score = discount * (10.0 + 80.0 * ub + 10.0)
            print(f"Recomputed single-price success score: Z={z:.3f}, u_b={ub:.4f}, discount={discount:.6f}, score={score:.3f}")


def trace_for_round(traces: List[Dict[str, Any]], index: int, round_id: Any) -> Optional[Dict[str, Any]]:
    if index < len(traces):
        return traces[index]
    for trace in traces:
        if trace.get("round") == round_id:
            return trace
    return None


def print_trace(trace: Optional[Dict[str, Any]], *, show_belief: bool, show_plan: bool, show_raw: bool) -> None:
    if not trace:
        return
    if show_belief:
        belief = trace.get("belief")
        print("\n  [belief]")
        print(compact_json(belief))
    if show_plan:
        plan = trace.get("plan")
        print("\n  [plan]")
        print(compact_json(plan))
    if show_raw:
        naturalization = trace.get("naturalization") or {}
        print("\n  [naturalization]")
        print(compact_json(naturalization))


def row_matches(row: Dict[str, Any], *, variant: Optional[str], task: Optional[str], index: Optional[int]) -> bool:
    if variant and row.get("buyer_variant") != variant:
        return False
    if task:
        hay = f"{row.get('task_path', '')} {row.get('task', '')}"
        if task not in hay:
            return False
    if index is not None and row.get("task_index") != index:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Run directory containing task_results.jsonl.")
    parser.add_argument("--run-dir", dest="run_dir_flag", help="Same as positional run_dir.")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--task", default=None, help="Substring filter for task path/name, e.g. Task1.")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--show-belief", action="store_true")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--show-score-formula", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Only print header and final scores.")
    args = parser.parse_args()

    run_dir_raw = args.run_dir_flag or args.run_dir
    if not run_dir_raw:
        raise SystemExit("Provide run_dir as a positional argument or --run-dir.")
    path = Path(run_dir_raw) / "task_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    count = 0
    for row in load_rows(path):
        if not row_matches(row, variant=args.variant, task=args.task, index=args.task_index):
            continue
        count += 1
        print("=" * 110)
        print(
            f"{row.get('buyer_variant')} | {row.get('task_path')} | "
            f"success={row.get('success')} term={row.get('termination_reason')} rounds={row.get('total_rounds')}"
        )
        print(
            f"score: global={fmt(row.get('global_score'))} buyer={fmt(row.get('buyer_score'))} "
            f"buyer_fixed={fmt(row.get('buyer_score_fixed_seller'))} seller={fmt(row.get('seller_score'))}"
        )
        print(
            f"prices: buyer_max={fmt(row.get('buyer_max_price'))} seller_min={fmt(row.get('seller_min_price'))} "
            f"agreed={fmt(row.get('agreed_price'))} buyer_final={fmt(row.get('buyer_price'))} seller_final={fmt(row.get('seller_price'))}"
        )
        if row.get("fixed_seller_eval_reason"):
            print(f"fixed_seller_reason: {row.get('fixed_seller_eval_reason')}")
        if args.show_score_formula:
            print_score_formula(row)
        if not args.compact:
            traces = row.get("framework_traces") or row.get("traces") or []
            for idx, turn in enumerate(row.get("rounds") or []):
                print("\n" + "-" * 80)
                print(f"[Round {turn.get('round')}]")
                print(f"BUYER valid={turn.get('buyer_format_valid')} price={fmt(turn.get('buyer_price_extracted'))}")
                print(str(turn.get("buyer_action") or "").strip())
                print(f"\nSELLER valid={turn.get('seller_format_valid')} price={fmt(turn.get('seller_price_extracted'))}")
                print(str(turn.get("seller_action") or "").strip())
                print(
                    f"\nstep reward={fmt(turn.get('reward'))} buyer_step={fmt(turn.get('step_buyer_reward'))} "
                    f"seller_step={fmt(turn.get('step_seller_reward'))} terminated={turn.get('terminated')}"
                )
                print_trace(
                    trace_for_round(traces, idx, turn.get("round")),
                    show_belief=args.show_belief,
                    show_plan=args.show_plan,
                    show_raw=args.show_raw,
                )
        if count >= args.limit:
            break


if __name__ == "__main__":
    main()
