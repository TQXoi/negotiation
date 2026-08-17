#!/usr/bin/env python3
"""Readable trajectory viewer for Simple_Env RLVR episode JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/simple/test_run",
        help="Directory containing *_episodes.jsonl files.",
    )
    parser.add_argument("--buyer", default=None, help="Only show one buyer variant. Alias of --variant.")
    parser.add_argument(
        "--variant",
        default=None,
        help="Only show selected buyer variant(s). Use comma-separated names, e.g. direct_prompt,full_framework.",
    )
    parser.add_argument("--scenario", default=None, help="Only show one scenario/item_id.")
    parser.add_argument("--rollout", type=int, default=None, help="Only show one rollout id.")
    parser.add_argument("--status", default=None, help="Only show one terminal status, e.g. deal or walk_away.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of episodes to print.")
    parser.add_argument("--no-thought", action="store_true", help="Hide Thought lines and show Talk/Action only.")
    parser.add_argument(
        "--show-belief",
        action="store_true",
        help="Show belief/planner trace and seller-cost calibration after buyer turns.",
    )
    parser.add_argument("--raw-json", action="store_true", help="Print compact JSON records instead of readable text.")
    return parser.parse_args()


def load_episodes(run_dir: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_episodes.jsonl")):
        if path.name == "failed_episodes.jsonl":
            continue
        buyer_from_file = path.name.removesuffix("_episodes.jsonl")
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc
                row.setdefault("variant", buyer_from_file)
                row["_source_file"] = str(path)
                row["_line_no"] = line_no
                episodes.append(row)
    return episodes


def keep_episode(ep: Dict[str, Any], args: argparse.Namespace) -> bool:
    scenario = ep.get("scenario") or {}
    variant_filter = args.variant or args.buyer
    if variant_filter:
        variants = {item.strip() for item in variant_filter.split(",") if item.strip()}
        if ep.get("variant") not in variants:
            return False
    if args.scenario and str(scenario.get("item_id")) != args.scenario:
        return False
    if args.rollout is not None and int(ep.get("rollout", -1)) != args.rollout:
        return False
    if args.status and ep.get("status") != args.status:
        return False
    return True


def action_line(turn: Dict[str, Any]) -> str:
    action = turn.get("action") or {}
    kind = action.get("action")
    price = action.get("price")
    valid = action.get("valid")
    if price is None:
        return f"{kind}, valid={valid}"
    return f"{kind} ${float(price):.2f}, valid={valid}"


def strip_thought(message: str) -> str:
    lines = []
    for line in (message or "").splitlines():
        if line.strip().lower().startswith("thought:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compact_message(message: str, hide_thought: bool) -> str:
    text = strip_thought(message) if hide_thought else (message or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def trace_for_round(ep: Dict[str, Any], round_id: Any) -> Optional[Dict[str, Any]]:
    for trace in ep.get("framework_trace") or []:
        if trace.get("round") == round_id:
            return trace
    return None


def belief_cost_diagnostics(belief: Dict[str, Any], seller_cost: Optional[float]) -> Dict[str, Any]:
    """Compare belief to true seller cost.

    Seller cost is the hard lower bound in RLVR, not necessarily the seller's
    strategic reservation. These diagnostics should be read as calibration
    signals, not perfect correctness labels.
    """

    if not belief or seller_cost is None:
        return {}
    cost = float(seller_cost)
    out: Dict[str, Any] = {"seller_cost": round(cost, 4)}
    reservation = belief.get("reservation") if isinstance(belief.get("reservation"), dict) else None
    if reservation:
        mean = _float_or_none(reservation.get("mean"))
        p10 = _float_or_none(reservation.get("p10"))
        p50 = _float_or_none(reservation.get("p50"))
        p90 = _float_or_none(reservation.get("p90"))
        out["reservation"] = {
            "mean": round(mean, 4) if mean is not None else None,
            "p10": round(p10, 4) if p10 is not None else None,
            "p50": round(p50, 4) if p50 is not None else None,
            "p90": round(p90, 4) if p90 is not None else None,
            "confidence": reservation.get("confidence"),
        }
        if p10 is not None and p90 is not None:
            out["cost_in_p10_p90"] = p10 <= cost <= p90
            out["p10_minus_cost"] = round(p10 - cost, 4)
            out["p90_minus_cost"] = round(p90 - cost, 4)
        if mean is not None:
            out["mean_minus_cost"] = round(mean - cost, 4)
            out["mean_cost_ratio"] = round(mean / max(cost, 1e-9), 4)
        if p50 is not None:
            out["p50_minus_cost"] = round(p50 - cost, 4)
    else:
        interval = belief.get("seller_reservation_range")
        estimate = _float_or_none(belief.get("seller_reservation_estimate"))
        if isinstance(interval, list) and len(interval) >= 2:
            low = _float_or_none(interval[0])
            high = _float_or_none(interval[1])
            out["seller_reservation_range"] = [low, high]
            if low is not None and high is not None:
                out["cost_in_range"] = low <= cost <= high
                out["low_minus_cost"] = round(low - cost, 4)
                out["high_minus_cost"] = round(high - cost, 4)
        if estimate is not None:
            out["estimate_minus_cost"] = round(estimate - cost, 4)
            out["estimate_cost_ratio"] = round(estimate / max(cost, 1e-9), 4)
    interaction = belief.get("interaction") if isinstance(belief.get("interaction"), dict) else None
    if interaction:
        out["interaction"] = {
            "finality": interaction.get("finality_prob"),
            "quit_risk": interaction.get("quit_risk"),
            "patience": interaction.get("patience"),
        }
    return out


def compact_model_belief_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    belief = trace.get("belief") or {}
    terms = belief.get("contract_term_preferences") if isinstance(belief, dict) else None
    if isinstance(terms, dict) and isinstance(terms.get("compact_bayesian_posterior"), dict):
        compact = terms["compact_bayesian_posterior"]
        return {
            "source": "prompt_compact_bayesian_posterior",
            "reservation": compact.get("reservation"),
            "acceptance_curve": compact.get("acceptance_curve"),
            "interaction": compact.get("interaction"),
            "posterior_entropy": compact.get("posterior_entropy"),
            "recent_evidence": compact.get("recent_evidence"),
        }
    if isinstance(belief, dict):
        return {
            "source": "trace_belief_summary",
            "reservation": belief.get("reservation"),
            "seller_reservation_range": belief.get("seller_reservation_range"),
            "seller_reservation_estimate": belief.get("seller_reservation_estimate"),
            "likely_acceptable_price_range": belief.get("likely_acceptable_price_range"),
            "deal_risk": belief.get("deal_risk"),
            "evidence": (belief.get("evidence") or [])[-4:],
        }
    return {}


def full_debug_belief_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    persistent = trace.get("persistent_bayesian_belief")
    if not isinstance(persistent, dict):
        return {}
    return {
        "source": "full_persistent_bayesian_belief_debug",
        "reservation": persistent.get("reservation"),
        "acceptance_curve_head": (persistent.get("acceptance_curve") or [])[:8],
        "interaction": persistent.get("interaction"),
        "concession": persistent.get("concession"),
        "evidence_count": len(persistent.get("evidence") or []),
        "recent_evidence": (persistent.get("evidence") or [])[-4:],
        "metadata": persistent.get("metadata"),
    }


def compact_trace(trace: Dict[str, Any], seller_cost: Optional[float]) -> str:
    planner = trace.get("planner") or {}
    selected = planner.get("selected") or {}
    plan = planner.get("plan") or trace.get("plan") or {}
    belief = trace.get("belief") or {}
    lines = []
    if selected:
        lines.append(
            "selected="
            + json.dumps(
                {
                    "type": selected.get("action_type"),
                    "price": selected.get("price"),
                    "ev": selected.get("ev"),
                    "p_accept": selected.get("p_accept"),
                    "p_quit": selected.get("p_quit"),
                    "reward_if_deal": selected.get("reward_if_deal"),
                },
                ensure_ascii=False,
            )
        )
    elif plan:
        lines.append(
            "plan="
            + json.dumps(
                {
                    "act": plan.get("strategic_act"),
                    "target": plan.get("target_price"),
                    "rationale": plan.get("private_rationale"),
                },
                ensure_ascii=False,
            )
        )
    compact_model_belief = compact_model_belief_from_trace(trace)
    if compact_model_belief:
        lines.append("compact_model_belief=" + json.dumps(compact_model_belief, ensure_ascii=False))
    full_debug_belief = full_debug_belief_summary(trace)
    if full_debug_belief:
        lines.append("full_debug_belief_summary=" + json.dumps(full_debug_belief, ensure_ascii=False))
    diagnostics = belief_cost_diagnostics(belief, seller_cost)
    if diagnostics:
        lines.append("belief_vs_cost=" + json.dumps(diagnostics, ensure_ascii=False))
    return "\n".join(lines)


def print_episode(ep: Dict[str, Any], hide_thought: bool, show_belief: bool = False) -> None:
    scenario = ep.get("scenario") or {}
    seller_cost = _float_or_none(scenario.get("seller_cost"))
    header = (
        f"## {ep.get('variant')} | {scenario.get('item_id')} | rollout={ep.get('rollout')} | "
        f"status={ep.get('status')} ({ep.get('terminal_reason')}) | reward={ep.get('reward')}"
    )
    print(header)
    print(
        f"- title: {scenario.get('title')}\n"
        f"- buyer_budget: {scenario.get('buyer_budget')} | seller_cost: {scenario.get('seller_cost')} | "
        f"reference_price: {scenario.get('reference_price')} | mutual_interest: {ep.get('mi')}\n"
        f"- final_price: {ep.get('final_price')} | bargained_ratio: {ep.get('buyer_bargained_ratio')} | "
        f"overshoot: {ep.get('buyer_overshoot')} | format_violation: {ep.get('buyer_format_violation')} | "
        f"rounds: {ep.get('rounds')} | elapsed_seconds: {ep.get('elapsed_seconds')}\n"
    )
    for turn in ep.get("transcript") or []:
        role = str(turn.get("role", "")).upper()
        round_id = turn.get("round")
        print(f"### Round {round_id} {role} [{action_line(turn)}]")
        print(compact_message(turn.get("message") or "", hide_thought))
        if show_belief and role == "BUYER":
            trace = trace_for_round(ep, round_id)
            if trace:
                rendered = compact_trace(trace, seller_cost)
                if rendered:
                    print("\n[belief/planner trace]")
                    print(rendered)
        print()
    print("---")


def summarize(episodes: Iterable[Dict[str, Any]], show_belief: bool = False) -> None:
    eps = list(episodes)
    if not eps:
        print("No matching episodes.")
        return
    by_status: Dict[str, int] = {}
    by_buyer: Dict[str, int] = {}
    for ep in eps:
        by_status[str(ep.get("status"))] = by_status.get(str(ep.get("status")), 0) + 1
        by_buyer[str(ep.get("variant"))] = by_buyer.get(str(ep.get("variant")), 0) + 1
    avg_reward = sum(float(ep.get("reward") or 0.0) for ep in eps) / len(eps)
    deal_rate = sum(1 for ep in eps if ep.get("status") == "deal") / len(eps)
    print(
        f"# Trajectory Summary\n"
        f"- episodes: {len(eps)}\n"
        f"- avg_reward: {avg_reward:.6f}\n"
        f"- deal_rate: {deal_rate:.6f}\n"
        f"- by_status: {by_status}\n"
        f"- by_buyer: {by_buyer}\n"
    )
    if show_belief:
        total = 0
        hits = 0
        mean_errors = []
        p50_errors = []
        floor_over = 0
        for ep in eps:
            scenario = ep.get("scenario") or {}
            seller_cost = _float_or_none(scenario.get("seller_cost"))
            if seller_cost is None:
                continue
            for trace in ep.get("framework_trace") or []:
                diag = belief_cost_diagnostics(trace.get("belief") or {}, seller_cost)
                if not diag:
                    continue
                total += 1
                if diag.get("cost_in_p10_p90") is True or diag.get("cost_in_range") is True:
                    hits += 1
                if "mean_minus_cost" in diag:
                    mean_errors.append(float(diag["mean_minus_cost"]))
                if "p50_minus_cost" in diag:
                    p50_errors.append(float(diag["p50_minus_cost"]))
                low_err = diag.get("p10_minus_cost", diag.get("low_minus_cost"))
                if isinstance(low_err, (int, float)) and low_err > 0:
                    floor_over += 1
        if total:
            avg_mean_error = f"{sum(mean_errors) / len(mean_errors):.6f}" if mean_errors else "NA"
            avg_p50_error = f"{sum(p50_errors) / len(p50_errors):.6f}" if p50_errors else "NA"
            print(
                f"# Belief Calibration vs Seller Cost\n"
                f"- traced_beliefs: {total}\n"
                f"- cost_interval_hit_rate: {hits / total:.6f}\n"
                f"- floor_overestimate_rate: {floor_over / total:.6f}\n"
                f"- avg_mean_minus_cost: {avg_mean_error}\n"
                f"- avg_p50_minus_cost: {avg_p50_error}\n"
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    episodes = [ep for ep in load_episodes(run_dir) if keep_episode(ep, args)]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if args.raw_json:
        for ep in episodes:
            print(json.dumps(ep, ensure_ascii=False))
        return
    summarize(episodes, show_belief=args.show_belief)
    for ep in episodes:
        print_episode(ep, args.no_thought, show_belief=args.show_belief)


if __name__ == "__main__":
    main()
