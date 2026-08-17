#!/usr/bin/env python3
"""Side-by-side trajectory comparison for Simple_Env RLVR episode JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-run",
        default="runs/simple/continuous_solver",
        help="Run directory for the first system, e.g. continuous solver.",
    )
    parser.add_argument(
        "--left-variant",
        default="continuous_hybrid_ev",
        help="Variant name in left run.",
    )
    parser.add_argument(
        "--right-run",
        default="runs/simple/full_framework_teacher",
        help="Run directory for the second system, e.g. previous full_framework.",
    )
    parser.add_argument(
        "--right-variant",
        default="full_framework",
        help="Variant name in right run.",
    )
    parser.add_argument("--scenario", default=None, help="Only compare one item_id.")
    parser.add_argument("--rollout", type=int, default=None, help="Only compare one rollout.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum aligned scenario/rollout pairs to print.")
    parser.add_argument("--no-thought", action="store_true", help="Hide Thought lines.")
    parser.add_argument(
        "--show-framework-trace",
        action="store_true",
        help="Show compact belief/planner diagnostics for each buyer turn when available.",
    )
    parser.add_argument(
        "--show-candidates",
        type=int,
        default=0,
        help="When showing framework trace, print top-N scored candidates for each buyer turn.",
    )
    parser.add_argument(
        "--only-common",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only show pairs present in both runs. Use --no-only-common to also list missing pairs.",
    )
    return parser.parse_args()


def load_variant(run_dir: Path, variant: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = run_dir / f"{variant}_episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing variant file: {path}")
    episodes: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            ep = json.loads(line)
            ep.setdefault("variant", variant)
            ep["_source_file"] = str(path)
            ep["_line_no"] = line_no
            key = episode_key(ep)
            episodes[key] = ep
    return episodes


def episode_key(ep: Dict[str, Any]) -> Tuple[str, int]:
    scenario = ep.get("scenario") or {}
    item_id = scenario.get("item_id") if isinstance(scenario, dict) else str(scenario)
    return str(item_id), int(ep.get("rollout", 0))


def strip_thought(message: str) -> str:
    lines = []
    for line in (message or "").splitlines():
        if line.strip().lower().startswith("thought:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compact_message(message: str, hide_thought: bool) -> str:
    text = strip_thought(message) if hide_thought else (message or "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def action_line(turn: Dict[str, Any]) -> str:
    action = turn.get("action") or {}
    kind = action.get("action")
    price = action.get("price")
    valid = action.get("valid")
    if price is None:
        return f"{kind}, valid={valid}"
    return f"{kind} ${float(price):.2f}, valid={valid}"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_line(ep: Optional[Dict[str, Any]]) -> str:
    if ep is None:
        return "MISSING"
    scenario = ep.get("scenario") or {}
    return (
        f"{ep.get('variant')} | status={ep.get('status')} ({ep.get('terminal_reason')}) | "
        f"reward={ep.get('reward')} | final={ep.get('final_price')} | "
        f"bargain={ep.get('buyer_bargained_ratio')} | rounds={ep.get('rounds')} | "
        f"budget={scenario.get('buyer_budget')} | cost={scenario.get('seller_cost')} | ref={scenario.get('reference_price')}"
    )


def belief_cost_diagnostics(belief: Dict[str, Any], seller_cost: Optional[float]) -> Dict[str, Any]:
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


def compact_trace_for_round(ep: Dict[str, Any], round_id: Any, *, top_n_candidates: int = 0) -> Optional[str]:
    scenario = ep.get("scenario") or {}
    seller_cost = _float_or_none(scenario.get("seller_cost"))
    traces = ep.get("framework_trace") or []
    for trace in traces:
        if trace.get("round") != round_id:
            continue
        planner = trace.get("planner") or {}
        selected = planner.get("selected") or {}
        plan = (planner.get("plan") or trace.get("plan") or {})
        belief = trace.get("belief") or {}
        interaction = belief.get("interaction") or {}
        reservation = belief.get("reservation") or {}
        parts = []
        if selected:
            parts.append(
                "selected="
                + json.dumps(
                    {
                        "type": selected.get("action_type"),
                        "price": selected.get("price"),
                        "ev": selected.get("ev"),
                        "p_accept": selected.get("p_accept"),
                        "p_quit": selected.get("p_quit"),
                    },
                    ensure_ascii=False,
                )
            )
        candidates = planner.get("candidates") or []
        if top_n_candidates > 0 and isinstance(candidates, list) and candidates:
            compact_candidates = []
            for candidate in candidates[:top_n_candidates]:
                if not isinstance(candidate, dict):
                    continue
                compact_candidates.append(
                    {
                        "type": candidate.get("action_type"),
                        "price": candidate.get("price"),
                        "ev": candidate.get("ev"),
                        "p_accept": candidate.get("p_accept"),
                        "p_quit": candidate.get("p_quit"),
                        "reward": candidate.get("reward_if_deal"),
                    }
                )
            if compact_candidates:
                parts.append("top_candidates=" + json.dumps(compact_candidates, ensure_ascii=False))
        elif plan:
            parts.append(
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
        diagnostics = belief_cost_diagnostics(belief, seller_cost)
        compact_model_belief = compact_model_belief_from_trace(trace)
        if compact_model_belief:
            parts.append("compact_model_belief=" + json.dumps(compact_model_belief, ensure_ascii=False))
        full_debug_belief = full_debug_belief_summary(trace)
        if full_debug_belief:
            parts.append("full_debug_belief_summary=" + json.dumps(full_debug_belief, ensure_ascii=False))
        generation = trace.get("generation") or {}
        validation = generation.get("validator") or {}
        selected_candidate = validation.get("selected_candidate")
        if selected_candidate:
            parts.append("generator_selected=" + json.dumps(selected_candidate, ensure_ascii=False))
        if diagnostics:
            parts.append("belief_vs_cost=" + json.dumps(diagnostics, ensure_ascii=False))
        elif reservation or interaction:
            parts.append(
                "belief="
                + json.dumps(
                    {
                        "reservation": {
                            "mean": reservation.get("mean"),
                            "p10": reservation.get("p10"),
                            "p50": reservation.get("p50"),
                            "p90": reservation.get("p90"),
                            "confidence": reservation.get("confidence"),
                        },
                        "interaction": {
                            "finality": interaction.get("finality_prob"),
                            "quit_risk": interaction.get("quit_risk"),
                            "patience": interaction.get("patience"),
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(parts) if parts else None
    return None


def print_episode(
    label: str,
    ep: Optional[Dict[str, Any]],
    *,
    hide_thought: bool,
    show_trace: bool,
    top_n_candidates: int,
) -> None:
    print(f"### {label}")
    print(metric_line(ep))
    if ep is None:
        print()
        return
    for turn in ep.get("transcript") or []:
        role = str(turn.get("role", "")).upper()
        round_id = turn.get("round")
        print(f"- Round {round_id} {role} [{action_line(turn)}]")
        print(indent(compact_message(turn.get("message") or "", hide_thought), "  "))
        if show_trace and role == "BUYER":
            trace = compact_trace_for_round(ep, round_id, top_n_candidates=top_n_candidates)
            if trace:
                print(indent("[trace]\n" + trace, "  "))
        print()


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def filter_keys(keys: Iterable[Tuple[str, int]], args: argparse.Namespace) -> List[Tuple[str, int]]:
    out = []
    for item_id, rollout in sorted(keys):
        if args.scenario and item_id != args.scenario:
            continue
        if args.rollout is not None and rollout != args.rollout:
            continue
        out.append((item_id, rollout))
    if args.limit is not None:
        out = out[: args.limit]
    return out


def main() -> None:
    args = parse_args()
    left = load_variant(Path(args.left_run), args.left_variant)
    right = load_variant(Path(args.right_run), args.right_variant)
    keys = set(left) & set(right) if args.only_common else set(left) | set(right)
    selected_keys = filter_keys(keys, args)

    print("# Trajectory Comparison")
    print(f"- left: {args.left_variant} @ {args.left_run} ({len(left)} episodes)")
    print(f"- right: {args.right_variant} @ {args.right_run} ({len(right)} episodes)")
    print(f"- shown pairs: {len(selected_keys)}")
    print()

    for idx, key in enumerate(selected_keys, start=1):
        left_ep = left.get(key)
        right_ep = right.get(key)
        scenario = (left_ep or right_ep or {}).get("scenario") or {}
        print("=" * 100)
        print(f"## Pair {idx}: {key[0]} | rollout={key[1]}")
        print(f"title: {scenario.get('title')}")
        print()
        print_episode(
            f"LEFT: {args.left_variant}",
            left_ep,
            hide_thought=args.no_thought,
            show_trace=args.show_framework_trace,
            top_n_candidates=args.show_candidates,
        )
        print_episode(
            f"RIGHT: {args.right_variant}",
            right_ep,
            hide_thought=args.no_thought,
            show_trace=args.show_framework_trace,
            top_n_candidates=args.show_candidates,
        )


if __name__ == "__main__":
    main()
