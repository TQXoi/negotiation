#!/usr/bin/env python3
"""Analyze paired Simple Env runs without changing any episode data.

The script reports outcome quality, belief calibration, action locking, and
paired action/outcome flips.  It intentionally keeps stochastic outcome flips
separate from belief calibration metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", default="universal_framework_v2")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_runs(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(root.glob("*_episodes.jsonl")):
        if path.name == "failed_episodes.jsonl":
            # Retry audit records are not completed benchmark episodes and
            # must never be interpreted as a buyer variant.
            continue
        variant = path.name.removesuffix("_episodes.jsonl")
        result[variant] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not result:
        raise FileNotFoundError(f"No *_episodes.jsonl files found under {root}")
    return result


def episode_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row["scenario"]["item_id"]), int(row.get("rollout", 0))


def scenario_key(row: Mapping[str, Any]) -> str:
    """Return the experimental cluster id shared by repeated rollouts."""
    return str(row["scenario"]["item_id"])


def buyer_sequence(row: Mapping[str, Any]) -> List[Tuple[str, float | None]]:
    sequence: List[Tuple[str, float | None]] = []
    for item in row.get("transcript") or []:
        if item.get("role") != "buyer":
            continue
        action = item.get("action") or {}
        price = action.get("price")
        sequence.append((str(action.get("action")), round(float(price), 6) if price is not None else None))
    return sequence


def selected_trace_score(trace: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selected = trace.get("selected_candidate_id")
    for candidate in trace.get("ranked_candidates") or []:
        if (candidate.get("candidate") or {}).get("candidate_id") == selected:
            return candidate
    return None


def seller_response_by_round(row: Mapping[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for item in row.get("transcript") or []:
        if item.get("role") == "seller":
            result[int(item.get("round", 0))] = str((item.get("action") or {}).get("action"))
    return result


def ece(points: Sequence[Tuple[float, int]], bins: int = 5) -> float | None:
    if not points:
        return None
    total = len(points)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        local = [item for item in points if low <= item[0] < high or (index == bins - 1 and item[0] == 1.0)]
        if not local:
            continue
        value += len(local) / total * abs(
            statistics.mean(item[0] for item in local)
            - statistics.mean(item[1] for item in local)
        )
    return value


def framework_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    belief_errors: List[float] = []
    coverage: List[int] = []
    calibration: List[Tuple[float, int]] = []
    lock_values: List[int] = []
    semantic_parse: List[int] = []
    frontier_decisions = 0
    selected_behind_frontier = 0
    selected_frontier_penalties: List[float] = []
    for row in rows:
        traces = row.get("framework_trace") or []
        scenario = row.get("scenario") or {}
        budget = float(scenario.get("buyer_budget") or 0.0)
        cost = scenario.get("seller_cost")
        if traces and budget > 0 and isinstance(cost, (int, float)):
            belief = traces[-1].get("belief") or {}
            preference = belief.get("preference") or {}
            truth = float(cost) / budget
            mean = preference.get("reservation_ratio_mean")
            low = preference.get("reservation_ratio_low")
            high = preference.get("reservation_ratio_high")
            if isinstance(mean, (int, float)):
                belief_errors.append(abs(float(mean) - truth))
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                coverage.append(int(float(low) <= truth <= float(high)))
        responses = seller_response_by_round(row)
        for trace in traces:
            validation = trace.get("validation") or {}
            if "action_lock_ok" in validation:
                lock_values.append(int(bool(validation["action_lock_ok"])))
            update = trace.get("semantic_update")
            if isinstance(update, dict) and "parsed" in update:
                semantic_parse.append(int(bool(update["parsed"])))
            selected = selected_trace_score(trace)
            if selected and "behavioral_frontier_penalty" in (selected.get("diagnostics") or {}):
                diagnostics = selected.get("diagnostics") or {}
                frontier_decisions += 1
                selected_behind_frontier += int(bool(diagnostics.get("behind_rejected_frontier")))
                selected_frontier_penalties.append(
                    float(diagnostics.get("behavioral_frontier_penalty") or 0.0)
                )
            if not selected or (selected.get("candidate") or {}).get("action_type") != "offer":
                continue
            turn = int(trace.get("turn", 0))
            response = responses.get(turn)
            if response is None:
                continue
            calibration.append((float(selected.get("p_accept", 0.0)), int(response == "accept")))
    return {
        "belief_last_turn_mae": statistics.mean(belief_errors) if belief_errors else None,
        "belief_interval_coverage": statistics.mean(coverage) if coverage else None,
        "belief_episode_count": len(belief_errors),
        "acceptance_brier": statistics.mean((p - y) ** 2 for p, y in calibration) if calibration else None,
        "acceptance_ece_5bin": ece(calibration),
        "acceptance_observations": len(calibration),
        "action_lock_rate": statistics.mean(lock_values) if lock_values else None,
        "action_lock_observations": len(lock_values),
        "semantic_parse_rate": statistics.mean(semantic_parse) if semantic_parse else None,
        "semantic_updates": len(semantic_parse),
        "frontier_decisions": frontier_decisions,
        "selected_behind_frontier_rate": (
            selected_behind_frontier / frontier_decisions if frontier_decisions else None
        ),
        "mean_selected_frontier_penalty": (
            statistics.mean(selected_frontier_penalties)
            if selected_frontier_penalties else None
        ),
    }


def cost_budget_stratum(row: Mapping[str, Any]) -> str:
    scenario = row.get("scenario") or {}
    budget = float(scenario.get("buyer_budget") or 0.0)
    cost = float(scenario.get("seller_cost") or 0.0)
    ratio = cost / budget if budget > 0 else math.inf
    if ratio < 0.40:
        return "wide_lt_0.40"
    if ratio < 0.75:
        return "medium_0.40_0.75"
    if ratio < 1.0:
        return "tight_0.75_1.00"
    return "no_mutual_interest_ge_1.00"


def outcome_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rewards = [float(row.get("reward") or 0.0) for row in rows]
    mutual = [row for row in rows if bool(row.get("mi"))]
    deals = [row for row in rows if row.get("status") == "deal"]
    repeated_final_offer_failures = 0
    for row in mutual:
        offers = [
            price for action, price in buyer_sequence(row)
            if action == "offer" and price is not None
        ]
        repeated_final_offer_failures += int(
            row.get("status") != "deal"
            and len(offers) >= 2
            and offers[-1] == offers[-2]
        )
    return {
        "episodes": len(rows),
        "mean_reward": statistics.mean(rewards) if rewards else None,
        "deal_rate": len(deals) / len(rows) if rows else None,
        "mutual_interest_episodes": len(mutual),
        "mutual_interest_deal_rate": (
            sum(row.get("status") == "deal" for row in mutual) / len(mutual)
            if mutual else None
        ),
        "mutual_interest_buyer_quit": sum(
            row.get("status") != "deal" and row.get("terminal_reason") == "buyer_quit"
            for row in mutual
        ),
        "mutual_interest_seller_quit": sum(
            row.get("status") != "deal" and row.get("terminal_reason") == "seller_quit"
            for row in mutual
        ),
        "mutual_interest_repeated_final_offer_failures": repeated_final_offer_failures,
    }


def bootstrap_ci(values: Sequence[float], samples: int, seed: int = 0) -> List[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def paired(reference: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]], samples: int) -> Dict[str, Any]:
    ref = {episode_key(row): row for row in reference}
    alt = {episode_key(row): row for row in other}
    keys = sorted(set(ref) & set(alt))
    deltas = [float(ref[key].get("reward") or 0.0) - float(alt[key].get("reward") or 0.0) for key in keys]
    first_flip = 0
    sequence_flip = 0
    agreement_flip = 0
    for key in keys:
        ref_sequence = buyer_sequence(ref[key])
        alt_sequence = buyer_sequence(alt[key])
        first_flip += int(bool(ref_sequence and alt_sequence and ref_sequence[0] != alt_sequence[0]))
        sequence_flip += int(ref_sequence != alt_sequence)
        agreement_flip += int((ref[key].get("status") == "deal") != (alt[key].get("status") == "deal"))
    wins = sum(delta > 1e-12 for delta in deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas)
    return {
        "paired_episodes": len(keys),
        "mean_reference_reward_delta": statistics.mean(deltas) if deltas else None,
        "median_reference_reward_delta": statistics.median(deltas) if deltas else None,
        "bootstrap_95_ci": bootstrap_ci(deltas, samples),
        "reference_wins": wins,
        "ties": ties,
        "reference_losses": len(deltas) - wins - ties,
        "first_action_flip_rate": first_flip / len(keys) if keys else None,
        "offer_sequence_flip_rate": sequence_flip / len(keys) if keys else None,
        "agreement_outcome_flip_rate": agreement_flip / len(keys) if keys else None,
    }


def cluster_paired(
    reference: Sequence[Mapping[str, Any]],
    other: Sequence[Mapping[str, Any]],
    samples: int,
) -> Dict[str, Any]:
    """Paired inference with scenario, rather than episode, as the unit.

    Repeated rollouts of one scenario share the same private values and are not
    independent benchmark items.  We therefore pair matching rollout ids,
    average their deltas within each scenario, and bootstrap those scenario
    means.  This prevents three rollouts from being reported as three times as
    many independent examples.
    """
    ref: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    alt: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for row in reference:
        ref.setdefault(scenario_key(row), {})[int(row.get("rollout", 0))] = row
    for row in other:
        alt.setdefault(scenario_key(row), {})[int(row.get("rollout", 0))] = row

    reward_deltas: List[float] = []
    agreement_deltas: List[float] = []
    mutual_interest_agreement_deltas: List[float] = []
    first_flips: List[float] = []
    sequence_flips: List[float] = []
    agreement_flips: List[float] = []
    rollouts_per_cluster: List[int] = []
    cluster_rows: List[Dict[str, Any]] = []

    for scenario in sorted(set(ref) & set(alt)):
        rollout_ids = sorted(set(ref[scenario]) & set(alt[scenario]))
        if not rollout_ids:
            continue
        local_reward: List[float] = []
        local_agreement: List[float] = []
        local_mi_agreement: List[float] = []
        local_first_flip: List[float] = []
        local_sequence_flip: List[float] = []
        local_agreement_flip: List[float] = []
        for rollout in rollout_ids:
            ref_row = ref[scenario][rollout]
            alt_row = alt[scenario][rollout]
            ref_deal = ref_row.get("status") == "deal"
            alt_deal = alt_row.get("status") == "deal"
            ref_sequence = buyer_sequence(ref_row)
            alt_sequence = buyer_sequence(alt_row)
            local_reward.append(
                float(ref_row.get("reward") or 0.0)
                - float(alt_row.get("reward") or 0.0)
            )
            local_agreement.append(float(ref_deal) - float(alt_deal))
            if bool(ref_row.get("mi")) and bool(alt_row.get("mi")):
                local_mi_agreement.append(float(ref_deal) - float(alt_deal))
            local_first_flip.append(float(
                bool(ref_sequence and alt_sequence and ref_sequence[0] != alt_sequence[0])
            ))
            local_sequence_flip.append(float(ref_sequence != alt_sequence))
            local_agreement_flip.append(float(ref_deal != alt_deal))

        reward_delta = statistics.mean(local_reward)
        agreement_delta = statistics.mean(local_agreement)
        reward_deltas.append(reward_delta)
        agreement_deltas.append(agreement_delta)
        if local_mi_agreement:
            mutual_interest_agreement_deltas.append(statistics.mean(local_mi_agreement))
        first_flips.append(statistics.mean(local_first_flip))
        sequence_flips.append(statistics.mean(local_sequence_flip))
        agreement_flips.append(statistics.mean(local_agreement_flip))
        rollouts_per_cluster.append(len(rollout_ids))
        cluster_rows.append({
            "scenario": scenario,
            "paired_rollouts": len(rollout_ids),
            "reference_reward_delta": reward_delta,
            "reference_agreement_delta": agreement_delta,
        })

    wins = sum(delta > 1e-12 for delta in reward_deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in reward_deltas)
    return {
        "estimand": "reference_minus_comparator; rollouts averaged within scenario",
        "scenario_clusters": len(reward_deltas),
        "paired_episodes": sum(rollouts_per_cluster),
        "rollouts_per_cluster_min": min(rollouts_per_cluster) if rollouts_per_cluster else None,
        "rollouts_per_cluster_max": max(rollouts_per_cluster) if rollouts_per_cluster else None,
        "mean_reference_reward_delta": (
            statistics.mean(reward_deltas) if reward_deltas else None
        ),
        "reward_cluster_bootstrap_95_ci": bootstrap_ci(reward_deltas, samples),
        "mean_reference_agreement_rate_delta": (
            statistics.mean(agreement_deltas) if agreement_deltas else None
        ),
        "agreement_cluster_bootstrap_95_ci": bootstrap_ci(agreement_deltas, samples, seed=1),
        "mean_reference_mutual_interest_agreement_rate_delta": (
            statistics.mean(mutual_interest_agreement_deltas)
            if mutual_interest_agreement_deltas else None
        ),
        "mutual_interest_agreement_cluster_bootstrap_95_ci": bootstrap_ci(
            mutual_interest_agreement_deltas, samples, seed=2
        ),
        "reference_scenario_wins": wins,
        "scenario_ties": ties,
        "reference_scenario_losses": len(reward_deltas) - wins - ties,
        "mean_first_action_flip_rate": statistics.mean(first_flips) if first_flips else None,
        "mean_offer_sequence_flip_rate": (
            statistics.mean(sequence_flips) if sequence_flips else None
        ),
        "mean_agreement_outcome_flip_rate": (
            statistics.mean(agreement_flips) if agreement_flips else None
        ),
        "scenario_deltas": cluster_rows,
    }


def main() -> None:
    args = parse_args()
    variants = load_runs(args.run_dir)
    if args.reference not in variants:
        raise KeyError(f"Reference {args.reference!r} not found; have {sorted(variants)}")
    output = {
        "run_dir": str(args.run_dir),
        "reference": args.reference,
        "diagnostics": {
            name: framework_diagnostics(rows) for name, rows in variants.items()
        },
        "outcomes": {
            name: outcome_diagnostics(rows) for name, rows in variants.items()
        },
        "stratified_outcomes": {
            name: {
                stratum: outcome_diagnostics(
                    [row for row in rows if cost_budget_stratum(row) == stratum]
                )
                for stratum in [
                    "wide_lt_0.40",
                    "medium_0.40_0.75",
                    "tight_0.75_1.00",
                    "no_mutual_interest_ge_1.00",
                ]
            }
            for name, rows in variants.items()
        },
        "paired": {
            name: paired(variants[args.reference], rows, args.bootstrap_samples)
            for name, rows in variants.items() if name != args.reference
        },
        "cluster_paired": {
            name: cluster_paired(variants[args.reference], rows, args.bootstrap_samples)
            for name, rows in variants.items() if name != args.reference
        },
    }
    target = args.output or args.run_dir / "universal_framework_diagnostics.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
