#!/usr/bin/env python3
"""Fixed-seller buyer metrics for AgenticPay trajectories.

The released AgenticPay score is symmetric environment outcome scoring: if the
seller fails to parse, refuses a feasible buyer offer, or otherwise prevents a
deal, the final `buyer_score` can still become a buyer failure penalty. That is
not appropriate for fixed-seller buyer ablations.

This module preserves the original fields and adds:

- `buyer_score_original`
- `buyer_score_fixed_seller`
- `buyer_score_fixed_seller_censored`
- `seller_error_attributed`
- `fixed_seller_eval_reason`

The corrected score uses a conservative oracle-accept rule: if the buyer made a
valid price offer that is within the buyer budget and at/above the seller
minimum, a rational fixed seller could have accepted it. If the recorded episode
did not reach a deal, buyer evaluation uses the best discounted buyer score among
those feasible buyer offers and attributes the miss to the seller.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_GAMMA = 0.99
DEFAULT_BUYER_DEAL_WEIGHT = 10.0
DEFAULT_BUYER_UTILITY_WEIGHT = 80.0
DEFAULT_BUYER_EFFICIENCY_WEIGHT = 10.0


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, bool)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: Iterable[Any]) -> Optional[float]:
    nums = [float(value) for value in values if isinstance(value, (int, float, bool))]
    return mean(nums) if nums else None


def discounted_buyer_score(
    *,
    price: float,
    buyer_max_price: float,
    seller_min_price: float,
    round_count: int,
    gamma: float = DEFAULT_GAMMA,
    buyer_deal_weight: float = DEFAULT_BUYER_DEAL_WEIGHT,
    buyer_utility_weight: float = DEFAULT_BUYER_UTILITY_WEIGHT,
    buyer_efficiency_weight: float = DEFAULT_BUYER_EFFICIENCY_WEIGHT,
) -> Optional[float]:
    spread = buyer_max_price - seller_min_price
    if spread <= 0 or price < seller_min_price or price > buyer_max_price:
        return None
    round_index = max(0, int(round_count) - 1)
    discount = gamma ** round_index
    buyer_utility = (buyer_max_price - price) / spread
    return discount * (buyer_deal_weight + buyer_utility_weight * buyer_utility + buyer_efficiency_weight)


def extract_contract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"<contract>\s*(.*?)\s*</contract>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        contract = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(contract, dict):
        return None
    try:
        price = float(contract["price"])
    except (KeyError, TypeError, ValueError):
        return None
    continuous_terms = contract.get("continuous_terms", {})
    discrete_terms = contract.get("discrete_terms", {})
    if not isinstance(continuous_terms, dict) or not isinstance(discrete_terms, dict):
        return None
    return {"price": price, "continuous_terms": continuous_terms, "discrete_terms": discrete_terms}


def term_weight(weights: Dict[str, Any], term: str, value: Any) -> float:
    term_weights = weights.get(term, {}) if isinstance(weights, dict) else {}
    if not isinstance(term_weights, dict):
        return 0.0
    if value in term_weights:
        return float(term_weights.get(value, 0.0))
    return float(term_weights.get(str(value), 0.0))


def contract_utilities(contract: Dict[str, Any], config: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    try:
        price = float(contract["price"])
    except (KeyError, TypeError, ValueError):
        return None
    buyer_prefs = config.get("buyer_preferences", {})
    seller_prefs = config.get("seller_preferences", {})
    continuous_terms = contract.get("continuous_terms", {}) if isinstance(contract.get("continuous_terms"), dict) else {}
    discrete_terms = contract.get("discrete_terms", {}) if isinstance(contract.get("discrete_terms"), dict) else {}

    for term, bounds in config.get("continuous_bounds", {}).items():
        value = numeric(continuous_terms.get(term))
        if value is None:
            return None
        if value < float(bounds.get("min", value)) or value > float(bounds.get("max", value)):
            return None
    for term, options in config.get("discrete_options", {}).items():
        if discrete_terms.get(term) not in options:
            return None

    buyer_utility = float(buyer_prefs.get("v_base", 0.0)) - price
    seller_utility = price - float(seller_prefs.get("c_base", 0.0))
    for term, raw_value in continuous_terms.items():
        value = numeric(raw_value)
        if value is None:
            return None
        buyer_utility += float(buyer_prefs.get("continuous_weights", {}).get(term, 0.0)) * value
        seller_utility += float(seller_prefs.get("continuous_weights", {}).get(term, 0.0)) * value
    for term, raw_value in discrete_terms.items():
        buyer_utility += term_weight(buyer_prefs.get("discrete_weights", {}), term, raw_value)
        seller_utility += term_weight(seller_prefs.get("discrete_weights", {}), term, raw_value)

    z_max = float(buyer_prefs.get("v_base", 0.0)) - float(seller_prefs.get("c_base", 0.0))
    for term, bounds in config.get("continuous_bounds", {}).items():
        total_weight = float(buyer_prefs.get("continuous_weights", {}).get(term, 0.0)) + float(seller_prefs.get("continuous_weights", {}).get(term, 0.0))
        min_v = float(bounds.get("min", 0.0))
        max_v = float(bounds.get("max", 0.0))
        z_max += total_weight * (max_v if total_weight >= 0 else min_v)
    for term, options in config.get("discrete_options", {}).items():
        best = None
        for option in options:
            value = (
                term_weight(buyer_prefs.get("discrete_weights", {}), term, option)
                + term_weight(seller_prefs.get("discrete_weights", {}), term, option)
            )
            best = value if best is None else max(best, value)
        if best is not None:
            z_max += best
    return buyer_utility, seller_utility, z_max


def discounted_contract_buyer_score(
    *,
    buyer_utility: float,
    seller_utility: float,
    z_max: float,
    round_count: int,
    gamma: float = DEFAULT_GAMMA,
    buyer_deal_weight: float = DEFAULT_BUYER_DEAL_WEIGHT,
    buyer_utility_weight: float = DEFAULT_BUYER_UTILITY_WEIGHT,
    buyer_efficiency_weight: float = DEFAULT_BUYER_EFFICIENCY_WEIGHT,
) -> Optional[float]:
    if z_max <= 0 or buyer_utility < 0 or seller_utility < 0:
        return None
    round_index = max(0, int(round_count) - 1)
    discount = gamma ** round_index
    return discount * (buyer_deal_weight + buyer_utility_weight * (buyer_utility / z_max) + buyer_efficiency_weight)


def feasible_buyer_offer_scores(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    buyer_max = numeric(record.get("buyer_max_price"))
    seller_min = numeric(record.get("seller_min_price"))
    contract_config = record.get("contract_config")
    if isinstance(contract_config, dict) and contract_config:
        return feasible_buyer_contract_scores(record, contract_config)
    if buyer_max is None or seller_min is None:
        return []
    offers: List[Dict[str, Any]] = []
    for idx, turn in enumerate(record.get("rounds") or [], start=1):
        buyer_valid = bool(turn.get("buyer_format_valid"))
        price = numeric(turn.get("buyer_price_extracted"))
        if price is None:
            price = numeric(turn.get("buyer_price_state"))
        if not buyer_valid or price is None:
            continue
        if not (seller_min <= price <= buyer_max):
            continue
        round_count = int(numeric(turn.get("round")) or idx)
        score = discounted_buyer_score(
            price=price,
            buyer_max_price=buyer_max,
            seller_min_price=seller_min,
            round_count=round_count,
        )
        if score is not None:
            offers.append({"round": round_count, "price": price, "buyer_score": score})
    return offers


def feasible_buyer_contract_scores(record: Dict[str, Any], contract_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    offers: List[Dict[str, Any]] = []
    for idx, turn in enumerate(record.get("rounds") or [], start=1):
        if not bool(turn.get("buyer_format_valid")):
            continue
        contract = extract_contract(str(turn.get("buyer_action") or ""))
        if contract is None:
            continue
        utilities = contract_utilities(contract, contract_config)
        if utilities is None:
            continue
        buyer_utility, seller_utility, z_max = utilities
        round_count = int(numeric(turn.get("round")) or idx)
        score = discounted_contract_buyer_score(
            buyer_utility=buyer_utility,
            seller_utility=seller_utility,
            z_max=z_max,
            round_count=round_count,
        )
        if score is not None:
            offers.append(
                {
                    "round": round_count,
                    "price": contract["price"],
                    "buyer_score": score,
                    "buyer_utility": buyer_utility,
                    "seller_utility": seller_utility,
                    "z_max": z_max,
                    "contract": contract,
                }
            )
    return offers


def seller_format_problem(record: Dict[str, Any]) -> bool:
    rounds = record.get("rounds") or []
    if not rounds:
        return False
    for turn in rounds:
        if turn.get("seller_format_valid") is False:
            return True
        if turn.get("seller_price_extracted") is None and turn.get("seller_price_state") is None:
            return True
    return False


def attach_fixed_seller_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(record)
    original = output.get("buyer_score")
    output.setdefault("buyer_score_original", original)

    if output.get("success") is True or output.get("termination_reason") == "agreed":
        output["buyer_score_fixed_seller"] = original
        output["buyer_score_fixed_seller_censored"] = False
        output["seller_error_attributed"] = False
        output["fixed_seller_eval_reason"] = "agreed_original_score"
        return output

    feasible_scores = feasible_buyer_offer_scores(output)
    if feasible_scores:
        best = max(feasible_scores, key=lambda item: item["buyer_score"])
        output["buyer_score_fixed_seller"] = best["buyer_score"]
        output["buyer_score_fixed_seller_censored"] = False
        output["seller_error_attributed"] = True
        output["fixed_seller_eval_reason"] = "seller_missed_feasible_buyer_offer"
        output["fixed_seller_best_feasible_buyer_offer"] = best
        output["fixed_seller_num_feasible_buyer_offers"] = len(feasible_scores)
        return output

    if seller_format_problem(output):
        output["buyer_score_fixed_seller"] = None
        output["buyer_score_fixed_seller_censored"] = True
        output["seller_error_attributed"] = True
        output["fixed_seller_eval_reason"] = "seller_format_or_missing_price_without_buyer_feasible_offer"
        return output

    output["buyer_score_fixed_seller"] = original
    output["buyer_score_fixed_seller_censored"] = False
    output["seller_error_attributed"] = False
    output["fixed_seller_eval_reason"] = "no_seller_error_detected"
    return output


def summarize_fixed_seller(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    no_seller_error_records = [
        record for record in records
        if record.get("seller_error_attributed") is False
    ]
    deal_records = [
        record for record in records
        if record.get("success") is True or record.get("termination_reason") == "agreed"
    ]
    return {
        "avg_buyer_score_original": avg(record.get("buyer_score_original", record.get("buyer_score")) for record in records),
        "avg_buyer_score_fixed_seller": avg(record.get("buyer_score_fixed_seller") for record in records),
        "avg_buyer_score_no_seller_error": avg(
            record.get("buyer_score_original", record.get("buyer_score"))
            for record in no_seller_error_records
        ),
        "avg_buyer_score_deals_only": avg(
            record.get("buyer_score_original", record.get("buyer_score"))
            for record in deal_records
        ),
        "num_no_seller_error_records": len(no_seller_error_records),
        "num_deal_records": len(deal_records),
        "seller_error_attribution_rate": avg(record.get("seller_error_attributed") for record in records),
        "buyer_score_fixed_seller_censor_rate": avg(record.get("buyer_score_fixed_seller_censored") for record in records),
        "fixed_seller_eval_reasons": count_by(records, "fixed_seller_eval_reason"),
    }


def count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_summary(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = dict(summary)
    fixed = summarize_fixed_seller(records)
    if "overall" in output and isinstance(output["overall"], dict):
        output["overall"] = {**output["overall"], **fixed}
    else:
        output["fixed_seller_overall"] = fixed
    return output


def patch_run_dir(run_dir: Path, *, in_place: bool) -> Dict[str, Any]:
    jsonl_candidates = [run_dir / "trajectories.jsonl", run_dir / "task_results.jsonl"]
    jsonl_path = next((path for path in jsonl_candidates if path.exists()), None)
    if jsonl_path is None:
        raise FileNotFoundError(f"No trajectories.jsonl or task_results.jsonl found in {run_dir}")
    rows = [attach_fixed_seller_metrics(row) for row in read_jsonl(jsonl_path)]
    output_jsonl = jsonl_path if in_place else jsonl_path.with_name(jsonl_path.stem + ".fixed_seller.jsonl")
    write_jsonl(output_jsonl, rows)

    summary_path = run_dir / "summary.json"
    output_summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        patched = patch_summary(summary, rows)
        output_summary = summary_path if in_place else summary_path.with_name("summary.fixed_seller.json")
        output_summary.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "input_jsonl": str(jsonl_path),
        "output_jsonl": str(output_jsonl),
        "output_summary": str(output_summary) if output_summary else None,
        **summarize_fixed_seller(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing trajectories.jsonl or task_results.jsonl.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite JSONL and summary.json with fixed-seller metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [patch_run_dir(Path(path), in_place=args.in_place) for path in args.run_dirs]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
