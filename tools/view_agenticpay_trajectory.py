#!/usr/bin/env python3
"""Inspect or export complete AgenticPay trajectories from experiment JSONL.

The tool is intentionally read-only and dependency-free. It understands the
``task_results.jsonl`` records produced by ``AgenticPay_Env.eval`` and the
single28 framework runner. It never recomputes or modifies benchmark scores.

Examples
--------
List all available episodes under an iteration::

    python tools/view_agenticpay_trajectory.py --input RUN_DIR --list

Export V30 Task15, including framework selection traces::

    python tools/view_agenticpay_trajectory.py \
      --input RUN_DIR --task Task15 --variant v30 \
      --include-framework-trace --output task15_v30.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def discover_jsonl(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    preferred = sorted(input_path.rglob("task_results.jsonl"))
    if preferred:
        return preferred
    return sorted(input_path.rglob("*.jsonl"))


def load_records(paths: Sequence[Path]) -> List[Tuple[Path, int, Dict[str, Any]]]:
    records: List[Tuple[Path, int, Dict[str, Any]]] = []
    for path in paths:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and (row.get("task") or row.get("rounds")):
                records.append((path, line_number, row))
    return records


def matches(value: Any, query: str | None) -> bool:
    return query is None or query.lower() in str(value or "").lower()


def score(row: Dict[str, Any]) -> Any:
    fixed = row.get("buyer_score_fixed_seller")
    return row.get("buyer_score") if fixed is None else fixed


def one_line(value: Any, width: int = 68) -> str:
    # Preserve meaningful falsy values (False and 0).  Treat only a missing
    # value as blank so protocol failures are not confused with absent data.
    text = "" if value is None else re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def list_records(records: Sequence[Tuple[Path, int, Dict[str, Any]]]) -> str:
    lines = [
        "idx\ttask\tvariant\tscore\tdeal\tmismatch\ttermination\trounds\tsource"
    ]
    for index, (path, line_number, row) in enumerate(records):
        lines.append(
            "\t".join(
                [
                    str(index),
                    str(row.get("task") or row.get("task_path") or "?"),
                    str(row.get("buyer_variant") or "?"),
                    str(score(row)),
                    str(bool(row.get("success"))),
                    str(bool(row.get("score_success_mismatch"))),
                    str(row.get("termination_reason") or row.get("status") or "?"),
                    str(row.get("total_rounds") or len(row.get("rounds") or [])),
                    f"{path}:{line_number}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"


def extract_contract(text: Any) -> Dict[str, Any] | None:
    match = re.search(
        r"<contract>\s*(\{.*?\})\s*</contract>",
        str(text or ""),
        flags=re.I | re.S,
    )
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    try:
        price = float(value["price"])
    except (KeyError, TypeError, ValueError):
        return None
    continuous = value.get("continuous_terms", {})
    discrete = value.get("discrete_terms", {})
    if not isinstance(continuous, dict) or not isinstance(discrete, dict):
        return None
    return {
        "price": price,
        "continuous_terms": continuous,
        "discrete_terms": discrete,
    }


def contract_schema_valid(contract: Dict[str, Any], config: Dict[str, Any]) -> bool:
    continuous = contract.get("continuous_terms", {})
    discrete = contract.get("discrete_terms", {})
    bounds_map = config.get("continuous_bounds", {}) or {}
    options_map = config.get("discrete_options", {}) or {}
    if set(continuous) != set(bounds_map) or set(discrete) != set(options_map):
        return False
    for field, bounds in bounds_map.items():
        try:
            value = float(continuous[field])
            if value < float(bounds.get("min", value)) or value > float(
                bounds.get("max", value)
            ):
                return False
        except (TypeError, ValueError):
            return False
    return all(discrete.get(field) in options for field, options in options_map.items())


def reconstruct_agreed_contract(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Reconstruct omitted summary fields using unchanged upstream semantics.

    This is an explicitly labelled post-hoc audit.  It never changes the raw
    record or score.  Some AgenticPay example scripts save prices and status but
    omit ``env.state.metadata['agreed_contract']`` from summary.json.
    """

    if isinstance(row.get("agreed_contract"), dict):
        return {
            "source": "recorded_agreed_contract",
            "contract": row["agreed_contract"],
        }
    config = row.get("contract_config") or {}
    if not config or not row.get("success"):
        return None
    buyer_contract: Dict[str, Any] | None = None
    seller_contract: Dict[str, Any] | None = None
    tolerance = float(row.get("price_tolerance") or 0.0)
    for round_row in row.get("rounds") or []:
        buyer = round_row.get("buyer_contract_extracted") or extract_contract(
            round_row.get("buyer_action")
        )
        seller = round_row.get("seller_contract_extracted") or extract_contract(
            round_row.get("seller_action")
        )
        if isinstance(buyer, dict) and contract_schema_valid(buyer, config):
            buyer_contract = buyer
        if isinstance(seller, dict) and contract_schema_valid(seller, config):
            seller_contract = seller
        if not buyer_contract or not seller_contract:
            continue
        same_terms = (
            buyer_contract.get("continuous_terms", {})
            == seller_contract.get("continuous_terms", {})
            and buyer_contract.get("discrete_terms", {})
            == seller_contract.get("discrete_terms", {})
        )
        buyer_price = float(buyer_contract["price"])
        seller_price = float(seller_contract["price"])
        compatible_price = (
            abs(buyer_price - seller_price) <= tolerance
            or seller_price <= buyer_price
        )
        if same_terms and compatible_price:
            agreed_price = (
                seller_price
                if seller_price <= buyer_price
                else (seller_price + buyer_price) / 2.0
            )
            return {
                "source": "reconstructed_from_rounds_using_upstream_contract_compatibility",
                "round": round_row.get("round"),
                "buyer_contract": buyer_contract,
                "seller_contract": seller_contract,
                "contract": {
                    "price": agreed_price,
                    "continuous_terms": buyer_contract.get("continuous_terms", {}),
                    "discrete_terms": buyer_contract.get("discrete_terms", {}),
                },
            }
    return None


def contract_utility_audit(
    contract: Dict[str, Any] | None, config: Dict[str, Any]
) -> Dict[str, Any] | None:
    if not contract or not config:
        return None
    buyer = config.get("buyer_preferences", {}) or {}
    seller = config.get("seller_preferences", {}) or {}
    price = float(contract["price"])
    buyer_utility = float(buyer.get("v_base", 0.0)) - price
    seller_utility = price - float(seller.get("c_base", 0.0))
    for field, raw in contract.get("continuous_terms", {}).items():
        value = float(raw)
        buyer_utility += float((buyer.get("continuous_weights", {}) or {}).get(field, 0.0)) * value
        seller_utility += float((seller.get("continuous_weights", {}) or {}).get(field, 0.0)) * value
    for field, value in contract.get("discrete_terms", {}).items():
        buyer_utility += _option_weight_for_audit(
            ((buyer.get("discrete_weights", {}) or {}).get(field, {}) or {}),
            value,
        )
        seller_utility += _option_weight_for_audit(
            ((seller.get("discrete_weights", {}) or {}).get(field, {}) or {}),
            value,
        )
    return {
        "buyer_utility": buyer_utility,
        "seller_utility": seller_utility,
        "buyer_ir": buyer_utility >= 0.0,
        "seller_ir": seller_utility >= 0.0,
        "failure_reason": (
            "negative_buyer_utility"
            if buyer_utility < 0.0
            else "negative_seller_utility"
            if seller_utility < 0.0
            else None
        ),
    }


def _option_weight_for_audit(weights: Dict[Any, Any], value: Any) -> float:
    for key in (value, str(value), str(value).lower(), json.dumps(value)):
        if key in weights:
            return float(weights[key])
    return 0.0


def metric_table(row: Dict[str, Any]) -> str:
    pairs = [
        ("Status", row.get("status")),
        ("Success / Deal", bool(row.get("success"))),
        ("Termination", row.get("termination_reason")),
        ("Rounds", row.get("total_rounds") or len(row.get("rounds") or [])),
        ("BuyerScore (native)", row.get("buyer_score_original", row.get("buyer_score"))),
        ("BuyerScore (fixed-seller if available)", row.get("buyer_score_fixed_seller")),
        ("GlobalScore", row.get("global_score")),
        ("Buyer reward", row.get("buyer_reward")),
        ("Seller reward", row.get("seller_reward")),
        ("Score-success mismatch", bool(row.get("score_success_mismatch"))),
        ("Contract failure", row.get("contract_failure_reason")),
        ("Buyer utility", row.get("contract_buyer_utility")),
        ("Seller utility (post-hoc audit)", row.get("contract_seller_utility")),
        ("Agreed price", row.get("agreed_price")),
        ("Buyer max price", row.get("buyer_max_price")),
        ("Seller min price", row.get("seller_min_price")),
        ("Elapsed seconds", row.get("elapsed_time", row.get("wrapper_elapsed_time"))),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {one_line(value, 120)} |" for name, value in pairs)
    return "\n".join(lines)


def trace_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    validation = trace.get("validation") or {}
    selector = validation.get("selection_validator") or {}
    return {
        "turn": trace.get("turn") or (trace.get("state") or {}).get("turn"),
        "selected_candidate_id": trace.get("selected_candidate_id"),
        "selection_validator_reason": selector.get("reason"),
        "selection_validator_type": selector.get("validator_type"),
        "overrode_selection": selector.get("overrode_selection"),
        "changed_fields": selector.get("changed_fields"),
        "rejected_fields": selector.get("rejected_fields"),
        "rendered_action": trace.get("rendered_action"),
    }


def markdown_episode(
    source: Path,
    line_number: int,
    row: Dict[str, Any],
    *,
    include_private_utility: bool,
    include_framework_trace: bool,
    full_framework_trace: bool,
) -> str:
    task = row.get("task") or row.get("task_path") or "unknown task"
    variant = row.get("buyer_variant") or "unknown variant"
    reconstructed = reconstruct_agreed_contract(row)
    reconstructed_contract = (
        reconstructed.get("contract") if isinstance(reconstructed, dict) else None
    )
    reconstructed_utility = contract_utility_audit(
        reconstructed_contract,
        row.get("contract_config") or {},
    )
    output: List[str] = [
        f"# AgenticPay trajectory: {task}",
        "",
        f"- Variant: `{variant}`",
        f"- Seller variant: `{row.get('seller_variant')}`",
        f"- Model: `{row.get('model_alias') or row.get('model')}`",
        f"- Source: `{source}:{line_number}`",
        "",
        "## 1. Episode result",
        "",
        metric_table(row),
        "",
        "### Agreed contract",
        "",
        json_block(row.get("agreed_contract")),
        "",
    ]
    if reconstructed and reconstructed.get("source") != "recorded_agreed_contract":
        output.extend(
            [
                "### Reconstructed contract audit (raw record unchanged)",
                "",
                "> The upstream example summary omitted `agreed_contract`. This block is reconstructed from the saved round texts using the unchanged AgenticPay contract-compatibility rule.",
                "",
                json_block(
                    {
                        **reconstructed,
                        "utility_audit": reconstructed_utility,
                    }
                ),
                "",
            ]
        )

    output.extend(["## 2. Environment / task construct", ""])
    environment = {
        "task": task,
        "task_path": row.get("task_path"),
        "category": row.get("category"),
        "scenario": row.get("scenario"),
        "user_requirement": row.get("user_requirement"),
        "user_profile": row.get("user_profile"),
        "product_info": row.get("product_info"),
        "contract_mode": row.get("contract_mode"),
        "price_tolerance": row.get("price_tolerance"),
        "max_rounds": row.get("max_rounds"),
        "gamma": row.get("gamma"),
    }
    output.extend([json_block(environment), ""])

    config = row.get("contract_config") or {}
    public_contract = {
        "contrainfo": config.get("contrainfo"),
        "field_descriptions": config.get("field_descriptions"),
        "continuous_bounds": config.get("continuous_bounds"),
        "discrete_options": config.get("discrete_options"),
    }
    output.extend(["### Public contract schema", "", json_block(public_contract), ""])
    if include_private_utility:
        output.extend(
            [
                "### Private utility configuration (post-hoc inspection only)",
                "",
                "> These fields are benchmark ground truth. A deployed buyer framework must not read seller preferences.",
                "",
                json_block(
                    {
                        "buyer_preferences": config.get("buyer_preferences"),
                        "seller_preferences": config.get("seller_preferences"),
                    }
                ),
                "",
            ]
        )

    output.extend(["## 3. Complete public interaction", ""])
    rounds = row.get("rounds") or []
    if not rounds:
        output.extend(["_No captured rounds in this record._", ""])
    for index, round_row in enumerate(rounds, 1):
        round_id = round_row.get("round", index)
        output.extend(
            [
                f"### Round {round_id}",
                "",
                "**Buyer**",
                "",
                "```text",
                str(round_row.get("buyer_action") or ""),
                "```",
                "",
                "**Seller**",
                "",
                "```text",
                str(round_row.get("seller_action") or ""),
                "```",
                "",
                "**Round state / prices**",
                "",
                json_block(
                    {
                        "buyer_price_extracted": round_row.get("buyer_price_extracted"),
                        "seller_price_extracted": round_row.get("seller_price_extracted"),
                        "buyer_price_state": round_row.get("buyer_price_state"),
                        "seller_price_state": round_row.get("seller_price_state"),
                        "buyer_format_valid": round_row.get("buyer_format_valid"),
                        "seller_format_valid": round_row.get("seller_format_valid"),
                        "status": round_row.get("status"),
                        "termination_reason": round_row.get("termination_reason"),
                        "reward": round_row.get("reward"),
                        "step_buyer_reward": round_row.get("step_buyer_reward"),
                        "step_seller_reward": round_row.get("step_seller_reward"),
                    }
                ),
                "",
            ]
        )

    output.extend(["## 4. Post-hoc scoring diagnostics", ""])
    output.extend(
        [
            json_block(
                {
                    "contract_score_feasible": row.get("contract_score_feasible"),
                    "contract_ir_violation": row.get("contract_ir_violation"),
                    "contract_failure_reason": row.get("contract_failure_reason"),
                    "contract_buyer_utility": row.get("contract_buyer_utility"),
                    "contract_seller_utility": row.get("contract_seller_utility"),
                    "contract_z_max": row.get("contract_z_max"),
                    "contract_quality_q": row.get("contract_quality_q"),
                    "score_success_mismatch": row.get("score_success_mismatch"),
                    "fixed_seller_eval_reason": row.get("fixed_seller_eval_reason"),
                    "fixed_seller_best_feasible_buyer_offer": row.get(
                        "fixed_seller_best_feasible_buyer_offer"
                    ),
                    "fixed_seller_num_feasible_buyer_offers": row.get(
                        "fixed_seller_num_feasible_buyer_offers"
                    ),
                    "reconstructed_contract_audit": reconstructed,
                    "reconstructed_utility_audit": reconstructed_utility,
                }
            ),
            "",
        ]
    )

    if include_framework_trace:
        traces = row.get("framework_traces") or row.get("framework_trace") or []
        if isinstance(traces, dict):
            traces = [traces]
        output.extend(["## 5. Framework decision trace", ""])
        if full_framework_trace:
            output.extend([json_block(traces), ""])
        else:
            output.extend([json_block([trace_summary(t) for t in traces]), ""])
    return "\n".join(output).rstrip() + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True, help="JSONL file or directory searched recursively")
    result.add_argument("--list", action="store_true", help="list matching episodes instead of rendering one")
    result.add_argument("--task", help="case-insensitive substring matched against task/task_path")
    result.add_argument("--variant", help="case-insensitive substring matched against buyer_variant")
    result.add_argument("--index", type=int, default=0, help="select index among filtered records (default: 0)")
    result.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    result.add_argument("--hide-private-utility", action="store_true", help="omit buyer/seller benchmark utility truth")
    result.add_argument("--include-framework-trace", action="store_true", help="append planner/validator trace summary")
    result.add_argument("--full-framework-trace", action="store_true", help="append full ranked-candidate traces (large)")
    return result


def main() -> int:
    args = parser().parse_args()
    paths = discover_jsonl(args.input.resolve())
    records = load_records(paths)
    records = [
        item
        for item in records
        if matches(item[2].get("task") or item[2].get("task_path"), args.task)
        and matches(item[2].get("buyer_variant"), args.variant)
    ]
    if args.list:
        sys.stdout.write(list_records(records))
        return 0
    if not records:
        print("No matching AgenticPay trajectory found. Use --list to inspect available records.", file=sys.stderr)
        return 2
    if args.index < 0 or args.index >= len(records):
        print(f"--index must be in [0, {len(records) - 1}]", file=sys.stderr)
        return 2
    source, line_number, row = records[args.index]
    rendered = markdown_episode(
        source,
        line_number,
        row,
        include_private_utility=not args.hide_private_utility,
        include_framework_trace=args.include_framework_trace or args.full_framework_trace,
        full_framework_trace=args.full_framework_trace,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output.resolve())
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
