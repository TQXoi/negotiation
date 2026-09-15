#!/usr/bin/env python3
"""Oracle-only offline audit for the public issue classifier.

Seller weights are used only after prediction to score identifiability. They
are never included in classifier prompts or cache request payloads.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from AgenticPay_Env.buyer.issue_classifier import (
    EvidenceDecomposedIssueClassifier,
    OntologicalIssueClassifier,
    PublicIssueClassifier,
    UNKNOWN,
)
from experiments.model_clients import make_model_client


def option_weight(weights: Dict[Any, Any], option: Any) -> float:
    for key in (option, str(option), str(option).lower(), json.dumps(option, sort_keys=True)):
        if key in weights:
            return float(weights[key])
    return 0.0


def load_task_configs(path: Path) -> Dict[str, Dict[str, Any]]:
    tasks = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        config = row.get("contract_config") or {}
        if config and row.get("task") not in tasks:
            tasks[row["task"]] = config
    return tasks


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(records)
    covered = [row for row in values if not row["abstained"]]
    return {
        "items": len(values),
        "covered": len(covered),
        "coverage": len(covered) / len(values) if values else None,
        "conditional_accuracy": (
            statistics.fmean(row["correct"] for row in covered) if covered else None
        ),
        "effective_accuracy": (
            statistics.fmean(row["correct"] for row in values) if values else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--consensus-passes", type=int, default=2)
    parser.add_argument(
        "--classifier-mode", choices=("direct", "decomposed", "ontology"), default="direct"
    )
    args = parser.parse_args()

    client = make_model_client(args.model)
    classifier_class = {
        "direct": PublicIssueClassifier,
        "decomposed": EvidenceDecomposedIssueClassifier,
        "ontology": OntologicalIssueClassifier,
    }[args.classifier_mode]
    classifier = classifier_class(
        client,
        min_confidence=args.min_confidence,
        consensus_passes=args.consensus_passes,
        cache_path=args.cache,
    )
    tasks = load_task_configs(args.source_results)
    records = []
    for task, config in sorted(tasks.items()):
        descriptions = config.get("field_descriptions") or {}
        seller = config.get("seller_preferences") or {}
        product_context = (config.get("contrainfo") or {}).get("product_request", "")
        for issue, bounds in (config.get("continuous_bounds") or {}).items():
            weight = float((seller.get("continuous_weights") or {}).get(issue, 0.0))
            gold = "MAX" if weight > 1e-12 else "MIN" if weight < -1e-12 else UNKNOWN
            decision = classifier.classify_continuous(
                issue=issue,
                description=str(descriptions.get(f"continuous_terms.{issue}") or ""),
                bounds=bounds,
                product_context=str(product_context),
            )
            records.append({
                "task": task,
                "kind": "continuous",
                "issue": issue,
                "gold": gold,
                "prediction": decision.label,
                "confidence": decision.confidence,
                "abstained": decision.abstained,
                "reason": decision.reason,
                "correct": decision.label == gold,
                "decision": decision.to_dict(),
            })
        for issue, options in (config.get("discrete_options") or {}).items():
            weights = (seller.get("discrete_weights") or {}).get(issue, {})
            scored = [(option, option_weight(weights, option)) for option in options]
            maximum = max((score for _, score in scored), default=0.0)
            gold = [option for option, value in scored if abs(value - maximum) <= 1e-12]
            decision = classifier.classify_discrete(
                issue=issue,
                description=str(descriptions.get(f"discrete_terms.{issue}") or ""),
                options=options,
                product_context=str(product_context),
            )
            records.append({
                "task": task,
                "kind": "discrete",
                "issue": issue,
                "gold": gold,
                "prediction": decision.label,
                "confidence": decision.confidence,
                "abstained": decision.abstained,
                "reason": decision.reason,
                "correct": any(decision.label == item and type(decision.label) is type(item) for item in gold),
                "decision": decision.to_dict(),
            })

    grouped = defaultdict(list)
    for row in records:
        grouped[row["kind"]].append(row)
    continuous = aggregate(grouped["continuous"])
    discrete = aggregate(grouped["discrete"])
    task15 = [
        row for row in grouped["continuous"]
        if row["task"].startswith("Task15_") and row["issue"] == "wait_time_mins"
    ]
    checks = {
        "all_25_contract_tasks_loaded": len(tasks) == 25,
        "continuous_coverage_ge_0_80": continuous["coverage"] >= 0.80,
        "continuous_conditional_accuracy_ge_0_90": (
            continuous["conditional_accuracy"] is not None
            and continuous["conditional_accuracy"] >= 0.90
        ),
        "task15_wait_correct": bool(task15 and task15[0]["correct"]),
        "discrete_coverage_ge_0_60": discrete["coverage"] >= 0.60,
        "discrete_conditional_accuracy_ge_0_75": (
            discrete["conditional_accuracy"] is not None
            and discrete["conditional_accuracy"] >= 0.75
        ),
    }
    payload = {
        "protocol": {
            "classifier_input": "public semantics/schema only",
            "oracle_use": "post-hoc offline scoring only",
            "model": args.model,
            "min_confidence": args.min_confidence,
            "consensus_passes": args.consensus_passes,
            "classifier_mode": args.classifier_mode,
        },
        "summary": {"continuous": continuous, "discrete": discrete},
        "gate_checks": checks,
        "passed": all(checks.values()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
