#!/usr/bin/env python3
"""Train a rejectable settlement-reserve gate from paired public trajectories.

The treatment label is whether the role-conditioned reserve improves final
BuyerScore without introducing a new score/success mismatch.  Runtime features
are reconstructed exclusively from the control policy's public trace; hidden
seller utility is never a feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "agenticpay_settlement_eligibility_logistic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--run-glob", default="targeted_v53_v54_seed_*/task_results.jsonl")
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260895)
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser.parse_args()


def _buffer_row(record: Dict[str, Any]) -> Dict[str, Any] | None:
    found = None
    for trace in record.get("framework_traces") or []:
        for row in trace.get("ranked_candidates") or []:
            candidate = row.get("candidate") or {}
            if candidate.get("candidate_id") == "classifier_settlement_buffer":
                found = {"trace": trace, "row": row, "candidate": candidate}
    return found


def public_features(record: Dict[str, Any]) -> Dict[str, float] | None:
    found = _buffer_row(record)
    if found is None:
        return None
    trace, row, candidate = found["trace"], found["row"], found["candidate"]
    metadata = candidate.get("metadata") or {}
    diagnostics = row.get("diagnostics") or {}
    risk = float(metadata.get("semantic_departure_risk") or 0.0)
    operational_share = float(metadata.get("operational_issue_share") or 0.0)
    observations = min(4.0, float(metadata.get("public_proposal_observation_count") or 0.0)) / 4.0
    plateau = min(4.0, float(metadata.get("public_proposal_price_plateau_count") or 0.0)) / 4.0
    own_utility = max(0.0, float(candidate.get("own_utility_normalized") or 0.0))
    progress = max(0.0, min(1.0, float(diagnostics.get("progress") or 0.0)))
    return {
        "semantic_risk": risk,
        "operational_share": operational_share,
        "public_observations": observations,
        "price_plateau": plateau,
        "buyer_cushion": min(1.0, own_utility),
        "progress": progress,
        "risk_x_observations": risk * observations,
        "risk_x_cushion": risk * min(1.0, own_utility),
        "operational_x_observations": operational_share * observations,
    }


def dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    lookup: Dict[tuple[str, int, str], Dict[str, Any]] = {}
    for path in sorted(args.runs.glob(args.run_glob)):
        seed = int(path.parent.name.rsplit("_", 1)[-1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            lookup[(record["buyer_variant"], seed, record["task"])] = record
    keys = sorted(
        {(seed, task) for variant, seed, task in lookup if variant == args.control}
        & {(seed, task) for variant, seed, task in lookup if variant == args.treatment}
    )
    rows: List[Dict[str, Any]] = []
    for seed, task in keys:
        control = lookup[(args.control, seed, task)]
        treatment = lookup[(args.treatment, seed, task)]
        features = public_features(control)
        if features is None:
            continue
        delta = float(treatment["buyer_score"]) - float(control["buyer_score"])
        new_mismatch = bool(treatment.get("score_success_mismatch")) and not bool(
            control.get("score_success_mismatch")
        )
        label = int(delta >= 2.0 and not new_mismatch)
        split_bucket = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 4
        rows.append(
            {
                "seed": seed,
                "task": task,
                "features": features,
                "delta": delta,
                "control_mismatch": bool(control.get("score_success_mismatch")),
                "treatment_mismatch": bool(treatment.get("score_success_mismatch")),
                "label": label,
                "split": "dev" if split_bucket == 0 else "train",
            }
        )
    return rows


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def fit(rows: List[Dict[str, Any]], args: argparse.Namespace):
    train = [row for row in rows if row["split"] == "train"]
    names = sorted({name for row in train for name in row["features"]})
    weights = {name: 0.0 for name in names}
    bias = 0.0
    positives = max(1, sum(row["label"] for row in train))
    negatives = max(1, len(train) - positives)
    class_weight = {1: len(train) / (2 * positives), 0: len(train) / (2 * negatives)}
    rng = random.Random(args.seed)
    log = []
    for epoch in range(args.epochs):
        rng.shuffle(train)
        rate = args.learning_rate / math.sqrt(1.0 + epoch / 100.0)
        loss = 0.0
        for row in train:
            label = int(row["label"])
            score = bias + sum(weights[name] * row["features"].get(name, 0.0) for name in names)
            probability = sigmoid(score)
            scale = class_weight[label]
            error = scale * (probability - label)
            bias -= rate * error
            for name in names:
                weights[name] -= rate * (
                    error * row["features"].get(name, 0.0) + args.l2 * weights[name]
                )
            loss += scale * (
                -label * math.log(max(probability, 1e-9))
                - (1 - label) * math.log(max(1 - probability, 1e-9))
            )
        if epoch % 100 == 0 or epoch + 1 == args.epochs:
            log.append({"epoch": epoch, "weighted_train_log_loss": loss / len(train)})
    return bias, weights, log


def metrics(rows: Iterable[Dict[str, Any]], bias: float, weights: Dict[str, float]):
    output = []
    for row in rows:
        score = bias + sum(weights.get(name, 0.0) * value for name, value in row["features"].items())
        probability = sigmoid(score)
        decision = "allow" if probability >= 0.8 else "deny" if probability <= 0.2 else "abstain"
        output.append({**row, "probability": probability, "decision": decision})
    allowed = [row for row in output if row["decision"] == "allow"]
    return {
        "n": len(output),
        "positive_n": sum(row["label"] for row in output),
        "allow_n": len(allowed),
        "allow_precision": statistics.fmean(row["label"] for row in allowed) if allowed else None,
        "allow_mean_delta": statistics.fmean(row["delta"] for row in allowed) if allowed else None,
        "allow_new_mismatch_n": sum(
            row["treatment_mismatch"] and not row["control_mismatch"] for row in allowed
        ),
        "rows": output,
    }


def main() -> None:
    args = parse_args()
    rows = dataset(args)
    if not rows or len({row["label"] for row in rows}) != 2:
        raise RuntimeError("Settlement gate requires paired positive and negative labels")
    bias, weights, log = fit(rows, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "feature_source": "control_public_framework_trace_only",
        "label_source": "paired_buyer_score_delta_ge_2_without_new_mismatch",
        "allow_threshold": 0.8,
        "deny_threshold": 0.2,
        "bias": bias,
        "weights": weights,
    }
    (args.output_dir / "settlement_gate_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "settlement_gate_dataset.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "rows": len(rows),
        "train": metrics((row for row in rows if row["split"] == "train"), bias, weights),
        "dev": metrics((row for row in rows if row["split"] == "dev"), bias, weights),
        "train_log": log,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
