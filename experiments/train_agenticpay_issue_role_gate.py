#!/usr/bin/env python3
"""Train a rejectable public issue-role gate from pre-final AgenticPay schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from AgenticPay_Env.buyer.issue_classifier import LearnedIssueRoleGate


SCHEMA_VERSION = "agenticpay_issue_role_logistic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.002)
    return parser.parse_args()


def weak_role_label(issue: str, description: str, options: List[Any]) -> int:
    """Label factual match assessments, not operational obligations.

    This rule is used only to build offline supervision. Runtime receives the
    trained checkpoint and public schema text, never this rule or private
    utilities. Requiring both field semantics and the graded match domain
    prevents generic guarantee/confirmation fields from being mislabeled.
    """

    normalized = f"{issue} {description}".lower()
    option_names = {str(value).lower() for value in options}
    graded_match_domain = {
        "strong_match",
        "partial_match",
        "mismatch_or_uncertain",
    }.issubset(option_names)
    factual_wording = (
        issue == "user_product_preference"
        or "how well" in normalized
        or "listing photo" in normalized
        or "map aligns" in normalized
    )
    return int(graded_match_domain and factual_wording)


def schema_rows(path: Path) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("contract_config") or {}
        descriptions = config.get("field_descriptions") or {}
        for issue, options in (config.get("discrete_options") or {}).items():
            description = str(descriptions.get(f"discrete_terms.{issue}") or "")
            key = hashlib.sha256(
                json.dumps(
                    [issue, description, options],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            unique[key] = {
                "schema_id": key,
                "issue": str(issue),
                "description": description,
                "options": list(options),
                "label": weak_role_label(str(issue), description, list(options)),
            }
    rows = sorted(unique.values(), key=lambda row: row["schema_id"])
    for label in (0, 1):
        members = [row for row in rows if row["label"] == label]
        for index, row in enumerate(members):
            row["split"] = "dev" if index % 5 == 0 else "train"
    return rows


def vector(row: Dict[str, Any]) -> Dict[str, float]:
    return LearnedIssueRoleGate.features(
        row["issue"], row["description"], row["options"]
    )


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def fit(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    train = [row for row in rows if row["split"] == "train"]
    vocab = sorted({name for row in train for name in vector(row)})
    weights = {name: 0.0 for name in vocab}
    bias = 0.0
    log: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        rng.shuffle(train)
        rate = learning_rate / math.sqrt(1.0 + epoch / 100.0)
        total_loss = 0.0
        for row in train:
            features = vector(row)
            score = bias + sum(weights.get(name, 0.0) * value for name, value in features.items())
            probability = sigmoid(score)
            label = float(row["label"])
            error = probability - label
            bias -= rate * error
            for name, value in features.items():
                if name in weights:
                    weights[name] -= rate * (error * value + l2 * weights[name])
            total_loss += -label * math.log(max(probability, 1e-9)) - (1.0 - label) * math.log(
                max(1.0 - probability, 1e-9)
            )
        if epoch % 100 == 0 or epoch + 1 == epochs:
            log.append({"epoch": epoch, "train_log_loss": total_loss / len(train)})
    return bias, weights, log


def metrics(rows: Iterable[Dict[str, Any]], bias: float, weights: Dict[str, float]) -> Dict[str, Any]:
    rows = list(rows)
    predictions = []
    for row in rows:
        score = bias + sum(weights.get(name, 0.0) * value for name, value in vector(row).items())
        probability = sigmoid(score)
        predicted = 1 if probability >= 0.8 else 0 if probability <= 0.2 else None
        predictions.append((row["label"], predicted, probability))
    covered = [(gold, pred) for gold, pred, _ in predictions if pred is not None]
    return {
        "n": len(rows),
        "descriptive_n": sum(row["label"] for row in rows),
        "coverage": len(covered) / len(rows) if rows else None,
        "covered_accuracy": (
            sum(gold == pred for gold, pred in covered) / len(covered) if covered else None
        ),
        "reject_n": len(rows) - len(covered),
        "probabilities": [
            {"schema_id": row["schema_id"], "gold": gold, "predicted": pred, "p": probability}
            for row, (gold, pred, probability) in zip(rows, predictions)
        ],
    }


def main() -> None:
    args = parse_args()
    rows = schema_rows(args.input_jsonl)
    if len({row["label"] for row in rows}) != 2:
        raise RuntimeError("Issue-role training requires both classes")
    bias, weights, train_log = fit(
        rows,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
        "label_source": "public_schema_weak_supervision_v1",
        "all28_final_seed_excluded": 20260857,
        "descriptive_threshold": 0.8,
        "operational_threshold": 0.2,
        "bias": bias,
        "weights": weights,
    }
    (args.output_dir / "issue_role_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "schema_dataset.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "train": metrics((row for row in rows if row["split"] == "train"), bias, weights),
        "dev": metrics((row for row in rows if row["split"] == "dev"), bias, weights),
        "train_log": train_log,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
