#!/usr/bin/env python3
"""Fit and evaluate a response-likelihood calibrator on verified events.

The model conditions on evaluator-only reservation labels during training, but
deployed posterior updates receive only public event outcomes and offers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from scipy.optimize import minimize


OUTCOMES = ("accept", "counter", "quit")
POLICIES = (
    ("patient", 0.92, 0.02, 0.20),
    ("neutral", 0.78, 0.08, 0.25),
    ("firm", 0.48, 0.28, 0.75),
)
READINESS = (0.25, 0.50, 0.75, 1.00)
RESERVATION_GRID = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class CalibrationParams:
    likelihood_tau: float = 0.05
    readiness_progress_power: float = 1.0
    readiness_terminal_fraction: float = 1.0
    quit_progress_weight: float = 0.32
    quit_finality_weight: float = 0.22
    counter_progress_decay: float = 0.35

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "CalibrationParams":
        return cls(*map(float, values))

    def vector(self) -> list[float]:
        return list(asdict(self).values())


BASELINE = CalibrationParams()
BOUNDS = (
    (0.01, 0.30),
    (0.25, 8.0),
    (0.0, 1.0),
    (0.0, 0.90),
    (0.0, 0.90),
    (0.0, 0.95),
)


def response_distribution(
    offer: float,
    reservation: float,
    progress: float,
    params: CalibrationParams,
    policy: tuple[str, float, float, float],
    readiness: float,
) -> dict[str, float]:
    _, counter_propensity, quit_bias, finality = policy
    accept = sigmoid((offer - reservation) / max(params.likelihood_tau, 1e-6))
    progress_credit = (
        params.readiness_terminal_fraction
        * clamp(progress, 0.0, 1.0) ** params.readiness_progress_power
    )
    effective_readiness = clamp(
        readiness + (1.0 - readiness) * progress_credit, 0.0, 1.0
    )
    accept *= effective_readiness
    remaining = 1.0 - accept
    quit_share = clamp(
        quit_bias
        + params.quit_progress_weight * progress
        + params.quit_finality_weight * finality,
        0.01,
        0.95,
    )
    counter_share = clamp(
        counter_propensity * (1.0 - params.counter_progress_decay * progress),
        0.01,
        max(0.01, 0.99 - quit_share),
    )
    raw = {
        "accept": accept,
        "counter": remaining * counter_share,
        "quit": remaining * (1.0 - counter_share),
    }
    total = sum(raw.values())
    return {key: max(1e-8, value / total) for key, value in raw.items()}


def marginal_distribution(
    offer: float,
    reservation: float,
    progress: float,
    params: CalibrationParams,
) -> dict[str, float]:
    result = {key: 0.0 for key in OUTCOMES}
    count = len(POLICIES) * len(READINESS)
    for policy in POLICIES:
        for readiness in READINESS:
            local = response_distribution(
                offer, reservation, progress, params, policy, readiness
            )
            for outcome in OUTCOMES:
                result[outcome] += local[outcome] / count
    return result


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_fields(row: dict[str, Any]) -> tuple[float, float, float, str]:
    event = row["event"]
    return (
        float(event["tested_compatibility"]),
        float(row["labels"]["reservation_ratio"]),
        float(event["progress"]),
        str(event["outcome"]),
    )


def mean_nll(rows: Sequence[dict[str, Any]], params: CalibrationParams) -> float:
    losses = []
    for row in rows:
        offer, reservation, progress, outcome = event_fields(row)
        probability = marginal_distribution(offer, reservation, progress, params)[outcome]
        losses.append(-math.log(max(1e-8, probability)))
    return statistics.mean(losses) if losses else math.inf


def fit(rows: Sequence[dict[str, Any]], seed: int) -> tuple[CalibrationParams, dict[str, Any]]:
    rng = random.Random(seed)

    def objective(values: Sequence[float]) -> float:
        return mean_nll(rows, CalibrationParams.from_vector(values))

    starts = [BASELINE.vector()]
    for _ in range(11):
        starts.append([rng.uniform(low, high) for low, high in BOUNDS])
    attempts = []
    for values in starts:
        result = minimize(objective, values, method="L-BFGS-B", bounds=BOUNDS)
        attempts.append({
            "success": bool(result.success),
            "fun": float(result.fun),
            "message": str(result.message),
            "x": list(map(float, result.x)),
        })
    best = min(attempts, key=lambda item: item["fun"])
    return CalibrationParams.from_vector(best["x"]), {
        "starts": len(starts),
        "best_train_nll": best["fun"],
        "successful_starts": sum(item["success"] for item in attempts),
        "attempts": attempts,
    }


def ece(points: Sequence[tuple[float, int]], bins: int = 10) -> float:
    if not points:
        return math.nan
    total = len(points)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        local = [p for p in points if low <= p[0] < high or (index == bins - 1 and p[0] == 1.0)]
        if local:
            result += len(local) / total * abs(
                statistics.mean(p for p, _ in local)
                - statistics.mean(y for _, y in local)
            )
    return result


def response_metrics(rows: Sequence[dict[str, Any]], params: CalibrationParams) -> dict[str, Any]:
    nll = []
    brier = []
    accept_points = []
    per_scenario: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        offer, reservation, progress, outcome = event_fields(row)
        probabilities = marginal_distribution(offer, reservation, progress, params)
        loss = -math.log(max(1e-8, probabilities[outcome]))
        nll.append(loss)
        per_scenario[row["scenario_id"]].append(loss)
        brier.append(sum(
            (probabilities[key] - float(outcome == key)) ** 2 for key in OUTCOMES
        ))
        accept_points.append((probabilities["accept"], int(outcome == "accept")))
    return {
        "events": len(rows),
        "scenarios": len(per_scenario),
        "nll": statistics.mean(nll),
        "multiclass_brier": statistics.mean(brier),
        "accept_ece_10bin": ece(accept_points),
        "scenario_nll": {key: statistics.mean(values) for key, values in per_scenario.items()},
    }


def profile_rows() -> list[tuple[float, tuple[str, float, float, float], float]]:
    return [(reservation, policy, readiness) for reservation in RESERVATION_GRID for policy in POLICIES for readiness in READINESS]


def posterior_for_session(
    rows: Sequence[dict[str, Any]],
    params: CalibrationParams,
    *,
    likelihood_power: float,
    hazard: float = 0.02,
) -> dict[str, Any]:
    profiles = profile_rows()
    weights = [1.0 / len(profiles)] * len(profiles)
    for row in sorted(rows, key=lambda item: (item["event"]["turn"], item["transcript_index"])):
        offer, _, progress, outcome = event_fields(row)
        prior = [(1.0 - hazard) * weight + hazard / len(profiles) for weight in weights]
        likelihood = [
            response_distribution(offer, reservation, progress, params, policy, readiness)[outcome]
            ** likelihood_power
            for reservation, policy, readiness in profiles
        ]
        values = [p * max(1e-8, l) for p, l in zip(prior, likelihood)]
        total = sum(values)
        weights = [value / total for value in values]
    reservation_mass = {reservation: 0.0 for reservation in RESERVATION_GRID}
    for (reservation, _, _), weight in zip(profiles, weights):
        reservation_mass[reservation] += weight
    mean = sum(r * weight for r, weight in reservation_mass.items())

    def quantile(level: float) -> float:
        total = 0.0
        for reservation in sorted(reservation_mass):
            total += reservation_mass[reservation]
            if total >= level:
                return reservation
        return max(reservation_mass)

    truth = float(rows[0]["labels"]["reservation_ratio"])
    nearest = min(RESERVATION_GRID, key=lambda value: abs(value - truth))
    return {
        "scenario_id": rows[0]["scenario_id"],
        "session_id": rows[0]["event"]["session_id"],
        "truth": truth,
        "mean": mean,
        "q10": quantile(0.10),
        "q90": quantile(0.90),
        "abs_error": abs(mean - truth),
        "coverage": float(quantile(0.10) <= truth <= quantile(0.90)),
        "nearest_truth_probability": reservation_mass[nearest],
        "events": len(rows),
    }


def belief_metrics(
    rows: Sequence[dict[str, Any]],
    params: CalibrationParams,
    likelihood_power: float,
) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[row["event"]["session_id"]].append(row)
    posterior = [
        posterior_for_session(local, params, likelihood_power=likelihood_power)
        for local in sessions.values()
    ]
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row in posterior:
        by_scenario[row["scenario_id"]].append(row["abs_error"])
    return {
        "sessions": len(posterior),
        "scenarios": len(by_scenario),
        "likelihood_power": likelihood_power,
        "reservation_mae": statistics.mean(row["abs_error"] for row in posterior),
        "q10_q90_coverage": statistics.mean(row["coverage"] for row in posterior),
        "mean_nearest_truth_probability": statistics.mean(row["nearest_truth_probability"] for row in posterior),
        "scenario_mae": {key: statistics.mean(values) for key, values in by_scenario.items()},
        "posterior": posterior,
    }


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def paired_scenario_improvement(
    baseline: dict[str, float], calibrated: dict[str, float], samples: int, seed: int
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(calibrated))
    deltas = [baseline[key] - calibrated[key] for key in keys]
    return {
        "estimand": "baseline_loss_minus_calibrated_loss; positive favors calibrated",
        "scenario_clusters": len(keys),
        "mean_improvement": statistics.mean(deltas),
        "cluster_bootstrap_95_ci": bootstrap_ci(deltas, samples, seed),
        "scenario_deltas": dict(zip(keys, deltas)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    args = parser.parse_args()
    train = load(args.data_dir / "train.jsonl")
    calibration = load(args.data_dir / "calibration.jsonl")
    test = load(args.data_dir / "test.jsonl")

    learned, fit_diagnostics = fit(train, args.seed)
    powers = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
    calibration_power_metrics = {
        str(power): belief_metrics(calibration, learned, power) for power in powers
    }
    selected_power = min(
        powers,
        key=lambda power: (
            calibration_power_metrics[str(power)]["reservation_mae"],
            -calibration_power_metrics[str(power)]["q10_q90_coverage"],
        ),
    )

    baseline_response = response_metrics(test, BASELINE)
    learned_response = response_metrics(test, learned)
    baseline_belief = belief_metrics(test, BASELINE, 1.0)
    learned_belief = belief_metrics(test, learned, selected_power)
    comparisons = {
        "test_response_nll": paired_scenario_improvement(
            baseline_response["scenario_nll"], learned_response["scenario_nll"],
            args.bootstrap_samples, args.seed,
        ),
        "test_reservation_mae": paired_scenario_improvement(
            baseline_belief["scenario_mae"], learned_belief["scenario_mae"],
            args.bootstrap_samples, args.seed + 1,
        ),
    }
    gate = {
        "response_nll_mean_improved": comparisons["test_response_nll"]["mean_improvement"] > 0,
        "response_brier_improved": learned_response["multiclass_brier"] < baseline_response["multiclass_brier"],
        "reservation_mae_mean_improved": comparisons["test_reservation_mae"]["mean_improvement"] > 0,
        "coverage_not_below_0_75": learned_belief["q10_q90_coverage"] >= 0.75,
        "at_least_one_cluster_ci_stably_improved": (
            comparisons["test_response_nll"]["cluster_bootstrap_95_ci"][0] > 0
            or comparisons["test_reservation_mae"]["cluster_bootstrap_95_ci"][0] > 0
        ),
    }
    gate["planner_training_allowed"] = all(gate.values())
    output = {
        "data_dir": str(args.data_dir.resolve()),
        "baseline_params": asdict(BASELINE),
        "learned_params": asdict(learned),
        "selected_likelihood_power": selected_power,
        "fit": fit_diagnostics,
        "train_response": {
            "baseline": response_metrics(train, BASELINE),
            "calibrated": response_metrics(train, learned),
        },
        "calibration_response": {
            "baseline": response_metrics(calibration, BASELINE),
            "calibrated": response_metrics(calibration, learned),
        },
        "calibration_power_selection": calibration_power_metrics,
        "test_response": {"baseline": baseline_response, "calibrated": learned_response},
        "test_belief": {"baseline": baseline_belief, "calibrated": learned_belief},
        "paired_cluster_comparisons": comparisons,
        "gate": gate,
        "limitations": [
            "Only 24 development scenarios were available; test inference has four scenario clusters.",
            "Labels use evaluator-side seller cost only during calibration/evaluation, never deployed input.",
            "Public LLM claims are not used as utility truth.",
            "This checkpoint must not be selected on validation40 or final64.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibrated_response_params.json").write_text(
        json.dumps({
            "schema_version": "verified_response_calibration_v1",
            "params": asdict(learned),
            "likelihood_power": selected_power,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibration_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "learned_params": asdict(learned),
        "selected_likelihood_power": selected_power,
        "test_response": output["test_response"],
        "test_belief": {
            key: {k: v for k, v in value.items() if k not in {"posterior", "scenario_mae"}}
            for key, value in output["test_belief"].items()
        },
        "paired_cluster_comparisons": comparisons,
        "gate": gate,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
