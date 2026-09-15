from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from anl2025.negotiator import Boulware2025, Conceder2025, Linear2025
from anl2025.runner import run_session
from anl2025.scenario import MultidealScenario, make_multideal_scenario

from .agent import BeliefGlobalPlannerANL2025Center


BASELINES: dict[str, type] = {
    "boulware2025": Boulware2025,
    "linear2025": Linear2025,
    "conceder2025": Conceder2025,
}

FRAMEWORK_VARIANTS = {
    "framework_static_belief_global_planner",
    "framework_continuous_belief_local_planner",
    "framework_continuous_belief_global_planner",
    "framework_no_information_gain",
    "framework_oracle_partner_belief",
    "framework_shuffled_partner_belief",
}

ORACLE_VARIANTS = {"framework_oracle_partner_belief"}
SHUFFLED_VARIANTS = {"framework_shuffled_partner_belief"}
DEFAULT_VARIANTS = (
    "boulware2025",
    "linear2025",
    "conceder2025",
    "framework_static_belief_global_planner",
    "framework_continuous_belief_local_planner",
    "framework_continuous_belief_global_planner",
    "framework_no_information_gain",
    "framework_oracle_partner_belief",
    "framework_shuffled_partner_belief",
)
DEFAULT_EDGE_TYPES = (Boulware2025, Linear2025, Conceder2025)


class _PermutedOutcomeUFun:
    """Evaluator-only wrong preference ordering on the *same* outcome space."""

    def __init__(self, base: Any, outcomes: tuple[Any, ...]) -> None:
        if not outcomes:
            raise ValueError("Cannot permute an empty outcome space")
        self.base = copy.deepcopy(base)
        self.outcomes = outcomes
        shifted = outcomes[1:] + outcomes[:1]
        self._mapping = {
            tuple(source): target for source, target in zip(outcomes, shifted, strict=True)
        }
        self.reserved_value = float(base.reserved_value)

    def __call__(self, outcome: Any | None) -> float:
        if outcome is None:
            return self.reserved_value
        mapped = self._mapping.get(tuple(outcome), outcome)
        return float(self.base(mapped))

    def minmax(self) -> tuple[float, float]:
        values = [self(outcome) for outcome in self.outcomes]
        return min(values), max(values)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ci95(values: list[float]) -> list[float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return [mean - margin, mean + margin]


def _framework_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "belief_planner_anl" / "anl2025").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _framework_params(
    variant: str, *, evaluator_edge_ufuns: tuple[Any, ...]
) -> dict[str, Any]:
    if variant == "framework_static_belief_global_planner":
        return {"belief_mode": "static", "use_future_value": True}
    if variant == "framework_continuous_belief_local_planner":
        return {"belief_mode": "continuous", "use_future_value": False}
    if variant == "framework_continuous_belief_global_planner":
        return {"belief_mode": "continuous", "use_future_value": True}
    if variant == "framework_no_information_gain":
        return {
            "belief_mode": "continuous",
            "use_future_value": True,
            "information_gain_weight": 0.0,
        }
    if variant in ORACLE_VARIANTS:
        return {
            "belief_mode": "oracle",
            "use_future_value": True,
            "oracle_edge_ufuns": tuple(copy.deepcopy(evaluator_edge_ufuns)),
        }
    if variant in SHUFFLED_VARIANTS:
        shuffled_ufuns: list[Any] = []
        for edge_ufun in evaluator_edge_ufuns:
            outcomes = tuple(
                edge_ufun.outcome_space.enumerate_or_sample(
                    levels=10, max_cardinality=200
                )
            )
            shuffled_ufuns.append(_PermutedOutcomeUFun(edge_ufun, outcomes))
        return {
            "belief_mode": "fixed",
            "use_future_value": True,
            "oracle_edge_ufuns": tuple(shuffled_ufuns),
        }
    raise ValueError(f"Unknown framework variant: {variant}")


def _pairwise_preference_accuracy(
    belief: Any, edge_ufun: Any, outcomes: tuple[Any, ...]
) -> float | None:
    if len(outcomes) < 2:
        return None
    predicted = [float(belief.preference_score(outcome)) for outcome in outcomes]
    truth = [float(edge_ufun(outcome)) for outcome in outcomes]
    score = 0.0
    comparisons = 0
    for left in range(len(outcomes)):
        for right in range(left + 1, len(outcomes)):
            true_delta = truth[left] - truth[right]
            if abs(true_delta) <= 1e-12:
                continue
            predicted_delta = predicted[left] - predicted[right]
            comparisons += 1
            if abs(predicted_delta) <= 1e-12:
                score += 0.5
            elif predicted_delta * true_delta > 0.0:
                score += 1.0
    return score / comparisons if comparisons else None


def _belief_diagnostics(center: Any, evaluator_edge_ufuns: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(center, BeliefGlobalPlannerANL2025Center):
        return {
            "belief_pairwise_preference_accuracy": None,
            "belief_acceptance_brier": None,
            "belief_offer_observations": None,
            "belief_response_observations": None,
        }
    accuracies: list[float] = []
    calibration: list[dict[str, Any]] = []
    for belief, edge_ufun, outcomes in zip(
        center.beliefs,
        evaluator_edge_ufuns,
        center.outcomes_by_thread,
        strict=True,
    ):
        accuracy = _pairwise_preference_accuracy(belief, edge_ufun, outcomes)
        if accuracy is not None:
            accuracies.append(accuracy)
        calibration.extend(belief.calibration_records)
    brier = _mean(
        [
            (float(row["predicted"]) - float(bool(row["accepted"]))) ** 2
            for row in calibration
        ]
    )
    return {
        "belief_pairwise_preference_accuracy": _mean(accuracies),
        "belief_acceptance_brier": brier,
        "belief_offer_observations": sum(
            belief.offer_observations for belief in center.beliefs
        ),
        "belief_response_observations": sum(
            belief.response_observations for belief in center.beliefs
        ),
    }


def _edge_assignment(episode_seed: int) -> list[type]:
    edges = list(DEFAULT_EDGE_TYPES)
    random.Random(episode_seed).shuffle(edges)
    return edges


def run_episode(
    *,
    scenario: MultidealScenario,
    variant: str,
    repetition: int,
    nsteps: int,
    episode_seed: int,
) -> dict[str, Any]:
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    episode_scenario = copy.deepcopy(scenario)
    evaluator_edge_ufuns = tuple(copy.deepcopy(episode_scenario.edge_ufuns))
    center_type: type
    center_params: dict[str, Any]
    if variant in BASELINES:
        center_type = BASELINES[variant]
        center_params = {}
    elif variant in FRAMEWORK_VARIANTS:
        center_type = BeliefGlobalPlannerANL2025Center
        center_params = _framework_params(
            variant, evaluator_edge_ufuns=evaluator_edge_ufuns
        )
    else:
        raise ValueError(f"Unknown ANL 2025 variant: {variant}")

    edge_types = _edge_assignment(episode_seed)
    started = time.perf_counter()
    try:
        result = run_session(
            episode_scenario,
            center_type=center_type,
            center_params=center_params,
            edge_types=edge_types,
            nsteps=nsteps,
            keep_order=True,
            share_ufuns=False,
            atomic=False,
            output=None,
            method="sequential",
            sample_edges=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        diagnostics = _belief_diagnostics(result.center, evaluator_edge_ufuns)
        framework_trace = (
            result.center.framework_trace
            if isinstance(result.center, BeliefGlobalPlannerANL2025Center)
            else []
        )
        agreements = list(result.agreements)
        return {
            "variant": variant,
            "scenario_name": scenario.name,
            "repetition": repetition,
            "episode_seed": episode_seed,
            "edge_types": [edge.__name__ for edge in edge_types],
            "n_edges": len(agreements),
            "nsteps": nsteps,
            "share_ufuns": False,
            "method": "sequential",
            "agreements": _jsonable(agreements),
            "n_agreements": sum(outcome is not None for outcome in agreements),
            "agreement_fraction": (
                sum(outcome is not None for outcome in agreements) / len(agreements)
                if agreements
                else 0.0
            ),
            "center_utility": float(result.center_utility),
            "edge_utilities": [float(value) for value in result.edge_utilities],
            "mean_edge_utility": _mean(
                [float(value) for value in result.edge_utilities]
            ),
            "total_time": float(result.total_time),
            "elapsed_seconds": round(elapsed, 6),
            "run_error": str(result.run_error or ""),
            "has_error": bool(result.run_error),
            **diagnostics,
            "framework_trace": _jsonable(framework_trace),
        }
    except Exception as exc:
        return {
            "variant": variant,
            "scenario_name": scenario.name,
            "repetition": repetition,
            "episode_seed": episode_seed,
            "edge_types": [edge.__name__ for edge in edge_types],
            "n_edges": len(scenario.edge_ufuns),
            "nsteps": nsteps,
            "share_ufuns": False,
            "method": "sequential",
            "agreements": [],
            "n_agreements": 0,
            "agreement_fraction": 0.0,
            "center_utility": None,
            "edge_utilities": [],
            "mean_edge_utility": None,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "run_error": f"{type(exc).__name__}: {exc}",
            "has_error": True,
            "belief_pairwise_preference_accuracy": None,
            "belief_acceptance_brier": None,
            "belief_offer_observations": None,
            "belief_response_observations": None,
            "framework_trace": [],
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    summary: dict[str, Any] = {}
    for variant, group in groups.items():
        valid = [row for row in group if not row["has_error"]]
        center_utilities = [float(row["center_utility"]) for row in valid]
        summary[variant] = {
            "episodes": len(group),
            "valid_episodes": len(valid),
            "errors": len(group) - len(valid),
            "mean_center_utility": _mean(center_utilities),
            "center_utility_ci95": _ci95(center_utilities),
            "mean_edge_utility": _mean(
                [float(row["mean_edge_utility"]) for row in valid]
            ),
            "mean_agreement_fraction": _mean(
                [float(row["agreement_fraction"]) for row in valid]
            ),
            "full_agreement_rate": _mean(
                [float(row["agreement_fraction"] == 1.0) for row in valid]
            ),
            "belief_pairwise_preference_accuracy": _mean(
                [
                    float(row["belief_pairwise_preference_accuracy"])
                    for row in valid
                    if row["belief_pairwise_preference_accuracy"] is not None
                ]
            ),
            "belief_acceptance_brier": _mean(
                [
                    float(row["belief_acceptance_brier"])
                    for row in valid
                    if row["belief_acceptance_brier"] is not None
                ]
            ),
        }
    return summary


def _scenario_family(name: str) -> str:
    if name.startswith("generated_max_"):
        return "generated_max"
    if name.startswith("generated_linear_"):
        return "generated_linear"
    if name.startswith("generated_"):
        return "generated"
    return name


def summarize_by_scenario_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_scenario_family(str(row["scenario_name"]))].append(row)
    return {family: summarize(group) for family, group in sorted(grouped.items())}


def _action_signature(row: dict[str, Any]) -> list[Any]:
    return [
        (
            event.get("event"),
            event.get("thread_index"),
            event.get("chosen", {}).get("outcome")
            if isinstance(event.get("chosen"), dict)
            else None,
        )
        for event in row.get("framework_trace", [])
        if event.get("event") in {"propose", "accept", "reject"}
    ]


def paired_diagnostics(
    rows: list[dict[str, Any]],
    reference: str = "framework_static_belief_global_planner",
) -> dict[str, Any]:
    keyed = {
        (row["scenario_name"], row["repetition"], row["variant"]): row
        for row in rows
        if not row["has_error"]
    }
    comparisons: dict[str, Any] = {}
    variants = sorted({row["variant"] for row in rows if row["variant"] != reference})
    for variant in variants:
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for (scenario_name, repetition, name), reference_row in keyed.items():
            if name != reference:
                continue
            candidate = keyed.get((scenario_name, repetition, variant))
            if candidate is not None:
                pairs.append((reference_row, candidate))
        comparisons[variant] = {
            "paired_episodes": len(pairs),
            "agreement_vector_flip_rate": _mean(
                [float(left["agreements"] != right["agreements"]) for left, right in pairs]
            ),
            "framework_action_flip_rate": _mean(
                [
                    float(_action_signature(left) != _action_signature(right))
                    for left, right in pairs
                ]
            ) if variant in FRAMEWORK_VARIANTS else None,
            "mean_center_utility_delta": _mean(
                [
                    float(right["center_utility"]) - float(left["center_utility"])
                    for left, right in pairs
                ]
            ),
            "center_utility_win_rate": _mean(
                [
                    float(float(right["center_utility"]) > float(left["center_utility"]))
                    for left, right in pairs
                ]
            ),
        }
    return {"reference_variant": reference, "comparisons": comparisons}


def load_scenario(scenario_root: Path, name: str, seed: int) -> MultidealScenario:
    generated_types = {
        "generated": "LinearCombinationCenterUFun",
        "generated_linear": "LinearCombinationCenterUFun",
        "generated_max": "MaxCenterUFun",
    }
    if name in generated_types:
        random.seed(seed)
        np.random.seed(seed)
        return make_multideal_scenario(
            nedges=4,
            nissues=2,
            nvalues=4,
            center_ufun_type=generated_types[name],
            name=f"{name}_{seed}",
        )
    scenario = MultidealScenario.from_folder(scenario_root / name)
    if scenario is None:
        raise FileNotFoundError(f"Could not load ANL 2025 scenario: {scenario_root / name}")
    return scenario


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Paired ANL 2025 belief/global-planner experiments"
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=root / "official" / "anl-2025-platform" / "scenarios",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["dinners", "job_hunt", "target_quantity"],
    )
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--nsteps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20250805)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "runs" / "anl2025_framework_basic"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    known_variants = set(BASELINES) | FRAMEWORK_VARIANTS
    unknown = [variant for variant in args.variants if variant not in known_variants]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "episodes.jsonl"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, int, str]] = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            completed.add((row["scenario_name"], row["repetition"], row["variant"]))

    root = Path(__file__).resolve().parents[2]
    config = {
        "runner_version": "anl2025-v1.0",
        "scenarios": args.scenarios,
        "variants": args.variants,
        "repetitions": args.repetitions,
        "nsteps": args.nsteps,
        "seed": args.seed,
        "share_ufuns": False,
        "method": "sequential",
        "framework_hash": _framework_hash(root),
    }
    config_path = args.output_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        protected = (
            "scenarios",
            "variants",
            "repetitions",
            "nsteps",
            "seed",
            "share_ufuns",
            "method",
            "framework_hash",
        )
        changed = [key for key in protected if previous.get(key) != config.get(key)]
        if changed:
            raise RuntimeError(
                "Refusing to resume with changed configuration/code: "
                + ", ".join(changed)
                + ". Use a new output directory to preserve versioned records."
            )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    manifest = {
        "framework_hash": config["framework_hash"],
        "python": sys.version,
        "anl2025_platform_commit": _git_commit(
            root / "official" / "anl-2025-platform"
        ),
        "anl_platform_commit": _git_commit(root / "official" / "anl-platform-current"),
        "negmas_commit": _git_commit(root / "official" / "negmas"),
    }
    (args.output_dir / "code_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    file_mode = "a" if args.resume else "w"
    with rows_path.open(file_mode) as stream:
        for scenario_offset, scenario_name in enumerate(args.scenarios):
            scenario_seed = args.seed + scenario_offset * 100_000
            generated = scenario_name.startswith("generated")
            fixed_scenario = (
                None
                if generated
                else load_scenario(args.scenario_root, scenario_name, scenario_seed)
            )
            for repetition in range(args.repetitions):
                episode_seed = scenario_seed + repetition
                scenario = (
                    load_scenario(args.scenario_root, scenario_name, episode_seed)
                    if generated
                    else fixed_scenario
                )
                if scenario is None:
                    raise RuntimeError(f"Scenario loading failed: {scenario_name}")
                for variant in args.variants:
                    key = (scenario.name, repetition, variant)
                    if key in completed:
                        continue
                    row = run_episode(
                        scenario=scenario,
                        variant=variant,
                        repetition=repetition,
                        nsteps=args.nsteps,
                        episode_seed=episode_seed,
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "episode_result": {
                                    "scenario": row["scenario_name"],
                                    "variant": variant,
                                    "repetition": repetition,
                                    "center_utility": row["center_utility"],
                                    "agreement_fraction": row["agreement_fraction"],
                                    "error": row["run_error"],
                                }
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    output = {
        "summary": summarize(rows),
        "summary_by_scenario_family": summarize_by_scenario_family(rows),
        "paired_diagnostics": paired_diagnostics(rows),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
