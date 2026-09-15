from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from anl.anl2024.negotiators.builtins import Boulware, Conceder, Linear, RVFitter
from anl.anl2024.runner import mixed_scenarios
from negmas import SAOMechanism

from .agent import BeliefPlannerANL2024Negotiator


BASELINES: dict[str, type] = {
    "boulware": Boulware,
    "linear": Linear,
    "conceder": Conceder,
    "rv_fitter": RVFitter,
}

VARIANT_ALIASES = {
    # Legacy aliases are accepted only for command compatibility. New outputs
    # always use the framework_* scientific names.
    "astra_static": "framework_static_belief",
    "astra_continuous": "framework_continuous_belief",
    "astra_no_info_gain": "framework_no_info_gain",
    "astra_frozen50": "framework_frozen_belief",
    "astra_oracle": "framework_oracle_belief",
    "astra_shuffled": "framework_shuffled_belief",
}

ORACLE_VARIANTS = {
    "framework_oracle_belief",
    "framework_oracle_belief_planner_v2",
}
SHUFFLED_VARIANTS = {
    "framework_shuffled_belief",
    "framework_shuffled_belief_planner_v2",
}


def canonical_variant(variant: str) -> str:
    return VARIANT_ALIASES.get(variant, variant)


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


def _known_shape(ufun: Any) -> Any:
    """Copy a ufun while masking its hidden reservation value."""

    result = copy.deepcopy(ufun)
    result.reserved_value = 0.0
    return result


def _advantage(ufun: Any, agreement: Any) -> float:
    reservation = float(ufun.reserved_value)
    utility = float(ufun(agreement)) if agreement is not None else reservation
    maximum = float(ufun.max())
    if maximum <= reservation:
        return 0.0
    return (utility - reservation) / (maximum - reservation)


def make_agent(variant: str, *, opponent_shape: Any, oracle_rv: float | None = None) -> Any:
    variant = canonical_variant(variant)
    private_info = {"opponent_ufun": opponent_shape}
    if variant in BASELINES:
        return BASELINES[variant](name=variant, private_info=private_info)
    if variant == "framework_static_belief":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="static",
        )
    if variant == "framework_continuous_belief":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
        )
    if variant == "framework_no_info_gain":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
            information_gain_weight=0.0,
        )
    if variant == "framework_frozen_belief":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
            freeze_after=0.5,
        )
    if variant == "framework_oracle_belief":
        if oracle_rv is None:
            raise ValueError("framework_oracle_belief requires the evaluator-only oracle RV")
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="oracle",
            oracle_rv=oracle_rv,
        )
    if variant == "framework_shuffled_belief":
        if oracle_rv is None:
            raise ValueError("framework_shuffled_belief requires an evaluator-supplied paired RV")
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="fixed",
            oracle_rv=oracle_rv,
        )
    if variant == "framework_static_belief_planner_v2":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="static",
            planner_version="v2",
        )
    if variant == "framework_continuous_belief_planner_v2":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
            planner_version="v2",
        )
    if variant == "framework_no_info_gain_planner_v2":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
            planner_version="v2",
            information_gain_weight=0.0,
        )
    if variant == "framework_frozen_belief_planner_v2":
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="continuous",
            planner_version="v2",
            freeze_after=0.5,
        )
    if variant == "framework_oracle_belief_planner_v2":
        if oracle_rv is None:
            raise ValueError(
                "framework_oracle_belief_planner_v2 requires evaluator-only RV"
            )
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="oracle",
            planner_version="v2",
            oracle_rv=oracle_rv,
        )
    if variant == "framework_shuffled_belief_planner_v2":
        if oracle_rv is None:
            raise ValueError(
                "framework_shuffled_belief_planner_v2 requires paired RV"
            )
        return BeliefPlannerANL2024Negotiator(
            name=variant,
            private_info=private_info,
            belief_mode="fixed",
            planner_version="v2",
            oracle_rv=oracle_rv,
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_episode(
    *,
    scenario: Any,
    variant: str,
    opponent: str,
    n_steps: int,
    episode_seed: int,
    belief_override_rv: float | None = None,
) -> dict[str, Any]:
    variant = canonical_variant(variant)
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    own_ufun = copy.deepcopy(scenario.ufuns[0])
    partner_ufun = copy.deepcopy(scenario.ufuns[1])
    true_partner_rv = float(partner_ufun.reserved_value)
    agent = make_agent(
        variant,
        opponent_shape=_known_shape(partner_ufun),
        oracle_rv=(
            true_partner_rv
            if variant in ORACLE_VARIANTS
            else belief_override_rv
            if variant in SHUFFLED_VARIANTS
            else None
        ),
    )
    if opponent not in BASELINES:
        raise ValueError(f"Unknown opponent: {opponent}")
    partner = BASELINES[opponent](
        name=f"opponent_{opponent}",
        private_info={"opponent_ufun": _known_shape(own_ufun)},
    )
    mechanism = SAOMechanism(
        outcome_space=own_ufun.outcome_space,
        n_steps=n_steps,
        name=f"anl2024_{variant}_{opponent}_{episode_seed}",
    )
    mechanism.add(agent, preferences=own_ufun)
    mechanism.add(partner, preferences=partner_ufun)
    started = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - started
    agreement = state.agreement
    own_utility = float(own_ufun(agreement)) if agreement is not None else float(own_ufun.reserved_value)
    partner_utility = (
        float(partner_ufun(agreement))
        if agreement is not None
        else float(partner_ufun.reserved_value)
    )
    final_belief = None
    trace: list[dict[str, Any]] = []
    if isinstance(agent, BeliefPlannerANL2024Negotiator):
        final_belief = agent.belief.to_json()
        trace = agent.framework_trace
    row = {
        "variant": variant,
        "opponent": opponent,
        "episode_seed": episode_seed,
        "scenario_name": str(own_ufun.outcome_space.name),
        "n_outcomes": int(own_ufun.outcome_space.cardinality),
        "n_steps_limit": n_steps,
        "steps_elapsed": int(state.step),
        "agreement": _jsonable(agreement),
        "agreement_reached": agreement is not None,
        "own_utility": own_utility,
        "partner_utility": partner_utility,
        "joint_utility": own_utility + partner_utility,
        "own_reserved_value": float(own_ufun.reserved_value),
        "partner_reserved_value_truth": true_partner_rv,
        "own_advantage": _advantage(own_ufun, agreement),
        "partner_advantage": _advantage(partner_ufun, agreement),
        "final_belief": final_belief,
        "belief_abs_error": (
            abs(float(final_belief["mean"]) - true_partner_rv)
            if final_belief is not None
            else None
        ),
        "belief_truth_covered_q10_q90": (
            bool(final_belief["q10"] <= true_partner_rv <= final_belief["q90"])
            if final_belief is not None
            else None
        ),
        "has_error": bool(state.has_error),
        "error_details": str(state.error_details or ""),
        "timedout": bool(state.timedout),
        "elapsed_seconds": round(elapsed, 6),
        "framework_trace": _jsonable(trace),
    }
    return row


def generate_scenario(*, seed: int, n_outcomes: int) -> Any:
    random.seed(seed)
    np.random.seed(seed)
    scenarios = mixed_scenarios(
        n_scenarios=1,
        n_outcomes=n_outcomes,
        reserved_ranges=((0.05, 0.75), (0.05, 0.75)),
        log_uniform=False,
    )
    if not scenarios:
        raise RuntimeError(f"Scenario generation failed for seed {seed}")
    return scenarios[0]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    result: dict[str, Any] = {}
    for variant, values in sorted(groups.items()):
        belief_rows = [row for row in values if row["belief_abs_error"] is not None]
        result[variant] = {
            "episodes": len(values),
            "agreement_rate": sum(row["agreement_reached"] for row in values) / len(values),
            "mean_own_utility": sum(row["own_utility"] for row in values) / len(values),
            "mean_partner_utility": sum(row["partner_utility"] for row in values) / len(values),
            "mean_joint_utility": sum(row["joint_utility"] for row in values) / len(values),
            "mean_own_advantage": sum(row["own_advantage"] for row in values) / len(values),
            "mean_belief_abs_error": (
                sum(row["belief_abs_error"] for row in belief_rows) / len(belief_rows)
                if belief_rows
                else None
            ),
            "belief_q10_q90_coverage": (
                sum(row["belief_truth_covered_q10_q90"] for row in belief_rows) / len(belief_rows)
                if belief_rows
                else None
            ),
            "errors": sum(row["has_error"] for row in values),
        }
    return result


def paired_diagnostics(
    rows: list[dict[str, Any]], reference_variant: str = "framework_static_belief"
) -> dict[str, Any]:
    """Measure whether beliefs cause action/outcome changes on paired episodes."""

    def key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["scenario_seed"]), str(row["opponent"])

    def offers(row: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            tuple(event["chosen"]["outcome"])
            for event in row.get("framework_trace", [])
            if event.get("event") == "offer"
            and event.get("chosen", {}).get("outcome")
        )

    reference = {key(row): row for row in rows if row["variant"] == reference_variant}
    variants = sorted(
        {
            str(row["variant"])
            for row in rows
            if str(row["variant"]).startswith("framework_")
            and row["variant"] != reference_variant
        }
    )
    result: dict[str, Any] = {}
    for variant in variants:
        comparisons = []
        for row in rows:
            if row["variant"] != variant or key(row) not in reference:
                continue
            base = reference[key(row)]
            current_offers, base_offers = offers(row), offers(base)
            comparisons.append(
                {
                    "first_action_flip": bool(
                        current_offers
                        and base_offers
                        and current_offers[0] != base_offers[0]
                    ),
                    "offer_sequence_flip": current_offers != base_offers,
                    "agreement_flip": row["agreement"] != base["agreement"],
                }
            )
        if comparisons:
            result[variant] = {
                "paired_episodes": len(comparisons),
                "first_action_flip_rate": sum(
                    value["first_action_flip"] for value in comparisons
                )
                / len(comparisons),
                "offer_sequence_flip_rate": sum(
                    value["offer_sequence_flip"] for value in comparisons
                )
                / len(comparisons),
                "agreement_outcome_flip_rate": sum(
                    value["agreement_flip"] for value in comparisons
                )
                / len(comparisons),
            }
    return {"reference_variant": reference_variant, "comparisons": result}


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


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def run_experiment(
    *,
    output_dir: Path,
    variants: list[str],
    opponents: list[str],
    runs: int,
    n_outcomes: int,
    n_steps: int,
    seed: int,
    resume: bool,
    reference_variant: str | None = None,
) -> dict[str, Any]:
    requested_variants = list(variants)
    variants = list(dict.fromkeys(canonical_variant(value) for value in variants))
    if reference_variant is None:
        reference_variant = (
            "framework_static_belief_planner_v2"
            if "framework_static_belief_planner_v2" in variants
            else "framework_static_belief"
        )
    reference_variant = canonical_variant(reference_variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    existing: list[dict[str, Any]] = []
    completed: set[tuple[int, str, str]] = set()
    if resume and episodes_path.exists():
        existing = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        for row in existing:
            row["variant"] = canonical_variant(str(row["variant"]))
        completed = {
            (
                int(row["scenario_seed"]),
                canonical_variant(str(row["variant"])),
                str(row["opponent"]),
            )
            for row in existing
        }
    config = {
        "requested_variants": requested_variants,
        "variants": variants,
        "opponents": opponents,
        "runs": runs,
        "n_outcomes": n_outcomes,
        "n_steps": n_steps,
        "seed": seed,
        "resume": resume,
        "paired_reference_variant": reference_variant,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    root = Path(__file__).resolve().parents[2]
    official = root / "official"
    manifest = {
        "framework_root": str(root),
        "framework_source_sha256": _source_hash(root / "belief_planner_anl"),
        "anl_platform_commit": _git_commit(official / "anl-platform-current"),
        "negmas_commit": _git_commit(official / "negmas"),
        "anl_agents_commit": _git_commit(official / "anl-agents"),
    }
    (output_dir / "code_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = list(existing)
    scenarios = [
        generate_scenario(seed=seed + run_index, n_outcomes=n_outcomes)
        for run_index in range(runs)
    ]
    for run_index in range(runs):
        scenario_seed = seed + run_index
        scenario = scenarios[run_index]
        true_rv = float(scenario.ufuns[1].reserved_value)
        shuffled_rv = (
            float(scenarios[(run_index + 1) % runs].ufuns[1].reserved_value)
            if runs > 1
            else max(0.0, min(1.0, 1.0 - true_rv))
        )
        for opponent_index, opponent in enumerate(opponents):
            paired_episode_seed = scenario_seed * 1009 + opponent_index
            for variant in variants:
                key = (scenario_seed, variant, opponent)
                if key in completed:
                    continue
                row = run_episode(
                    scenario=scenario,
                    variant=variant,
                    opponent=opponent,
                    n_steps=n_steps,
                    episode_seed=paired_episode_seed,
                    belief_override_rv=(
                        shuffled_rv if variant in SHUFFLED_VARIANTS else None
                    ),
                )
                row["scenario_seed"] = scenario_seed
                with episodes_path.open("a") as handle:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add(key)
                print(json.dumps({"episode_result": {key: value for key, value in row.items() if key != "framework_trace"}}))
    summary = summarize(rows)
    payload = {
        "summary": summary,
        "paired_diagnostics": paired_diagnostics(rows, reference_variant),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous-belief/planner framework comparison on ANL 2024"
    )
    parser.add_argument(
        "--variants",
        default=(
            "boulware,rv_fitter,framework_static_belief,"
            "framework_continuous_belief,framework_no_info_gain"
        ),
    )
    parser.add_argument("--opponents", default="boulware,linear,conceder")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--n-outcomes", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/anl2024_basic")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reference-variant",
        default=None,
        help="Paired diagnostic reference; auto-selects the matching static planner",
    )
    args = parser.parse_args()
    run_experiment(
        output_dir=Path(args.output_dir),
        variants=[value.strip() for value in args.variants.split(",") if value.strip()],
        opponents=[value.strip() for value in args.opponents.split(",") if value.strip()],
        runs=args.runs,
        n_outcomes=args.n_outcomes,
        n_steps=args.n_steps,
        seed=args.seed,
        resume=args.resume,
        reference_variant=args.reference_variant,
    )


if __name__ == "__main__":
    main()
