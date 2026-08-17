#!/usr/bin/env python3
"""Fine-tune Planner V2 on oracle-labelled normalized candidate sets.

This is listwise policy training, not CEM and not language-model SFT.  Hidden
seller profiles are used only by the evaluator-side oracle that assigns a
counterfactual return to every feasible candidate.  The deployed network sees
only public-history belief features and normalized candidate attributes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.planner import BehavioralFrontierPlanner
from framework.schemas import (
    CandidateAction,
    CanonicalOffer,
    CanonicalState,
    OpponentBelief,
    PreferenceBelief,
    ResponsePolicyBelief,
)
from framework.trainable_planner import FEATURE_NAMES, candidate_features
from Simple_Env.buyer.continuous_solver.identifiable import UtilityProfile


MAX_TURNS = 6
BUDGET = 95.0
REFERENCE = 100.0


def profile_suite(split: str) -> list[UtilityProfile]:
    if split == "train":
        ratios = (0.35, 0.50, 0.65, 0.80)
        policies = (
            ("patient", 0.94, 0.015, 0.10),
            ("neutral", 0.76, 0.09, 0.38),
            ("firm", 0.45, 0.30, 0.78),
        )
    elif split == "dev":
        ratios = (0.425, 0.575, 0.725, 0.875)
        policies = (
            ("flexible", 0.87, 0.04, 0.18),
            ("hardball", 0.62, 0.16, 0.62),
            ("impatient", 0.34, 0.40, 0.90),
        )
    elif split == "test":
        ratios = (0.275, 0.475, 0.675, 0.925)
        policies = (
            ("very_patient", 0.97, 0.01, 0.06),
            ("strategic", 0.69, 0.11, 0.70),
            ("very_firm", 0.28, 0.49, 0.96),
        )
    else:
        raise ValueError(split)
    return [
        UtilityProfile(
            profile_id=f"{split}_{name}_{ratio:.3f}",
            cost_ratio=ratio,
            counter_propensity=counter,
            quit_bias=quit,
            finality=finality,
        )
        for ratio in ratios
        for name, counter, quit, finality in policies
    ]


def response_distribution(profile: UtilityProfile, proxy: float, progress: float) -> dict[str, float]:
    """Strategic response oracle with a time-varying aspiration boundary."""

    offer = proxy * BUDGET
    aspiration_margin = REFERENCE * (0.04 + 0.18 * profile.finality) * (1.0 - progress)
    threshold = profile.cost_ratio * REFERENCE + aspiration_margin
    tau = max(0.25, profile.rationality_tau_ratio * REFERENCE)
    accept = 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, (threshold - offer) / tau))))
    remaining = 1.0 - accept
    quit_share = min(0.95, max(0.01, profile.quit_bias + 0.30 * progress + 0.20 * profile.finality))
    counter_share = min(0.99 - quit_share, max(0.01, profile.counter_propensity * (1.0 - 0.32 * progress)))
    raw = {
        "accept": accept,
        "counter": remaining * counter_share,
        "quit": remaining * (1.0 - counter_share),
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def make_belief(profile: UtilityProfile, turn: int, evidence: int, rng: random.Random) -> OpponentBelief:
    progress = turn / MAX_TURNS
    true_ratio = profile.cost_ratio * REFERENCE / BUDGET
    if evidence == 0:
        # No public response means no profile-specific information.  This must
        # match the deployed broad prior; sampling around evaluator truth here
        # would leak hidden utility into the first action.
        return OpponentBelief(
            counterparty_id="seller",
            preference=PreferenceBelief(
                reservation_ratio_low=0.20,
                reservation_ratio_mean=0.575,
                reservation_ratio_high=0.95,
                reliability=0.10,
            ),
            response_policy=ResponsePolicyBelief(
                counter_rate=0.7266666666666667,
                quit_rate=0.12666666666666668,
                patience=0.60,
                reliability=0.15,
                aspiration_ratio_low=0.30,
                aspiration_ratio_mean=0.70,
                aspiration_ratio_high=1.00,
                aspiration_margin_mean=0.25,
                aspiration_reliability=0.10,
            ),
            latent_profile_entropy_bits=5.5,
        )
    reliability = min(0.92, 0.10 + 0.15 * evidence)
    noise = rng.gauss(0.0, 0.17 * (1.0 - reliability))
    mean = max(0.15, min(1.05, true_ratio + noise))
    width = max(0.06, 0.42 * (1.0 - reliability))
    low, high = max(0.05, mean - width), min(1.12, mean + width)
    readiness = true_ratio + (0.04 + 0.18 * profile.finality) * REFERENCE / BUDGET * (1.0 - progress)
    frontier = 0.0
    if evidence:
        frontier = max(0.12, min(0.98, readiness - rng.uniform(0.015, 0.16)))
    policy_reliability = min(0.95, 0.15 + 0.17 * evidence)
    aspiration_width = max(0.05, 0.38 * (1.0 - policy_reliability))
    policy = ResponsePolicyBelief(
        counter_rate=max(0.0, min(1.0, profile.counter_propensity + rng.gauss(0, 0.08))),
        quit_rate=max(0.0, min(1.0, profile.quit_bias + rng.gauss(0, 0.05))),
        patience=max(0.0, min(1.0, 1.0 - profile.finality + rng.gauss(0, 0.07))),
        reliability=policy_reliability,
        rejected_compatibility_max=frontier,
        repeated_rejection_streak=min(4, evidence),
        last_rejected_compatibility=frontier if evidence else None,
        aspiration_ratio_low=max(0.05, readiness - aspiration_width),
        aspiration_ratio_mean=max(0.05, min(1.10, readiness + rng.gauss(0, 0.05))),
        aspiration_ratio_high=min(1.15, readiness + aspiration_width),
        aspiration_margin_mean=max(0.0, readiness - true_ratio),
        aspiration_reliability=policy_reliability,
        last_observed_ask=min(
            1.05,
            (
                profile.cost_ratio * REFERENCE
                + (REFERENCE - profile.cost_ratio * REFERENCE)
                * max(0.08, 0.62 * (1.0 - progress))
            ) / BUDGET,
        ),
    )
    return OpponentBelief(
        counterparty_id="seller",
        preference=PreferenceBelief(
            reservation_ratio_low=low,
            reservation_ratio_mean=mean,
            reservation_ratio_high=high,
            reliability=reliability,
        ),
        response_policy=policy,
        evidence_count=evidence,
        direct_response_count=evidence,
        regime_change_probability=rng.uniform(0.0, 0.25 if evidence else 0.05),
        latent_profile_entropy_bits=max(0.0, 5.5 * (1.0 - reliability)),
    )


def make_state(profile: UtilityProfile, case_id: int, turn: int) -> CanonicalState:
    return CanonicalState(
        session_id=f"planner_v2_{profile.profile_id}_{case_id}",
        environment_id="normalized_negotiation_training",
        self_id="focal_agent",
        role="buyer",
        counterparty_id="seller",
        turn=turn,
        max_turns=MAX_TURNS,
        issues=[],
        own_value_scale=BUDGET,
        own_outside_option=0.0,
        reference_value=REFERENCE,
        metadata={"framework_version": "trainable_planner_v2"},
    )


def make_candidates(state: CanonicalState, belief: OpponentBelief, rng: random.Random) -> list[CandidateAction]:
    frontier = belief.response_policy.rejected_compatibility_max
    points = {0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.98}
    points.update({
        belief.preference.reservation_ratio_low,
        belief.preference.reservation_ratio_mean,
        belief.preference.reservation_ratio_high,
        frontier + 0.03,
        frontier + 0.10,
    })
    offers = []
    for index, proxy in enumerate(sorted(max(0.20, min(0.995, value)) for value in points)):
        own = max(-0.05, 1.0 - proxy)
        information = math.exp(-0.5 * ((proxy - belief.preference.reservation_ratio_mean) / 0.14) ** 2)
        offers.append(CandidateAction(
            candidate_id=f"offer_{index}_{proxy:.4f}",
            action_type="offer",
            counterparty_id="seller",
            offer=CanonicalOffer(price=proxy * BUDGET),
            own_utility=own * BUDGET,
            own_utility_normalized=own,
            opponent_value_proxy=proxy,
            base_acceptance=max(0.03, min(0.97, 0.10 + 0.43 * proxy + 0.27 * state.progress)),
            information_gain=information,
            feasibility_margin=own,
            rationale="normalized candidate",
            metadata={"action_novelty": rng.uniform(0.35, 1.0)},
        ))
    if state.turn > 1 and belief.direct_response_count > 0:
        ask_proxy = belief.response_policy.last_observed_ask
        if ask_proxy is None:
            ask_proxy = belief.response_policy.aspiration_ratio_mean
        ask_proxy = max(0.35, min(1.0, ask_proxy))
        own = max(-0.05, 1.0 - ask_proxy)
        offers.append(CandidateAction(
            candidate_id=f"accept_{ask_proxy:.4f}", action_type="accept", counterparty_id="seller",
            offer=CanonicalOffer(price=ask_proxy * BUDGET), own_utility=own * BUDGET,
            own_utility_normalized=own, opponent_value_proxy=1.0, base_acceptance=1.0,
            feasibility_margin=own, rationale="accept current counteroffer",
        ))
    offers.append(CandidateAction(
        candidate_id="quit", action_type="quit", counterparty_id="seller", offer=None,
        own_utility=0.0, own_utility_normalized=0.0, opponent_value_proxy=0.0,
        base_acceptance=0.0, rationale="outside option",
    ))
    return offers


def offer_oracle_value(
    profile: UtilityProfile,
    proxy: float,
    progress: float,
    successors: list[float],
    *,
    depth: int,
) -> float:
    own = surplus_reward(profile, proxy)
    dist = response_distribution(profile, proxy, progress)
    if depth <= 0 or progress >= 1.0:
        return dist["accept"] * own
    next_progress = min(1.0, progress + 1.0 / MAX_TURNS)
    future = max(
        (offer_oracle_value(profile, value, next_progress, successors, depth=depth - 1)
         for value in successors if value > proxy + 1e-6),
        default=0.0,
    )
    return dist["accept"] * own + dist["counter"] * 0.88 * future


def surplus_reward(profile: UtilityProfile, proxy: float) -> float:
    """Paper-aligned bargained-surplus reward used only for oracle labels."""

    price = proxy * BUDGET
    available = BUDGET - profile.cost_ratio * REFERENCE
    if available <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (BUDGET - price) / available))


def build_case(profile: UtilityProfile, case_id: int, rng: random.Random) -> dict[str, Any]:
    turn = rng.randint(1, MAX_TURNS)
    evidence = 0 if turn == 1 else rng.randint(1, min(4, turn - 1))
    state = make_state(profile, case_id, turn)
    belief = make_belief(profile, turn, evidence, rng)
    candidates = make_candidates(state, belief, rng)
    base_rows = BehavioralFrontierPlanner().rank(state, belief, candidates)
    top = max(row.score for row in base_rows)
    best_offer = max(row.candidate.own_utility_normalized for row in base_rows if row.candidate.action_type == "offer")
    proxies = sorted(row.candidate.opponent_value_proxy for row in base_rows if row.candidate.action_type == "offer")
    uncertainty = 1.0 - 0.5 * (belief.preference.reliability + belief.response_policy.reliability)
    rows = []
    for row in base_rows:
        candidate = row.candidate
        if candidate.action_type == "offer":
            q = offer_oracle_value(profile, candidate.opponent_value_proxy, state.progress, proxies, depth=2)
            # A response near an uncertain posterior boundary has option value;
            # this is capped so information can never dominate focal surplus.
            info = 0.07 * (1.0 - state.progress) * uncertainty * candidate.information_gain
            q = min(1.0, q + min(info, 0.025))
        elif candidate.action_type == "accept":
            q = surplus_reward(profile, candidate.offer.price / BUDGET)
        else:
            q = 0.0
        rows.append({
            "candidate_id": candidate.candidate_id,
            "action_type": candidate.action_type,
            "features": candidate_features(state, belief, row, base_top_score=top, best_offer_utility=best_offer),
            "oracle_q": q,
            "base_score": row.score,
            "opponent_proxy": candidate.opponent_value_proxy,
            "information_gain": candidate.information_gain,
        })
    return {
        "context_id": state.session_id,
        "profile_id": profile.profile_id,
        "turn": turn,
        "progress": state.progress,
        "belief_only_inputs": True,
        "candidates": rows,
    }


def generate_split(split: str, cases_per_profile: int, seed: int) -> list[dict[str, Any]]:
    cases = []
    for profile_id, profile in enumerate(profile_suite(split)):
        rng = random.Random(seed + profile_id * 1_000_003)
        for case_id in range(cases_per_profile):
            cases.append(build_case(profile, case_id, rng))
    return cases


class CandidateQNetwork(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width), nn.Tanh(),
            nn.Linear(width, width // 2), nn.Tanh(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def tensors(case: dict[str, Any], mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([row["features"] for row in case["candidates"]], dtype=torch.float32)
    y = torch.tensor([row["oracle_q"] for row in case["candidates"]], dtype=torch.float32)
    return (x - mean) / std, y


def loss_for_predictions(pred: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    target_policy = torch.softmax(target_q / 0.055, dim=0)
    policy_loss = -(target_policy * torch.log_softmax(pred / 0.10, dim=0)).sum()
    value_loss = F.smooth_l1_loss(pred, target_q)
    best = int(torch.argmax(target_q))
    gaps = target_q[best] - target_q
    mask = gaps > 0.02
    if bool(mask.any()):
        margin_loss = F.relu(0.04 + pred[mask] - pred[best]).mean()
    else:
        margin_loss = pred.sum() * 0.0
    return policy_loss + 0.45 * value_loss + 0.30 * margin_loss


def batch_loss(
    model: nn.Module,
    cases: list[dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    encoded = [tensors(case, mean, std) for case in cases]
    lengths = [len(x) for x, _ in encoded]
    predictions = model(torch.cat([x for x, _ in encoded], dim=0))
    losses = []
    offset = 0
    for length, (_, target_q) in zip(lengths, encoded):
        losses.append(loss_for_predictions(predictions[offset:offset + length], target_q))
        offset += length
    return torch.stack(losses).mean()


def evaluate(model: nn.Module, cases: list[dict[str, Any]], mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    regrets, base_regrets = [], []
    correct = base_correct = flips = 0
    selected_actions: dict[str, int] = {}
    with torch.no_grad():
        for case in cases:
            x, q = tensors(case, mean, std)
            pred = model(x)
            chosen = int(torch.argmax(pred))
            oracle = int(torch.argmax(q))
            base = max(range(len(case["candidates"])), key=lambda i: case["candidates"][i]["base_score"])
            regrets.append(float(q[oracle] - q[chosen]))
            base_regrets.append(float(q[oracle] - q[base]))
            correct += chosen == oracle
            base_correct += base == oracle
            flips += chosen != base
            action = case["candidates"][chosen]["action_type"]
            selected_actions[action] = selected_actions.get(action, 0) + 1
    return {
        "contexts": len(cases),
        "top1_accuracy": correct / max(1, len(cases)),
        "base_top1_accuracy": base_correct / max(1, len(cases)),
        "mean_oracle_regret": statistics.mean(regrets),
        "base_mean_oracle_regret": statistics.mean(base_regrets),
        "mean_regret_improvement": statistics.mean(base_regrets) - statistics.mean(regrets),
        "action_flip_rate_vs_base": flips / max(1, len(cases)),
        "selected_action_distribution": selected_actions,
    }


def export_checkpoint(model: CandidateQNetwork, mean: torch.Tensor, std: torch.Tensor, path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    layers = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            layers.append({
                "weight": module.weight.detach().cpu().tolist(),
                "bias": module.bias.detach().cpu().tolist(),
            })
    payload = {
        "schema_version": "trainable_decision_boundary_planner_v2",
        "feature_names": list(FEATURE_NAMES),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "layers": layers,
        "base_residual_weight": args.base_residual_weight,
        "belief_calibration_json": (
            str(args.belief_calibration_json.resolve())
            if args.belief_calibration_json is not None else None
        ),
        "training": {
            "method": "AdamW listwise counterfactual policy fine-tuning",
            "seed": args.seed,
            "epochs": args.epochs,
            "hidden_width": args.hidden_width,
            "language_generator_frozen": True,
            "belief_model_frozen": True,
            "oracle_truth_available_at_inference": False,
        },
        "offline_metrics": metrics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--belief-calibration-json", type=Path)
    parser.add_argument("--train-cases-per-profile", type=int, default=72)
    parser.add_argument("--eval-cases-per-profile", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-residual-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.belief_calibration_json is not None and not args.belief_calibration_json.exists():
        raise FileNotFoundError(args.belief_calibration_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    train = generate_split("train", args.train_cases_per_profile, args.seed)
    dev = generate_split("dev", args.eval_cases_per_profile, args.seed + 101)
    test = generate_split("test", args.eval_cases_per_profile, args.seed + 202)
    train_x = torch.tensor(
        [row["features"] for case in train for row in case["candidates"]], dtype=torch.float32
    )
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-4)
    model = CandidateQNetwork(args.hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    best_state, best_regret, stale = None, math.inf, 0
    log_path = args.output_dir / "train_log.jsonl"
    for epoch in range(args.epochs):
        rng.shuffle(train)
        model.train()
        losses = []
        for start in range(0, len(train), args.batch_size):
            loss = batch_loss(model, train[start:start + args.batch_size], mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        dev_metrics = evaluate(model, dev, mean, std)
        event = {"epoch": epoch, "train_loss": statistics.mean(losses), "dev": dev_metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if dev_metrics["mean_oracle_regret"] < best_regret - 1e-5:
            best_regret = dev_metrics["mean_oracle_regret"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No planner checkpoint selected.")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "train": evaluate(model, train, mean, std),
        "dev": evaluate(model, dev, mean, std),
        "test": evaluate(model, test, mean, std),
    }
    checkpoint = args.output_dir / "planner_v2_checkpoint.json"
    export_checkpoint(model, mean, std, checkpoint, args, metrics)
    torch.save(best_state, args.output_dir / "planner_v2_state.pt")
    for split, cases in (("train", train), ("dev", dev), ("test", test)):
        with (args.output_dir / f"{split}_contexts.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "planner_v2_counterfactual_dataset_v1",
        "feature_names": list(FEATURE_NAMES),
        "split_contexts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "split_unit": "hidden_profile_id",
        "profile_overlap": False,
        "leakage_guards": {
            "hidden_profile_used_only_for_oracle_label": True,
            "planner_inputs_use_belief_not_private_truth": True,
            "environment_units_normalized": True,
            "language_generator_frozen": True,
            "belief_model_frozen": True,
        },
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "offline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(checkpoint), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
