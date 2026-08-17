#!/usr/bin/env python3
"""Train a conservative cross-environment residual planner with offline AWR.

The dataset combines Simple/static, repeated-opponent, old/new change-point,
and AgenticPay-shaped multi-issue candidate sets. Hidden preferences are used
only to compute offline counterfactual returns. Runtime features contain only
public-history beliefs and normalized candidate/state attributes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
AGENTICPAY_ROOT = ROOT / "benchmarks" / "AgenticPay"
for path in (ROOT, AGENTICPAY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AgenticPay_Env.buyer.universal_framework import AgenticPayAdapter  # noqa: E402
from framework import (  # noqa: E402
    AWR_FEATURE_NAMES,
    BehavioralFrontierPlanner,
    CanonicalOffer,
    ConservativeAWRCheckpoint,
    OpponentBelief,
    PreferenceBelief,
    ResponsePolicyBelief,
    awr_candidate_features,
)
from Simple_Env.tools.train_decision_boundary_planner_v2 import (  # noqa: E402
    BUDGET,
    MAX_TURNS,
    REFERENCE,
    make_belief,
    make_candidates,
    make_state,
    offer_oracle_value,
    profile_suite,
    surplus_reward,
)


DEPLOYMENT = {
    "uncertainty_scale": 1.0,
    "minimum_override_lcb": 0.015,
    "initial_override_lcb": 0.035,
    "accept_override_lcb": 0.025,
    "continue_over_accept_lcb": 0.035,
    "ood_z_threshold": 3.5,
    "ood_penalty": 0.02,
    "max_initial_proxy_jump": 0.12,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _case_from_objects(
    *,
    context_id: str,
    cluster_id: str,
    domain: str,
    state,
    belief: OpponentBelief,
    candidates,
    q_values: list[float],
) -> dict[str, Any]:
    """Serialize one decision state into a base-relative training context.

    ``q_values`` may use private utility, but only inside this offline label
    builder.  The exported candidate feature vectors are produced by
    ``awr_candidate_features`` and therefore contain public-history belief and
    adapter-side attributes only.  This is the main truth-leakage boundary.
    """
    base_rows = BehavioralFrontierPlanner().rank(state, belief, candidates)
    base_winner = base_rows[0]
    base_index = next(
        index
        for index, row in enumerate(base_rows)
        if row.candidate.candidate_id == base_winner.candidate.candidate_id
    )
    base_q = float(q_values[next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.candidate_id == base_winner.candidate.candidate_id
    )])
    q_by_id = {candidate.candidate_id: float(q) for candidate, q in zip(candidates, q_values)}
    top = base_winner.score
    best_offer = max(
        (row.candidate.own_utility_normalized for row in base_rows if row.candidate.action_type == "offer"),
        default=0.0,
    )
    rows = []
    for index, row in enumerate(base_rows):
        q = q_by_id[row.candidate.candidate_id]
        rows.append({
            "candidate_id": row.candidate.candidate_id,
            "action_type": row.candidate.action_type,
            "features": awr_candidate_features(
                state,
                belief,
                row,
                base_winner=base_winner,
                base_top_score=top,
                best_offer_utility=best_offer,
            ),
            "oracle_q": q,
            "advantage": q - base_q,
            "base_score": row.diagnostics.get("base_pre_awr_score", row.score),
            "is_base_action": index == base_index,
            "direct_response_count": belief.direct_response_count,
            "opponent_proxy": row.candidate.opponent_value_proxy,
        })
    return {
        "context_id": context_id,
        "cluster_id": cluster_id,
        "domain": domain,
        "base_candidate_id": base_winner.candidate.candidate_id,
        "base_oracle_q": base_q,
        "belief_only_inputs": True,
        "candidates": rows,
    }


def _simple_q_values(profile, state, candidates) -> list[float]:
    proxies = sorted(
        candidate.opponent_value_proxy for candidate in candidates if candidate.action_type == "offer"
    )
    values = []
    for candidate in candidates:
        if candidate.action_type == "offer":
            q = offer_oracle_value(
                profile,
                candidate.opponent_value_proxy,
                state.progress,
                proxies,
                depth=2,
            )
            q += min(0.02, 0.04 * (1.0 - state.progress) * candidate.information_gain)
        elif candidate.action_type == "accept" and candidate.offer is not None:
            q = surplus_reward(profile, float(candidate.offer.price) / BUDGET)
        else:
            q = 0.0
        values.append(float(_clamp(q)))
    return values


def _simple_context(profile, case_id: int, rng: random.Random, domain: str, peer=None):
    turn = rng.randint(1, MAX_TURNS)
    if domain == "simple_static":
        evidence = 0 if turn == 1 else rng.randint(1, min(4, turn - 1))
        belief = make_belief(profile, turn, evidence, rng)
        session_progress = 0.0
    else:
        episode = rng.randint(2, 20)
        evidence = rng.randint(1, min(6, episode - 1))
        belief = make_belief(profile, turn, evidence, rng)
        session_progress = (episode - 1) / 19.0
        if domain == "simple_repeated":
            belief.preference.reliability = min(0.96, belief.preference.reliability + 0.10)
            belief.response_policy.reliability = min(0.96, belief.response_policy.reliability + 0.10)
            belief.old_regime_weight = 0.95
            belief.new_regime_weight = 0.05
            belief.regime_change_probability = 0.05
        else:
            if peer is None:
                raise ValueError("change-point context requires an old/new peer profile")
            new_weight = rng.uniform(0.15, 0.90)
            old_ratio = peer.cost_ratio * REFERENCE / BUDGET
            current_ratio = profile.cost_ratio * REFERENCE / BUDGET
            noisy = (1.0 - new_weight) * old_ratio + new_weight * current_ratio + rng.gauss(0.0, 0.025)
            belief.preference.reservation_ratio_mean = _clamp(noisy, 0.05, 1.10)
            width = 0.10 + 0.20 * (1.0 - abs(new_weight - 0.5) * 2.0)
            belief.preference.reservation_ratio_low = max(0.05, noisy - width)
            belief.preference.reservation_ratio_high = min(1.15, noisy + width)
            belief.preference.reliability = 0.35 + 0.45 * abs(new_weight - 0.5) * 2.0
            belief.old_regime_weight = 1.0 - new_weight
            belief.new_regime_weight = new_weight
            belief.regime_change_probability = new_weight
    state = make_state(profile, case_id, turn)
    state.environment_id = "simple_env"
    state.metadata.update({
        "repeated_opponent": domain != "simple_static",
        "cross_session_progress": session_progress,
        "counterparty_count": 1,
    })
    candidates = make_candidates(state, belief, rng)
    q_values = _simple_q_values(profile, state, candidates)
    return _case_from_objects(
        context_id=f"{domain}:{profile.profile_id}:{case_id}",
        cluster_id=profile.profile_id,
        domain=domain,
        state=state,
        belief=belief,
        candidates=candidates,
        q_values=q_values,
    )


def _contract_template(template_id: int, rng: random.Random) -> dict[str, Any]:
    # Iteration 023 synthetic regularizer only.  Do not call these "official
    # AgenticPay templates": Iteration 024 extracts those from upstream task
    # source and marks them with a different data_source.
    scale = rng.uniform(35.0, 180.0)
    buyer = {
        "v_base": scale,
        "continuous_weights": {"delivery_days": -rng.uniform(0.02, 0.10) * scale},
        "discrete_weights": {
            "warranty": {"long": rng.uniform(0.04, 0.10) * scale, "short": -rng.uniform(0.01, 0.05) * scale},
            "packaging": {"protective": rng.uniform(0.02, 0.08) * scale, "standard": -rng.uniform(0.0, 0.03) * scale},
        },
    }
    seller = {
        "c_base": rng.uniform(0.52, 0.82) * scale,
        "continuous_weights": {"delivery_days": rng.uniform(0.015, 0.08) * scale},
        "discrete_weights": {
            "warranty": {"long": -rng.uniform(0.03, 0.10) * scale, "short": rng.uniform(0.01, 0.05) * scale},
            "packaging": {"protective": -rng.uniform(0.02, 0.07) * scale, "standard": rng.uniform(0.0, 0.04) * scale},
        },
    }
    return {
        "template_id": template_id,
        "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
        "discrete_options": {"warranty": ["long", "short"], "packaging": ["protective", "standard"]},
        "buyer_preferences": buyer,
        "seller_preferences": seller,
    }


def _seller_utility(config: dict[str, Any], offer: CanonicalOffer) -> float:
    prefs = config["seller_preferences"]
    utility = float(offer.price or 0.0) - float(prefs["c_base"])
    for issue, value in offer.continuous_terms.items():
        utility += float(prefs["continuous_weights"].get(issue, 0.0)) * float(value)
    for issue, value in offer.discrete_terms.items():
        utility += float(prefs["discrete_weights"].get(issue, {}).get(value, 0.0))
    return utility


def _contract_text(offer: CanonicalOffer) -> str:
    payload = offer.to_dict()
    payload.pop("allocations", None)
    return "<contract>\n" + json.dumps(payload, ensure_ascii=False) + "\n</contract>"


def _agenticpay_context(template: dict[str, Any], case_id: int, rng: random.Random):
    buyer_max = float(template["buyer_preferences"]["v_base"])
    turn = rng.randint(2, 6)
    seller_offer = CanonicalOffer(
        price=rng.uniform(0.72, 1.02) * buyer_max,
        continuous_terms={"delivery_days": rng.choice([4.0, 6.0, 7.0])},
        discrete_terms={
            "warranty": rng.choice(["long", "short"]),
            "packaging": rng.choice(["protective", "standard"]),
        },
    )
    context = {"contract_config": template, "max_price": buyer_max}
    history = [{"role": "seller", "content": _contract_text(seller_offer), "round": turn - 1}]
    adapter = AgenticPayAdapter(
        context=context,
        history=history,
        current_state={"current_round": turn, "max_rounds": 6},
        buyer_max_price=buyer_max,
        self_id="buyer",
        session_id=f"agenticpay:{template['template_id']}:{case_id}",
        counterparty_id="seller",
    )
    state = adapter.state("learned")
    state.metadata.update({
        "repeated_opponent": rng.random() < 0.35,
        "cross_session_progress": rng.random(),
        "counterparty_count": rng.choice([1, 2, 3]),
    })
    seller_floor_ratio = template["seller_preferences"]["c_base"] / buyer_max
    reliability = rng.uniform(0.25, 0.82)
    mean = _clamp(seller_floor_ratio + rng.gauss(0.0, 0.14 * (1.0 - reliability)), 0.10, 1.10)
    option_scores = {}
    for issue, weights in template["seller_preferences"]["discrete_weights"].items():
        minimum, maximum = min(weights.values()), max(weights.values())
        span = max(1e-6, maximum - minimum)
        option_scores[issue] = {
            json.dumps(option): _clamp((value - minimum) / span + rng.gauss(0, 0.08))
            for option, value in weights.items()
        }
    belief = OpponentBelief(
        counterparty_id="seller",
        preference=PreferenceBelief(
            reservation_ratio_low=max(0.05, mean - 0.25 * (1.0 - reliability)),
            reservation_ratio_mean=mean,
            reservation_ratio_high=min(1.15, mean + 0.25 * (1.0 - reliability)),
            issue_option_scores=option_scores,
            reliability=reliability,
        ),
        response_policy=ResponsePolicyBelief(
            counter_rate=rng.uniform(0.45, 0.85),
            quit_rate=rng.uniform(0.03, 0.25),
            patience=rng.uniform(0.35, 0.90),
            reliability=reliability,
            rejected_compatibility_max=rng.uniform(0.25, 0.68),
            aspiration_ratio_mean=_clamp(mean + rng.uniform(0.05, 0.25), 0.10, 1.15),
            aspiration_reliability=reliability,
        ),
        evidence_count=rng.randint(1, 5),
        direct_response_count=rng.randint(1, 4),
        old_regime_weight=1.0,
        new_regime_weight=0.0,
        latent_profile_entropy_bits=5.0 * (1.0 - reliability),
    )
    candidates = adapter.candidates(state, belief)
    own_best = max((candidate.own_utility_normalized for candidate in candidates), default=1.0)
    seller_scale = max(1.0, buyer_max)
    q_values = []
    for candidate in candidates:
        if candidate.action_type == "quit" or candidate.offer is None:
            q = 0.0
        elif candidate.action_type == "accept":
            q = _clamp(candidate.own_utility_normalized / max(0.05, own_best))
        else:
            seller_u = _seller_utility(template, candidate.offer) / seller_scale
            aspiration = 0.12 * (1.0 - state.progress)
            p_accept = 1.0 / (1.0 + math.exp(_clamp((aspiration - seller_u) / 0.035, -25, 25)))
            own = _clamp(candidate.own_utility_normalized / max(0.05, own_best))
            option = 0.035 * candidate.information_gain * (1.0 - state.progress)
            q = _clamp(p_accept * own + (1.0 - p_accept) * option)
        q_values.append(q)
    return _case_from_objects(
        context_id=f"agenticpay:{template['template_id']}:{case_id}",
        cluster_id=f"contract_{template['template_id']}",
        domain="agenticpay_multi_issue",
        state=state,
        belief=belief,
        candidates=candidates,
        q_values=q_values,
    )


def generate_split(split: str, contexts_per_cluster: int, seed: int) -> list[dict[str, Any]]:
    # Existing profile_suite definitions are disjoint by reservation/policy.
    profiles = profile_suite({"train": "train", "dev": "dev", "audit": "test"}[split])
    cases: list[dict[str, Any]] = []
    for profile_id, profile in enumerate(profiles):
        rng = random.Random(seed + profile_id * 1_000_003)
        peer = profiles[(profile_id + len(profiles) // 2) % len(profiles)]
        for case_id in range(contexts_per_cluster):
            domain = ("simple_static", "simple_repeated", "simple_change_point")[case_id % 3]
            case = _simple_context(profile, case_id, rng, domain, peer=peer)
            # Synthetic cases remain useful for preventing catastrophic
            # forgetting, but must not dominate real AgenticPay trajectory
            # anchors added through --extra-*-contexts.
            case.update({"data_source": "synthetic_regularizer", "training_weight": 0.5})
            cases.append(case)
    template_count = 12 if split == "train" else 6
    template_offset = {"train": 0, "dev": 100, "audit": 200}[split]
    for index in range(template_count):
        template_rng = random.Random(seed + 90_000_001 + index * 999_983)
        template = _contract_template(template_offset + index, template_rng)
        for case_id in range(contexts_per_cluster * 2):
            case = _agenticpay_context(template, case_id, template_rng)
            case.update({"data_source": "synthetic_regularizer", "training_weight": 0.5})
            cases.append(case)
    return cases


class AdvantageNetwork(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(AWR_FEATURE_NAMES), width), nn.Tanh(),
            nn.Linear(width, width // 2), nn.Tanh(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _case_tensors(case, mean, std):
    x = torch.tensor([row["features"] for row in case["candidates"]], dtype=torch.float32)
    advantage = torch.tensor([row["advantage"] for row in case["candidates"]], dtype=torch.float32)
    # Real trajectories can supply a logged-policy score.  Synthetic contexts
    # have no logged action distribution, so they intentionally fall back to
    # the frozen base planner score.
    behavior = torch.tensor(
        [row.get("behavior_score", row["base_score"]) for row in case["candidates"]],
        dtype=torch.float32,
    )
    base = torch.tensor([row["is_base_action"] for row in case["candidates"]], dtype=torch.bool)
    return (x - mean) / std, advantage, behavior, base


def _awr_loss(pred, advantage, behavior, base, beta: float):
    behavior_prob = torch.softmax(behavior / 0.12, dim=0)
    weights = behavior_prob * torch.exp(torch.clamp(advantage, min=-0.10, max=0.20) / beta)
    target_policy = weights / weights.sum().clamp_min(1e-8)
    policy_loss = -(target_policy.detach() * torch.log_softmax(pred / 0.10, dim=0)).sum()
    regression = F.smooth_l1_loss(pred, advantage)
    base_anchor = (pred[base] ** 2).mean()
    nonpositive = advantage <= 0.0
    false_positive = (
        F.relu(pred[nonpositive] + 0.005).pow(2).mean()
        if bool(nonpositive.any()) else pred.sum() * 0.0
    )
    return policy_loss + 0.75 * regression + 0.50 * base_anchor + 1.25 * false_positive


def _train_member(train, dev, mean, std, args, member_id: int):
    seed = args.seed + 10_007 * member_id
    torch.manual_seed(seed)
    rng = random.Random(seed)
    bootstrap = [rng.choice(train) for _ in range(len(train))]
    model = AdvantageNetwork(args.hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_loss, stale = None, math.inf, 0
    log = []
    dev_encoded = [_case_tensors(case, mean, std) for case in dev]
    for epoch in range(args.epochs):
        rng.shuffle(bootstrap)
        model.train()
        losses = []
        for start in range(0, len(bootstrap), args.batch_size):
            encoded = [
                _case_tensors(case, mean, std)
                for case in bootstrap[start:start + args.batch_size]
            ]
            lengths = [len(values[0]) for values in encoded]
            predictions = model(torch.cat([values[0] for values in encoded], dim=0))
            batch_losses, offset = [], 0
            for case, length, (_, advantage, behavior, base) in zip(
                bootstrap[start:start + args.batch_size], lengths, encoded
            ):
                # Source weights are applied at whole-decision granularity so
                # a context with many contract candidates cannot silently gain
                # more importance than a scalar-price context.
                batch_losses.append(
                    _awr_loss(
                        predictions[offset:offset + length],
                        advantage,
                        behavior,
                        base,
                        args.awr_beta,
                    ) * float(case.get("training_weight", 1.0))
                )
                offset += length
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            dev_x = torch.cat([values[0] for values in dev_encoded], dim=0)
            dev_predictions = model(dev_x)
            dev_losses, offset = [], 0
            for case, (x, advantage, behavior, base) in zip(dev, dev_encoded):
                length = len(x)
                dev_losses.append(float(_awr_loss(
                    dev_predictions[offset:offset + length],
                    advantage,
                    behavior,
                    base,
                    args.awr_beta,
                )) * float(case.get("training_weight", 1.0)))
                offset += length
            dev_loss = statistics.fmean(dev_losses)
        event = {"member": member_id, "epoch": epoch, "train_loss": statistics.fmean(losses), "dev_loss": dev_loss}
        log.append(event)
        print(json.dumps(event), flush=True)
        if dev_loss < best_loss - 1e-5:
            best_loss = dev_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("AWR member failed to select a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, log


def _layers(model):
    return [
        {"weight": module.weight.detach().cpu().tolist(), "bias": module.bias.detach().cpu().tolist()}
        for module in model.network if isinstance(module, nn.Linear)
    ]


def evaluate_checkpoint(checkpoint_path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = ConservativeAWRCheckpoint.load(checkpoint_path)
    rows = []
    for case in cases:
        candidates = case["candidates"]
        base = next(index for index, row in enumerate(candidates) if row["is_base_action"])
        selected = base
        best_lcb = 0.0
        for index, row in enumerate(candidates):
            if index == base:
                continue
            mean, std, z_rms = checkpoint.predict(row["features"])
            lcb = mean - checkpoint.uncertainty_scale * std
            lcb -= checkpoint.ood_penalty * max(0.0, z_rms - checkpoint.ood_z_threshold)
            threshold = checkpoint.minimum_override_lcb
            if row["direct_response_count"] <= 0:
                threshold = max(threshold, checkpoint.initial_override_lcb)
            if row["action_type"] == "accept":
                threshold = max(threshold, checkpoint.accept_override_lcb)
            if candidates[base]["action_type"] == "accept" and row["action_type"] != "accept":
                threshold = max(threshold, checkpoint.continue_over_accept_lcb)
            initial_jump_ok = not (
                row["direct_response_count"] <= 0
                and row["action_type"] == "offer"
                and row["opponent_proxy"]
                > candidates[base]["opponent_proxy"] + checkpoint.max_initial_proxy_jump
            )
            if lcb >= threshold and lcb > best_lcb and initial_jump_ok:
                selected, best_lcb = index, lcb
        delta = candidates[selected]["oracle_q"] - candidates[base]["oracle_q"]
        rows.append({
            "domain": case["domain"], "cluster_id": case["cluster_id"],
            "override": selected != base, "delta": delta,
            "harmful_override": selected != base and delta < -1e-8,
            "initial_flip": selected != base and candidates[base]["direct_response_count"] <= 0,
        })
    def summarize(group):
        clusters: dict[str, list[dict[str, Any]]] = {}
        for item in group:
            clusters.setdefault(item["cluster_id"], []).append(item)
        cluster_ids = sorted(clusters)
        rng = random.Random(20260816 + sum(ord(char) for char in "".join(cluster_ids)))
        boot = []
        if cluster_ids:
            for _ in range(2000):
                sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
                values = [row["delta"] for cluster_id in sampled for row in clusters[cluster_id]]
                boot.append(statistics.fmean(values))
        boot.sort()
        return {
            "contexts": len(group),
            "clusters": len(cluster_ids),
            "mean_oracle_delta_vs_base": statistics.fmean(row["delta"] for row in group),
            "paired_cluster_bootstrap_95_ci": (
                [boot[int(0.025 * len(boot))], boot[min(len(boot) - 1, int(0.975 * len(boot)))]]
                if boot else None
            ),
            "override_rate": statistics.fmean(float(row["override"]) for row in group),
            "harmful_override_rate": statistics.fmean(float(row["harmful_override"]) for row in group),
            "initial_action_flip_rate": statistics.fmean(float(row["initial_flip"]) for row in group),
        }
    domains = sorted({row["domain"] for row in rows})
    result = {"aggregate": summarize(rows), "by_domain": {domain: summarize([row for row in rows if row["domain"] == domain]) for domain in domains}}
    aggregate = result["aggregate"]
    result["gate"] = {
        "positive_aggregate_delta": aggregate["mean_oracle_delta_vs_base"] > 0.0,
        "no_domain_below_minus_0_005": all(value["mean_oracle_delta_vs_base"] >= -0.005 for value in result["by_domain"].values()),
        "harmful_override_rate_le_0_10": aggregate["harmful_override_rate"] <= 0.10,
        "nontrivial_override_rate": 0.02 <= aggregate["override_rate"] <= 0.60,
        "initial_flip_rate_le_0_25": aggregate["initial_action_flip_rate"] <= 0.25,
    }
    result["gate"]["passed"] = all(result["gate"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--contexts-jsonl", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--contexts-per-cluster", type=int, default=30)
    parser.add_argument("--extra-train-contexts", action="append", type=Path, default=[])
    parser.add_argument("--extra-dev-contexts", action="append", type=Path, default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--awr-beta", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--run-new-audit", action="store_true")
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        if args.contexts_jsonl is None:
            parser.error("--evaluate-checkpoint requires --contexts-jsonl")
        cases = [json.loads(line) for line in args.contexts_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = evaluate_checkpoint(args.evaluate_checkpoint, cases)
        if args.metrics_output is not None:
            args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    if args.output_dir is None:
        parser.error("training requires --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = generate_split("train", args.contexts_per_cluster, args.seed)
    dev = generate_split("dev", args.contexts_per_cluster, args.seed + 101)
    def load_extra(paths):
        rows = []
        for path in paths:
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return rows
    extra_train = load_extra(args.extra_train_contexts)
    extra_dev = load_extra(args.extra_dev_contexts)
    # Extra contexts are already split by the dedicated extractor.  Never
    # randomly repartition them here: doing so would leak turns from the same
    # AgenticPay scenario family across train and dev.
    train.extend(extra_train)
    dev.extend(extra_dev)
    train_x = torch.tensor([row["features"] for case in train for row in case["candidates"]], dtype=torch.float32)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-4)
    models, logs = [], []
    for member_id in range(args.ensemble_size):
        model, member_log = _train_member(train, dev, mean, std, args, member_id)
        models.append(model)
        logs.extend(member_log)
    checkpoint_path = args.output_dir / "cross_environment_awr_checkpoint.json"
    checkpoint = {
        "schema_version": "conservative_cross_environment_awr_v1",
        "feature_names": list(AWR_FEATURE_NAMES),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "ensemble": [{"member_id": index, "layers": _layers(model)} for index, model in enumerate(models)],
        "deployment": DEPLOYMENT,
        "training": {
            "method": "conservative offline AWR residual policy",
            "behavior_policy": "BehavioralFrontierPlanner",
            "seed": args.seed,
            "hidden_truth_at_inference": False,
            "language_model_frozen": True,
            "belief_model_frozen": True,
            "extra_train_contexts": [str(path.resolve()) for path in args.extra_train_contexts],
            "extra_dev_contexts": [str(path.resolve()) for path in args.extra_dev_contexts],
        },
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    dev_metrics = evaluate_checkpoint(checkpoint_path, dev)
    metrics = {"dev": dev_metrics}
    if args.run_new_audit:
        if not dev_metrics["gate"]["passed"]:
            metrics["audit"] = {"not_run": True, "reason": "development gate failed"}
        else:
            audit = generate_split("audit", args.contexts_per_cluster, args.seed + 20_260_816)
            metrics["audit"] = evaluate_checkpoint(checkpoint_path, audit)
            with (args.output_dir / "audit_contexts.jsonl").open("w", encoding="utf-8") as handle:
                for case in audit:
                    handle.write(json.dumps(case) + "\n")
    for split_name, cases in (("train", train), ("dev", dev)):
        with (args.output_dir / f"{split_name}_contexts.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")
    (args.output_dir / "train_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs), encoding="utf-8")
    manifest = {
        "schema_version": "cross_environment_awr_dataset_v1",
        "domains": ["simple_static", "simple_repeated", "simple_change_point", "agenticpay_multi_issue"],
        "contexts": {"train": len(train), "dev": len(dev)},
        "extra_contexts": {"train": len(extra_train), "dev": len(extra_dev)},
        "contexts_by_source": {
            split_name: {
                source: sum(case.get("data_source", "unknown") == source for case in cases)
                for source in sorted({case.get("data_source", "unknown") for case in cases})
            }
            for split_name, cases in (("train", train), ("dev", dev))
        },
        "split_unit": "utility profile or contract template cluster",
        "cluster_overlap": False,
        "old_planner_v2_final_split_reused": False,
        "hidden_truth_only_for_oracle_labels": True,
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
