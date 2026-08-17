"""Oracle-verified hidden utility profile suite for repeated Simple Env."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .oracle import SellerBehaviorModel, categorical_js_divergence


@dataclass(frozen=True)
class UtilityProfile:
    profile_id: str
    cost_ratio: float
    rationality_tau_ratio: float = 0.035
    counter_propensity: float = 0.78
    quit_bias: float = 0.08
    finality: float = 0.25

    def behavior(self, reference_price: float) -> SellerBehaviorModel:
        return SellerBehaviorModel(
            reservation_price=self.cost_ratio * reference_price,
            rationality_tau=max(0.25, self.rationality_tau_ratio * reference_price),
            counter_propensity=self.counter_propensity,
            quit_bias=self.quit_bias,
            finality=self.finality,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UtilityProfilePair:
    pair_id: str
    old: UtilityProfile
    new: UtilityProfile
    best_probe_ratio: float
    oracle_js_bits: float
    identifiable: bool

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "old": self.old.to_dict(),
            "new": self.new.to_dict(),
            "best_probe_ratio": self.best_probe_ratio,
            "oracle_js_bits": self.oracle_js_bits,
            "identifiable": self.identifiable,
        }


def verify_pair(
    pair_id: str,
    old: UtilityProfile,
    new: UtilityProfile,
    *,
    reference_price: float = 100.0,
    buyer_budget_ratio: float = 0.95,
    offer_ratios: Iterable[float] | None = None,
    min_js_bits: float = 0.12,
) -> UtilityProfilePair:
    ratios = list(offer_ratios or [index / 100.0 for index in range(30, 96, 2)])
    old_model = old.behavior(reference_price)
    new_model = new.behavior(reference_price)
    feasible = [ratio for ratio in ratios if ratio <= buyer_budget_ratio]
    scored = [
        (
            categorical_js_divergence(
                old_model.response_distribution(ratio * reference_price),
                new_model.response_distribution(ratio * reference_price),
            ),
            ratio,
        )
        for ratio in feasible
    ]
    divergence, ratio = max(scored)
    return UtilityProfilePair(
        pair_id=pair_id,
        old=old,
        new=new,
        best_probe_ratio=ratio,
        oracle_js_bits=divergence,
        identifiable=divergence >= min_js_bits,
    )


def build_profile_suite(*, min_js_bits: float = 0.12) -> list[UtilityProfilePair]:
    """Build diverse, nontrivial pairs and reject behaviorally silent changes."""

    specs = [
        ("cost_up_large", UtilityProfile("cost_055", 0.55), UtilityProfile("cost_080", 0.80)),
        ("cost_down_large", UtilityProfile("cost_080", 0.80), UtilityProfile("cost_055", 0.55)),
        (
            "cost_up_firm",
            UtilityProfile("flexible_060", 0.60, counter_propensity=0.88, quit_bias=0.04),
            UtilityProfile("firm_078", 0.78, counter_propensity=0.55, quit_bias=0.22, finality=0.70),
        ),
        (
            "policy_shift_same_cost",
            UtilityProfile("patient_068", 0.68, counter_propensity=0.92, quit_bias=0.02),
            UtilityProfile("impatient_068", 0.68, counter_propensity=0.42, quit_bias=0.32, finality=0.80),
        ),
    ]
    pairs = [verify_pair(pair_id, old, new, min_js_bits=min_js_bits) for pair_id, old, new in specs]
    return [pair for pair in pairs if pair.identifiable]
