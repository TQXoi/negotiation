"""Deployable behavior posterior and one-change Bayesian filter.

Unlike the evaluator-side profile oracle, these classes never receive the
ground-truth active profile.  They update only from public seller outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .oracle import OUTCOMES, SellerBehaviorModel


def _normalize(weights: Sequence[float]) -> list[float]:
    total = sum(max(0.0, float(value)) for value in weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, float(value)) / total for value in weights]


def _entropy(weights: Sequence[float]) -> float:
    return -sum(weight * math.log2(weight) for weight in weights if weight > 0)


def normalize_public_outcome(outcome: str) -> str:
    key = str(outcome).strip().lower()
    mapping = {
        "accept": "accept",
        "deal": "accept",
        "offer": "counter",
        "sell": "counter",
        "counter": "counter",
        "reject": "counter",
        "quit": "quit",
        "walk_away": "quit",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unsupported public seller outcome: {outcome}") from exc


@dataclass(frozen=True)
class BehaviorHypothesis:
    hypothesis_id: str
    model: SellerBehaviorModel


def default_behavior_hypotheses(reference_price: float) -> list[BehaviorHypothesis]:
    """A fixed, truth-agnostic hypothesis library for the deployed buyer."""

    rows: list[BehaviorHypothesis] = []
    policy_types = [
        ("patient", 0.92, 0.02, 0.20),
        ("neutral", 0.78, 0.08, 0.25),
        ("firm", 0.48, 0.28, 0.75),
    ]
    for cost_ratio in (0.45, 0.55, 0.65, 0.75, 0.85):
        for policy_name, counter, quit_bias, finality in policy_types:
            rows.append(
                BehaviorHypothesis(
                    hypothesis_id=f"cost_{cost_ratio:.2f}_{policy_name}",
                    model=SellerBehaviorModel(
                        reservation_price=reference_price * cost_ratio,
                        rationality_tau=max(0.25, 0.035 * reference_price),
                        counter_propensity=counter,
                        quit_bias=quit_bias,
                        finality=finality,
                    ),
                )
            )
    return rows


@dataclass(frozen=True)
class ChangePointUpdate:
    outcome: str
    change_probability: float
    predictive_probability: float
    expected_reservation: float
    profile_entropy_bits: float


class ChangePointBehaviorBelief:
    """Exact Bayesian filter for an absorbing old→new hidden regime.

    State mass is maintained over ``(old/new, behavior hypothesis)``.  At each
    step, an old regime changes with ``hazard`` and draws a fresh behavior
    hypothesis from a fixed diffuse prior.  The new regime is absorbing.  This
    gives a concrete old/new mixture without revealing the evaluator's active
    profile or resetting to an unscored natural-language guess.
    """

    def __init__(
        self,
        hypotheses: Sequence[BehaviorHypothesis],
        *,
        hazard: float = 0.03,
        prior_weights: Sequence[float] | None = None,
    ):
        if not hypotheses:
            raise ValueError("at least one behavior hypothesis is required")
        self.hypotheses = list(hypotheses)
        self.hazard = max(0.0, min(1.0, float(hazard)))
        self.base_prior = _normalize(prior_weights or [1.0] * len(hypotheses))
        self.old_mass = list(self.base_prior)
        self.new_mass = [0.0] * len(hypotheses)
        self.history: list[ChangePointUpdate] = []

    @property
    def change_probability(self) -> float:
        return sum(self.new_mass)

    def state_components(self) -> list[tuple[str, str, float, SellerBehaviorModel]]:
        rows = []
        for hypothesis, weight in zip(self.hypotheses, self.old_mass):
            if weight > 0:
                rows.append(("old", hypothesis.hypothesis_id, weight, hypothesis.model))
        for hypothesis, weight in zip(self.hypotheses, self.new_mass):
            if weight > 0:
                rows.append(("new", hypothesis.hypothesis_id, weight, hypothesis.model))
        return rows

    def profile_weights(self) -> list[float]:
        return [old + new for old, new in zip(self.old_mass, self.new_mass)]

    def expected_reservation(self) -> float:
        return sum(
            weight * hypothesis.model.reservation_price
            for weight, hypothesis in zip(self.profile_weights(), self.hypotheses)
        )

    def profile_entropy(self) -> float:
        return _entropy(self.profile_weights())

    def response_distribution(
        self, offer: float, *, round_id: int = 1, max_turns: int = 6
    ) -> Mapping[str, float]:
        result = {key: 0.0 for key in OUTCOMES}
        for _, _, weight, model in self.state_components():
            response = model.response_distribution(offer, round_id=round_id, max_turns=max_turns)
            for key in OUTCOMES:
                result[key] += weight * response[key]
        return result

    def update(
        self,
        *,
        offer: float,
        outcome: str,
        round_id: int = 1,
        max_turns: int = 6,
    ) -> ChangePointUpdate:
        observed = normalize_public_outcome(outcome)
        old_total = sum(self.old_mass)
        predicted_old = [(1.0 - self.hazard) * value for value in self.old_mass]
        switched_total = self.hazard * old_total
        predicted_new = [
            existing + switched_total * prior
            for existing, prior in zip(self.new_mass, self.base_prior)
        ]
        old_likelihood = [
            hypothesis.model.response_distribution(
                offer, round_id=round_id, max_turns=max_turns
            )[observed]
            for hypothesis in self.hypotheses
        ]
        new_likelihood = old_likelihood
        posterior_old = [mass * likelihood for mass, likelihood in zip(predicted_old, old_likelihood)]
        posterior_new = [mass * likelihood for mass, likelihood in zip(predicted_new, new_likelihood)]
        predictive = sum(posterior_old) + sum(posterior_new)
        if predictive <= 1e-15:
            # Keep the filter finite under a misspecified hypothesis library.
            posterior_old = predicted_old
            posterior_new = predicted_new
            predictive = 1e-15
            norm = sum(posterior_old) + sum(posterior_new)
        else:
            norm = predictive
        self.old_mass = [value / norm for value in posterior_old]
        self.new_mass = [value / norm for value in posterior_new]
        row = ChangePointUpdate(
            outcome=observed,
            change_probability=self.change_probability,
            predictive_probability=predictive,
            expected_reservation=self.expected_reservation(),
            profile_entropy_bits=self.profile_entropy(),
        )
        self.history.append(row)
        return row

    def to_dict(self, *, top_k: int = 5) -> dict:
        ranked = sorted(
            (
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "probability": weight,
                    "reservation_price": hypothesis.model.reservation_price,
                }
                for hypothesis, weight in zip(self.hypotheses, self.profile_weights())
            ),
            key=lambda row: row["probability"],
            reverse=True,
        )
        return {
            "change_probability": self.change_probability,
            "expected_reservation": self.expected_reservation(),
            "profile_entropy_bits": self.profile_entropy(),
            "hazard": self.hazard,
            "observations": len(self.history),
            "top_hypotheses": ranked[:top_k],
        }


def expected_information_gain_bits(
    belief: ChangePointBehaviorBelief,
    *,
    offer: float,
    round_id: int = 1,
    max_turns: int = 6,
) -> float:
    """Mutual information I((regime, profile); response | offer)."""

    states = belief.state_components()
    prior_entropy = _entropy([row[2] for row in states])
    outcome_probs = {key: 0.0 for key in OUTCOMES}
    joint: dict[str, list[float]] = {key: [] for key in OUTCOMES}
    for _, _, weight, model in states:
        response = model.response_distribution(offer, round_id=round_id, max_turns=max_turns)
        for outcome in OUTCOMES:
            value = weight * response[outcome]
            joint[outcome].append(value)
            outcome_probs[outcome] += value
    expected_posterior_entropy = 0.0
    for outcome in OUTCOMES:
        probability = outcome_probs[outcome]
        if probability <= 0:
            continue
        posterior = [value / probability for value in joint[outcome]]
        expected_posterior_entropy += probability * _entropy(posterior)
    return max(0.0, prior_entropy - expected_posterior_entropy)
