from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .belief import ReservationBelief


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


@dataclass
class OpponentConcessionModel:
    """Posterior over concession speed, separate from reservation belief.

    ANL 2024 reveals the opponent utility-function shape, but reservation value
    and behavior are distinct hidden variables. Planner v1 marginalized them
    inside one acceptance heuristic. V2 keeps a small behavior posterior so a
    fast conceder does not force the reservation posterior upward or make the
    planner concede early.
    """

    exponents: tuple[float, ...] = (0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0)
    observation_sigma: float = 0.045
    offer_temperature: float = 0.35
    rejection_temperature: float = 0.12
    weights: list[float] = field(init=False)
    first_offer_utility: float | None = None
    first_offer_time: float | None = None
    offer_history: list[dict[str, float]] = field(default_factory=list)
    rejection_history: list[dict[str, float]] = field(default_factory=list)
    _threshold_cache: dict[
        tuple[float, float], tuple[list[float], list[float]]
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.weights = [1.0 / len(self.exponents)] * len(self.exponents)

    def _replace_weights(self, raw: Iterable[float]) -> None:
        values = [max(float(value), 1e-12) for value in raw]
        total = sum(values)
        if not math.isfinite(total) or total <= 0.0:
            self.weights = [1.0 / len(self.exponents)] * len(self.exponents)
            return
        self.weights = [value / total for value in values]
        self._threshold_cache.clear()

    def observe_offer(
        self,
        *,
        opponent_utility: float,
        own_utility: float,
        relative_time: float,
        belief: ReservationBelief,
    ) -> None:
        utility = _clip(opponent_utility)
        time = _clip(relative_time)
        self.offer_history.append(
            {
                "relative_time": time,
                "opponent_utility": utility,
                "own_utility": _clip(own_utility),
            }
        )
        self._threshold_cache.clear()
        if self.first_offer_utility is None:
            self.first_offer_utility = utility
            self.first_offer_time = time
            return
        first = self.first_offer_utility
        first_time = float(self.first_offer_time or 0.0)
        progress = _clip((time - first_time) / max(1.0 - first_time, 1e-9))
        likelihoods: list[float] = []
        for exponent in self.exponents:
            likelihood = 0.0
            for rv, rv_weight in zip(belief.grid, belief.weights, strict=True):
                predicted = rv + (first - rv) * (1.0 - progress**exponent)
                error = (utility - predicted) / self.observation_sigma
                likelihood += rv_weight * math.exp(-0.5 * error * error)
            likelihoods.append(0.05 + 0.95 * likelihood)
        self._replace_weights(
            prior * likelihood**self.offer_temperature
            for prior, likelihood in zip(self.weights, likelihoods, strict=True)
        )

    def observe_rejection(
        self,
        *,
        opponent_utility: float,
        relative_time: float,
        belief: ReservationBelief,
        opponent_max: float,
    ) -> None:
        utility = _clip(opponent_utility, 0.0, max(1.0, opponent_max))
        time = _clip(relative_time)
        likelihoods: list[float] = []
        for exponent in self.exponents:
            likelihood = 0.0
            for rv, rv_weight in zip(belief.grid, belief.weights, strict=True):
                aspiration = rv + (opponent_max - rv) * (1.0 - time**exponent)
                p_accept = _sigmoid((utility - aspiration) / 0.035)
                likelihood += rv_weight * (1.0 - p_accept)
            likelihoods.append(0.05 + 0.95 * likelihood)
        self._replace_weights(
            prior * likelihood**self.rejection_temperature
            for prior, likelihood in zip(self.weights, likelihoods, strict=True)
        )
        self.rejection_history.append(
            {
                "relative_time": time,
                "opponent_utility": utility,
            }
        )

    def _threshold_distribution(
        self,
        *,
        relative_time: float,
        belief: ReservationBelief,
        opponent_max: float,
    ) -> tuple[list[float], list[float]]:
        key = (round(relative_time, 8), round(opponent_max, 8))
        cached = self._threshold_cache.get(key)
        if cached is not None:
            return cached
        points = sorted(
            (
                rv + (opponent_max - rv) * (1.0 - relative_time**exponent),
                exponent_weight * rv_weight,
            )
            for exponent, exponent_weight in zip(
                self.exponents, self.weights, strict=True
            )
            for rv, rv_weight in zip(belief.grid, belief.weights, strict=True)
        )
        thresholds = [point for point, _ in points]
        cumulative: list[float] = []
        total = 0.0
        for _, weight in points:
            total += weight
            cumulative.append(total)
        result = thresholds, cumulative
        self._threshold_cache[key] = result
        return result

    def acceptance_probability(
        self,
        *,
        opponent_utility: float,
        relative_time: float,
        belief: ReservationBelief,
        opponent_max: float,
    ) -> float:
        utility = _clip(opponent_utility, 0.0, max(1.0, opponent_max))
        time = _clip(relative_time)
        thresholds, cumulative = self._threshold_distribution(
            relative_time=time,
            belief=belief,
            opponent_max=opponent_max,
        )

        def threshold_cdf(value: float) -> float:
            index = bisect_right(thresholds, value) - 1
            return cumulative[index] if index >= 0 else 0.0

        # Five-point smoothing approximates the logistic response noise while
        # reducing candidate scoring from O(outcomes × RV-grid × exponent-grid)
        # to O(RV-grid × exponent-grid log n + outcomes log n).
        tau = 0.035
        probability = (
            0.10 * threshold_cdf(utility - 2.0 * tau)
            + 0.20 * threshold_cdf(utility - tau)
            + 0.40 * threshold_cdf(utility)
            + 0.20 * threshold_cdf(utility + tau)
            + 0.10 * threshold_cdf(utility + 2.0 * tau)
        )
        return _clip(probability)

    @property
    def mean_exponent(self) -> float:
        return sum(
            exponent * weight
            for exponent, weight in zip(self.exponents, self.weights, strict=True)
        )

    @property
    def normalized_entropy(self) -> float:
        entropy = -sum(weight * math.log(max(weight, 1e-15)) for weight in self.weights)
        return entropy / math.log(len(self.weights))

    @property
    def observed_time_step(self) -> float:
        if len(self.offer_history) < 2:
            return 0.02
        recent = self.offer_history[-5:]
        deltas = [
            current["relative_time"] - previous["relative_time"]
            for previous, current in zip(recent, recent[1:])
            if current["relative_time"] > previous["relative_time"]
        ]
        return sum(deltas) / len(deltas) if deltas else 0.02

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "opponent_concession_posterior",
            "mean_exponent": round(self.mean_exponent, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "offer_observations": len(self.offer_history),
            "rejection_observations": len(self.rejection_history),
            "exponent_posterior": {
                str(exponent): round(weight, 6)
                for exponent, weight in zip(
                    self.exponents, self.weights, strict=True
                )
            },
        }


@dataclass(frozen=True)
class CandidateScoreV2:
    outcome: Any
    own_utility: float
    own_surplus_norm: float
    opponent_utility: float
    opponent_rational_probability: float
    acceptance_probability_now: float
    acceptance_probability_future: float
    wait_gain: float
    own_aspiration: float
    aspiration_shortfall: float
    rational_probability_floor: float
    rational_shortfall: float
    information_gain: float
    planner_score: float

    @property
    def acceptance_probability(self) -> float:
        """Compatibility field used by the shared trace and agent adapter."""

        return self.acceptance_probability_now

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["outcome"] = list(self.outcome) if self.outcome is not None else None
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in row.items()
        }


class BeliefUsableRVPlannerV2:
    """Planner 2.0: RV belief + behavior belief + non-myopic concession.

    The v1 planner maximized one-step expected surplus and could purchase early
    agreement by offering too much. V2 keeps a Boulware-style own aspiration,
    treats risky opponent-rationality as tolerable early but not near the
    deadline, and separately predicts the opponent's concession speed.
    """

    version = "2.0"

    def __init__(
        self,
        *,
        own_concession_exponent: float = 5.0,
        information_gain_weight: float = 0.025,
        candidate_trace_k: int = 8,
    ) -> None:
        self.own_concession_exponent = float(own_concession_exponent)
        self.information_gain_weight = float(information_gain_weight)
        self.candidate_trace_k = int(candidate_trace_k)
        self.concession_model = OpponentConcessionModel()
        self._opponent_max = 1.0

    def observe_opponent_offer(
        self,
        *,
        opponent_utility: float,
        own_utility: float,
        relative_time: float,
        belief: ReservationBelief,
        opponent_max: float,
    ) -> None:
        self._opponent_max = max(float(opponent_max), 1e-9)
        self.concession_model.observe_offer(
            opponent_utility=opponent_utility,
            own_utility=own_utility,
            relative_time=relative_time,
            belief=belief,
        )

    def observe_rejection(
        self,
        *,
        opponent_utility: float,
        relative_time: float,
        belief: ReservationBelief,
        opponent_max: float,
    ) -> None:
        self._opponent_max = max(float(opponent_max), 1e-9)
        self.concession_model.observe_rejection(
            opponent_utility=opponent_utility,
            relative_time=relative_time,
            belief=belief,
            opponent_max=self._opponent_max,
        )

    def _own_aspiration(self, own_ufun: Any, relative_time: float) -> float:
        time = _clip(relative_time)
        reservation = _safe_float(own_ufun.reserved_value)
        maximum = max(_safe_float(own_ufun.max(), 1.0), reservation + 1e-9)
        return reservation + (maximum - reservation) * (
            1.0 - time**self.own_concession_exponent
        )

    def rank(
        self,
        *,
        outcomes: Iterable[Any],
        own_ufun: Any,
        opponent_ufun: Any,
        belief: ReservationBelief,
        relative_time: float,
    ) -> list[CandidateScoreV2]:
        time = _clip(relative_time)
        own_reservation = _safe_float(own_ufun.reserved_value)
        own_max = max(_safe_float(own_ufun.max(), 1.0), own_reservation + 1e-9)
        own_span = max(own_max - own_reservation, 1e-9)
        opponent_max = max(_safe_float(opponent_ufun.max(), 1.0), 1e-9)
        self._opponent_max = opponent_max
        own_aspiration = self._own_aspiration(own_ufun, time)

        # Early risky offers are cheap because rejection leaves future rounds.
        # Near the deadline, insist that the offer is likely individually
        # rational for the opponent.
        rational_floor = 0.05 + 0.75 * time**1.7
        observed_step = self.concession_model.observed_time_step
        future_time = min(1.0, time + max(0.04, 2.0 * observed_step))
        accept_weight = 0.04 + 0.96 * time**1.6

        rows: list[CandidateScoreV2] = []
        for outcome in outcomes:
            own = _safe_float(own_ufun(outcome))
            if own + 1e-12 < own_reservation:
                continue
            opponent = _safe_float(opponent_ufun(outcome))
            own_surplus_norm = max(0.0, (own - own_reservation) / own_span)
            rational_probability = belief.cdf(opponent)
            accept_now = self.concession_model.acceptance_probability(
                opponent_utility=opponent,
                relative_time=time,
                belief=belief,
                opponent_max=opponent_max,
            )
            accept_future = self.concession_model.acceptance_probability(
                opponent_utility=opponent,
                relative_time=future_time,
                belief=belief,
                opponent_max=opponent_max,
            )
            wait_gain = max(0.0, accept_future - accept_now)
            aspiration_shortfall = max(0.0, (own_aspiration - own) / own_span)
            rational_shortfall = max(0.0, rational_floor - rational_probability)
            information_gain = belief.expected_information_gain(opponent, time)

            # Own surplus is the anchor. Acceptance becomes important only as
            # time runs out. A large predicted gain from waiting penalizes an
            # unnecessarily generous early proposal.
            planner_score = (
                own_surplus_norm
                + accept_weight * accept_now
                + 0.06 * rational_probability
                + self.information_gain_weight * information_gain * (1.0 - time)
                - 2.50 * aspiration_shortfall
                - 1.50 * rational_shortfall
                - 0.20 * (1.0 - time) * wait_gain * (1.0 - own_surplus_norm)
            )
            rows.append(
                CandidateScoreV2(
                    outcome=outcome,
                    own_utility=own,
                    own_surplus_norm=own_surplus_norm,
                    opponent_utility=opponent,
                    opponent_rational_probability=rational_probability,
                    acceptance_probability_now=accept_now,
                    acceptance_probability_future=accept_future,
                    wait_gain=wait_gain,
                    own_aspiration=own_aspiration,
                    aspiration_shortfall=aspiration_shortfall,
                    rational_probability_floor=rational_floor,
                    rational_shortfall=rational_shortfall,
                    information_gain=information_gain,
                    planner_score=planner_score,
                )
            )
        rows.sort(
            key=lambda row: (
                row.planner_score,
                row.own_utility,
                row.acceptance_probability_now,
            ),
            reverse=True,
        )
        return rows

    def choose(self, **kwargs: Any) -> tuple[CandidateScoreV2, list[CandidateScoreV2]]:
        ranked = self.rank(**kwargs)
        if not ranked:
            raise RuntimeError("No individually rational candidate is available")
        return ranked[0], ranked[: self.candidate_trace_k]

    def should_accept(
        self,
        *,
        received_offer: Any,
        planned: CandidateScoreV2,
        own_ufun: Any,
        relative_time: float,
    ) -> tuple[bool, dict[str, float]]:
        time = _clip(relative_time)
        offered = _safe_float(own_ufun(received_offer))
        reservation = _safe_float(own_ufun.reserved_value)
        aspiration = self._own_aspiration(own_ufun, time)

        # AC-time/AC-next hybrid. The aspiration prevents the myopic early
        # accepts seen in v1; close to the deadline, a planned offer that is
        # already below aspiration provides a safe fallback.
        ac_next = planned.own_utility - 0.005 * time
        threshold = max(reservation, min(aspiration, ac_next))
        return offered + 1e-12 >= threshold, {
            "offered_utility": offered,
            "own_aspiration": aspiration,
            "ac_next": ac_next,
            "accept_threshold": threshold,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "planner_version": self.version,
            "own_concession_exponent": self.own_concession_exponent,
            "information_gain_weight": self.information_gain_weight,
            "concession_model": self.concession_model.to_json(),
        }
