from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .belief import ReservationBelief


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result


@dataclass(frozen=True)
class CandidateScore:
    outcome: Any
    own_utility: float
    own_surplus_norm: float
    opponent_utility: float
    opponent_rational_probability: float
    acceptance_probability: float
    expected_surplus: float
    information_gain: float
    aspiration_penalty: float
    disagreement_risk: float
    planner_score: float

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["outcome"] = list(self.outcome) if self.outcome is not None else None
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in row.items()
        }


class BeliefUsableRVPlanner:
    """Counterfactual offer planner that consumes the RV posterior."""

    def __init__(
        self,
        *,
        information_gain_weight: float = 0.08,
        aspiration_exponent: float = 4.0,
        candidate_trace_k: int = 8,
    ) -> None:
        self.information_gain_weight = float(information_gain_weight)
        self.aspiration_exponent = float(aspiration_exponent)
        self.candidate_trace_k = int(candidate_trace_k)

    def rank(
        self,
        *,
        outcomes: Iterable[Any],
        own_ufun: Any,
        opponent_ufun: Any,
        belief: ReservationBelief,
        relative_time: float,
    ) -> list[CandidateScore]:
        time = max(0.0, min(1.0, float(relative_time)))
        own_reservation = _safe_float(own_ufun.reserved_value)
        own_max = max(_safe_float(own_ufun.max(), 1.0), own_reservation + 1e-9)
        own_span = max(own_max - own_reservation, 1e-9)
        aspiration = own_reservation + own_span * (1.0 - time**self.aspiration_exponent)
        rows: list[CandidateScore] = []
        for outcome in outcomes:
            own = _safe_float(own_ufun(outcome))
            if own + 1e-12 < own_reservation:
                continue
            opponent = _safe_float(opponent_ufun(outcome))
            opponent_rational = belief.cdf(opponent)
            accept_probability = belief.acceptance_probability(opponent, time)
            own_surplus_norm = max(0.0, (own - own_reservation) / own_span)
            expected_surplus = own_surplus_norm * accept_probability
            information_gain = belief.expected_information_gain(opponent, time)
            aspiration_penalty = max(0.0, (aspiration - own) / own_span)
            disagreement_risk = 1.0 - accept_probability
            planner_score = (
                expected_surplus
                + 0.07 * accept_probability
                + self.information_gain_weight * information_gain * (1.0 - time)
                + 0.03 * opponent
                + 0.03 * opponent_rational
                - 0.20 * aspiration_penalty
            )
            rows.append(
                CandidateScore(
                    outcome=outcome,
                    own_utility=own,
                    own_surplus_norm=own_surplus_norm,
                    opponent_utility=opponent,
                    opponent_rational_probability=opponent_rational,
                    acceptance_probability=accept_probability,
                    expected_surplus=expected_surplus,
                    information_gain=information_gain,
                    aspiration_penalty=aspiration_penalty,
                    disagreement_risk=disagreement_risk,
                    planner_score=planner_score,
                )
            )
        rows.sort(
            key=lambda row: (row.planner_score, row.own_utility, row.acceptance_probability),
            reverse=True,
        )
        return rows

    def choose(self, **kwargs: Any) -> tuple[CandidateScore, list[CandidateScore]]:
        ranked = self.rank(**kwargs)
        if not ranked:
            raise RuntimeError("No individually rational candidate is available")
        return ranked[0], ranked[: self.candidate_trace_k]

    def should_accept(
        self,
        *,
        received_offer: Any,
        planned: CandidateScore,
        own_ufun: Any,
        relative_time: float,
    ) -> tuple[bool, dict[str, float]]:
        time = max(0.0, min(1.0, float(relative_time)))
        offered = _safe_float(own_ufun(received_offer))
        reservation = _safe_float(own_ufun.reserved_value)
        continuation = (
            planned.acceptance_probability * planned.own_utility
            + (1.0 - planned.acceptance_probability) * reservation
        )
        deadline_slack = 0.025 * time
        threshold = max(reservation, continuation - deadline_slack)
        return offered + 1e-12 >= threshold, {
            "offered_utility": offered,
            "continuation_value": continuation,
            "accept_threshold": threshold,
            "deadline_slack": deadline_slack,
        }
