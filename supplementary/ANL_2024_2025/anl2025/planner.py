from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .belief import EdgePreferenceBelief


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class GlobalCandidateScore:
    outcome: Any
    center_utility_if_accept: float
    center_utility_if_reject: float
    immediate_marginal_utility: float
    expected_final_utility: float
    future_value_if_accept: float
    edge_preference_score: float
    edge_acceptance_probability: float
    information_gain: float
    planner_score: float

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["outcome"] = list(self.outcome) if self.outcome is not None else None
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in row.items()
        }


class GlobalContinuationPlanner:
    """Belief-conditioned cross-thread planner for sequential multi-deals."""

    version = "1.1"

    def __init__(
        self,
        *,
        center_ufun: Any,
        use_future_value: bool = True,
        information_gain_weight: float = 0.02,
        future_acceptance_time: float = 0.8,
        future_state_beam: int = 16,
        candidate_trace_k: int = 8,
    ) -> None:
        self.center_ufun = center_ufun
        self.use_future_value = bool(use_future_value)
        self.information_gain_weight = float(information_gain_weight)
        self.future_acceptance_time = _clip(future_acceptance_time)
        self.future_state_beam = max(1, int(future_state_beam))
        self.candidate_trace_k = max(1, int(candidate_trace_k))

    def evaluate(self, agreements: list[Any | None]) -> float:
        separated = tuple(agreements)
        try:
            return float(self.center_ufun.eval(separated))
        except Exception:
            return float(self.center_ufun(separated, use_expected=False))

    def _future_value(
        self,
        *,
        thread_index: int,
        current_outcome: Any | None,
        agreements: list[Any | None],
        completed: list[bool],
        outcomes_by_thread: list[tuple[Any, ...]],
        beliefs: list[EdgePreferenceBelief],
    ) -> float:
        initial = list(agreements)
        initial[thread_index] = current_outcome
        if not self.use_future_value:
            return self.evaluate(initial)

        # A small probability beam represents accept/reject branches for
        # remaining edges. Each state selects the deal with highest one-step
        # expected global utility, then carries both branches forward so
        # complementarities in the center utility remain visible.
        states: list[tuple[float, list[Any | None]]] = [(1.0, initial)]
        for future_index in range(thread_index + 1, len(initial)):
            if completed[future_index]:
                continue
            expanded: list[tuple[float, list[Any | None]]] = []
            for state_probability, state_outcomes in states:
                no_deal = list(state_outcomes)
                no_deal[future_index] = None
                no_value = self.evaluate(no_deal)
                best_outcome = None
                best_probability = 0.0
                best_expected = no_value
                for outcome in outcomes_by_thread[future_index]:
                    yes_deal = list(no_deal)
                    yes_deal[future_index] = outcome
                    yes_value = self.evaluate(yes_deal)
                    p_accept = beliefs[future_index].acceptance_probability(
                        outcome, self.future_acceptance_time
                    )
                    expected = p_accept * yes_value + (1.0 - p_accept) * no_value
                    if expected > best_expected:
                        best_expected = expected
                        best_outcome = outcome
                        best_probability = p_accept
                if best_outcome is None or best_probability <= 1e-9:
                    expanded.append((state_probability, no_deal))
                    continue
                accepted = list(no_deal)
                accepted[future_index] = best_outcome
                expanded.append((state_probability * best_probability, accepted))
                expanded.append(
                    (state_probability * (1.0 - best_probability), no_deal)
                )
            expanded.sort(key=lambda item: item[0], reverse=True)
            states = expanded[: self.future_state_beam]
            mass = sum(probability for probability, _ in states)
            if mass > 1e-12:
                states = [
                    (probability / mass, state)
                    for probability, state in states
                ]
        return sum(
            probability * self.evaluate(state)
            for probability, state in states
        )

    def rank(
        self,
        *,
        thread_index: int,
        relative_time: float,
        agreements: list[Any | None],
        completed: list[bool],
        outcomes_by_thread: list[tuple[Any, ...]],
        beliefs: list[EdgePreferenceBelief],
    ) -> list[GlobalCandidateScore]:
        time = _clip(relative_time)
        reject_vector = list(agreements)
        reject_vector[thread_index] = None
        immediate_reject = self.evaluate(reject_vector)
        reject_value = self._future_value(
            thread_index=thread_index,
            current_outcome=None,
            agreements=agreements,
            completed=completed,
            outcomes_by_thread=outcomes_by_thread,
            beliefs=beliefs,
        )
        belief = beliefs[thread_index]
        rows: list[GlobalCandidateScore] = []
        for outcome in outcomes_by_thread[thread_index]:
            accept_vector = list(agreements)
            accept_vector[thread_index] = outcome
            immediate_accept = self.evaluate(accept_vector)
            accept_value = self._future_value(
                thread_index=thread_index,
                current_outcome=outcome,
                agreements=agreements,
                completed=completed,
                outcomes_by_thread=outcomes_by_thread,
                beliefs=beliefs,
            )
            p_accept = belief.acceptance_probability(outcome, time)
            information_gain = belief.expected_information_gain(outcome, time)
            expected_final = p_accept * accept_value + (1.0 - p_accept) * reject_value
            planner_score = (
                expected_final
                + self.information_gain_weight * information_gain * (1.0 - time)
                + 0.002 * p_accept * time
            )
            rows.append(
                GlobalCandidateScore(
                    outcome=outcome,
                    center_utility_if_accept=accept_value,
                    center_utility_if_reject=reject_value,
                    immediate_marginal_utility=immediate_accept - immediate_reject,
                    expected_final_utility=expected_final,
                    future_value_if_accept=accept_value - immediate_accept,
                    edge_preference_score=belief.preference_score(outcome),
                    edge_acceptance_probability=p_accept,
                    information_gain=information_gain,
                    planner_score=planner_score,
                )
            )
        rows.sort(
            key=lambda row: (
                row.planner_score,
                row.center_utility_if_accept,
                row.edge_acceptance_probability,
            ),
            reverse=True,
        )
        return rows

    def choose(self, **kwargs: Any) -> tuple[GlobalCandidateScore, list[GlobalCandidateScore]]:
        ranked = self.rank(**kwargs)
        if not ranked:
            raise RuntimeError("No ANL 2025 candidate outcomes are available")
        return ranked[0], ranked[: self.candidate_trace_k]

    def should_accept(
        self,
        *,
        received_offer: Any,
        planned: GlobalCandidateScore,
        thread_index: int,
        relative_time: float,
        agreements: list[Any | None],
        completed: list[bool],
        outcomes_by_thread: list[tuple[Any, ...]],
        beliefs: list[EdgePreferenceBelief],
    ) -> tuple[bool, dict[str, float]]:
        offered_value = self._future_value(
            thread_index=thread_index,
            current_outcome=received_offer,
            agreements=agreements,
            completed=completed,
            outcomes_by_thread=outcomes_by_thread,
            beliefs=beliefs,
        )
        time = _clip(relative_time)
        # The probability-discounted proposal score is deliberately *not* an
        # acceptance aspiration: using it here makes a low-value incoming deal
        # look attractive merely because our preferred deal may be rejected.
        # Start from the chosen deal's certain continuation value and concede
        # toward the best reject/no-deal continuation only near the deadline.
        continuation_floor = max(
            planned.center_utility_if_reject,
            min(planned.expected_final_utility, planned.center_utility_if_accept),
        )
        target = max(planned.center_utility_if_accept, continuation_floor)
        deadline_slack = 0.005 * time**4
        threshold = (
            continuation_floor
            + (target - continuation_floor) * (1.0 - time**2)
            - deadline_slack
        )
        accepted = offered_value + 1e-12 >= threshold
        return accepted, {
            "offered_expected_final_utility": offered_value,
            "planned_expected_final_utility": planned.expected_final_utility,
            "planned_certain_continuation_utility": target,
            "reject_continuation_floor": continuation_floor,
            "accept_threshold": threshold,
            "deadline_slack": deadline_slack,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "planner_version": self.version,
            "use_future_value": self.use_future_value,
            "information_gain_weight": self.information_gain_weight,
            "future_acceptance_time": self.future_acceptance_time,
            "future_state_beam": self.future_state_beam,
        }
