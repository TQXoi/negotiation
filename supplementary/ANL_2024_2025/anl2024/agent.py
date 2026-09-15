from __future__ import annotations

from typing import Any

from anl.anl2024.negotiators.base import ANLNegotiator
from negmas import Outcome, ResponseType, SAOResponse, SAOState

from .belief import ReservationBelief
from .planner import BeliefUsableRVPlanner, CandidateScore
from .planner_v2 import BeliefUsableRVPlannerV2, CandidateScoreV2


class BeliefPlannerANL2024Negotiator(ANLNegotiator):
    """ANL 2024 adapter for continuous belief + belief-usable planning.

    The structured ANL protocol already executes the selected outcome, so a
    natural-language generator is intentionally absent. This isolates belief
    and planner effects from formatting/action-lock effects.
    """

    def __init__(
        self,
        *args: Any,
        belief_mode: str = "continuous",
        planner_version: str = "v1",
        oracle_rv: float | None = None,
        freeze_after: float | None = None,
        information_gain_weight: float | None = None,
        max_outcomes: int = 10_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if planner_version not in {"v1", "v2"}:
            raise ValueError(f"Unknown planner version: {planner_version}")
        self.belief_mode = belief_mode
        self.planner_version = planner_version
        self.oracle_rv = oracle_rv
        self.freeze_after = freeze_after
        self.max_outcomes = int(max_outcomes)
        self.belief = ReservationBelief(
            mode=belief_mode,
            oracle_value=oracle_rv,
        )
        if planner_version == "v2":
            self.planner = BeliefUsableRVPlannerV2(
                information_gain_weight=(
                    0.025
                    if information_gain_weight is None
                    else information_gain_weight
                ),
            )
        else:
            self.planner = BeliefUsableRVPlanner(
                information_gain_weight=(
                    0.08 if information_gain_weight is None else information_gain_weight
                ),
            )
        self.framework_trace: list[dict[str, Any]] = []
        self._outcomes: tuple[Any, ...] = tuple()
        self._last_proposal: Outcome | None = None
        self._last_processed: tuple[int, Any] | None = None

    def on_preferences_changed(self, changes: list[Any]) -> None:
        if self.ufun is None or self.nmi is None:
            return
        self._outcomes = tuple(
            self.nmi.outcome_space.enumerate_or_sample(
                levels=10,
                max_cardinality=self.max_outcomes,
            )
        )

    def _updates_enabled(self, relative_time: float) -> bool:
        # Static/oracle beliefs still consume and record public observations;
        # their ReservationBelief simply declines to change posterior weights.
        if self.belief_mode != "continuous":
            return True
        return self.freeze_after is None or relative_time <= self.freeze_after

    def _observe(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None or self.opponent_ufun is None:
            return
        signature = (int(state.step), tuple(offer))
        if signature == self._last_processed:
            return
        self._last_processed = signature
        relative_time = float(state.relative_time)
        belief_updates_enabled = self._updates_enabled(relative_time)
        if self._last_proposal is not None:
            rejected_utility = float(self.opponent_ufun(self._last_proposal))
            if belief_updates_enabled:
                self.belief.observe_rejection(rejected_utility, float(state.relative_time))
            if isinstance(self.planner, BeliefUsableRVPlannerV2):
                self.planner.observe_rejection(
                    opponent_utility=rejected_utility,
                    relative_time=relative_time,
                    belief=self.belief,
                    opponent_max=float(self.opponent_ufun.max()),
                )
        offered_utility = float(self.opponent_ufun(offer))
        if belief_updates_enabled:
            self.belief.observe_opponent_offer(offered_utility, float(state.relative_time))
        if isinstance(self.planner, BeliefUsableRVPlannerV2):
            if self.ufun is None:
                raise RuntimeError("Own utility function is unavailable")
            self.planner.observe_opponent_offer(
                opponent_utility=offered_utility,
                own_utility=float(self.ufun(offer)),
                relative_time=relative_time,
                belief=self.belief,
                opponent_max=float(self.opponent_ufun.max()),
            )

    def _plan(
        self, state: SAOState
    ) -> tuple[
        CandidateScore | CandidateScoreV2,
        list[CandidateScore | CandidateScoreV2],
    ]:
        if self.ufun is None or self.opponent_ufun is None:
            raise RuntimeError("ANL 2024 requires own and known-shape opponent ufuns")
        if not self._outcomes:
            self.on_preferences_changed([])
        return self.planner.choose(
            outcomes=self._outcomes,
            own_ufun=self.ufun,
            opponent_ufun=self.opponent_ufun,
            belief=self.belief,
            relative_time=float(state.relative_time),
        )

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        del dest
        self._observe(state)
        chosen, candidates = self._plan(state)
        accepted = False
        acceptance: dict[str, float] | None = None
        if state.current_offer is not None and self.ufun is not None:
            accepted, acceptance = self.planner.should_accept(
                received_offer=state.current_offer,
                planned=chosen,
                own_ufun=self.ufun,
                relative_time=float(state.relative_time),
            )
        trace = {
            "event": "accept" if accepted else "offer",
            "step": int(state.step),
            "relative_time": round(float(state.relative_time), 6),
            "belief": self.belief.to_json(),
            "planner_version": self.planner_version,
            "planner_state": (
                self.planner.to_json()
                if isinstance(self.planner, BeliefUsableRVPlannerV2)
                else {"planner_version": "1.0"}
            ),
            "received_offer": list(state.current_offer) if state.current_offer is not None else None,
            "acceptance": acceptance,
            "chosen": chosen.to_json(),
            "candidates": [candidate.to_json() for candidate in candidates],
        }
        self.framework_trace.append(trace)
        if accepted:
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        self._last_proposal = chosen.outcome
        return SAOResponse(ResponseType.REJECT_OFFER, chosen.outcome)

    def on_negotiation_end(self, state: SAOState) -> None:
        self.framework_trace.append(
            {
                "event": "end",
                "step": int(state.step),
                "agreement": list(state.agreement) if state.agreement is not None else None,
                "final_belief": self.belief.to_json(),
                "planner_version": self.planner_version,
                "planner_state": (
                    self.planner.to_json()
                    if isinstance(self.planner, BeliefUsableRVPlannerV2)
                    else {"planner_version": "1.0"}
                ),
            }
        )
        super().on_negotiation_end(state)
