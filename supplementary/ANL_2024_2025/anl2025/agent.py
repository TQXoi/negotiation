from __future__ import annotations

from typing import Any

from anl2025.negotiator import ANL2025Negotiator
from negmas import ResponseType
from negmas.outcomes import Outcome
from negmas.sao import SAOState

from .belief import EdgePreferenceBelief
from .planner import GlobalCandidateScore, GlobalContinuationPlanner


def _outcome_json(outcome: Any | None) -> list[Any] | None:
    return list(outcome) if outcome is not None else None


class BeliefGlobalPlannerANL2025Center(ANL2025Negotiator):
    """ANL 2025 center adapter for per-edge belief and global planning.

    ``continuous`` and ``static`` receive only protocol-visible offers and
    responses. ``oracle`` and ``fixed`` can receive edge utility functions,
    but are reserved for evaluator-side causal diagnostics and are never used
    by the standard framework variants.
    """

    framework_version = "anl2025-v1.0"

    def __init__(
        self,
        *args: Any,
        belief_mode: str = "continuous",
        use_future_value: bool = True,
        information_gain_weight: float = 0.02,
        future_acceptance_time: float = 0.8,
        future_state_beam: int = 16,
        max_outcomes_per_thread: int = 200,
        oracle_edge_ufuns: tuple[Any, ...] | list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.belief_mode = belief_mode
        self.use_future_value = bool(use_future_value)
        self.information_gain_weight = float(information_gain_weight)
        self.future_acceptance_time = float(future_acceptance_time)
        self.future_state_beam = int(future_state_beam)
        self.max_outcomes_per_thread = int(max_outcomes_per_thread)
        self._oracle_edge_ufuns = (
            tuple(oracle_edge_ufuns) if oracle_edge_ufuns is not None else None
        )

        self.beliefs: list[EdgePreferenceBelief] = []
        self.outcomes_by_thread: list[tuple[Any, ...]] = []
        self.agreements: list[Any | None] = []
        self.completed: list[bool] = []
        self.framework_trace: list[dict[str, Any]] = []
        self.planner: GlobalContinuationPlanner | None = None
        self._thread_index: dict[str, int] = {}
        self._last_seen_offer: dict[int, tuple[int, Any]] = {}
        self._pending_proposals: dict[int, dict[str, Any]] = {}

    def init(self) -> None:
        infos: list[tuple[int, str, Any]] = []
        for negotiator_id, info in self.negotiators.items():
            index = int(info.context["index"])
            infos.append((index, negotiator_id, info))
        infos.sort(key=lambda row: row[0])
        if not infos:
            raise RuntimeError("ANL 2025 center was initialized without negotiation threads")

        nthreads = max(index for index, _, _ in infos) + 1
        self._thread_index = {
            negotiator_id: index for index, negotiator_id, _ in infos
        }
        self.outcomes_by_thread = [tuple() for _ in range(nthreads)]
        for index, _, info in infos:
            outcome_space = info.negotiator.nmi.outcome_space
            outcomes = tuple(
                outcome_space.enumerate_or_sample(
                    levels=10,
                    max_cardinality=self.max_outcomes_per_thread,
                )
            )
            if not outcomes:
                raise RuntimeError(f"Thread {index} has no candidate outcomes")
            self.outcomes_by_thread[index] = outcomes

        if self.belief_mode in {"oracle", "fixed"}:
            if self._oracle_edge_ufuns is None:
                raise ValueError(
                    f"{self.belief_mode} is evaluator-only and requires oracle_edge_ufuns"
                )
            if len(self._oracle_edge_ufuns) != nthreads:
                raise ValueError(
                    "oracle_edge_ufuns must contain one utility function per thread"
                )
        elif self._oracle_edge_ufuns is not None:
            raise ValueError(
                "Standard ANL 2025 framework variants must not receive edge utility truth"
            )

        self.beliefs = [
            EdgePreferenceBelief(
                outcomes=self.outcomes_by_thread[index],
                mode=self.belief_mode,
                oracle_ufun=(
                    self._oracle_edge_ufuns[index]
                    if self._oracle_edge_ufuns is not None
                    else None
                ),
            )
            for index in range(nthreads)
        ]
        self.agreements = [None] * nthreads
        self.completed = [False] * nthreads
        self.planner = GlobalContinuationPlanner(
            center_ufun=self.ufun,
            use_future_value=self.use_future_value,
            information_gain_weight=self.information_gain_weight,
            future_acceptance_time=self.future_acceptance_time,
            future_state_beam=self.future_state_beam,
        )
        self.framework_trace.append(
            {
                "event": "framework_init",
                "framework_version": self.framework_version,
                "belief_mode": self.belief_mode,
                "n_threads": nthreads,
                "outcomes_per_thread": [
                    len(outcomes) for outcomes in self.outcomes_by_thread
                ],
                "planner": self.planner.to_json(),
                "truth_access": self._oracle_edge_ufuns is not None,
            }
        )

    def _relative_time(self, state: SAOState) -> float:
        value = getattr(state, "relative_time", 0.0)
        return max(0.0, min(1.0, float(value or 0.0)))

    def _record_pending_response(
        self, index: int, *, accepted: bool, relative_time: float
    ) -> None:
        pending = self._pending_proposals.get(index)
        if pending is None or pending["recorded"]:
            return
        self.beliefs[index].observe_response(
            pending["outcome"],
            relative_time,
            accepted=accepted,
            predicted_probability=float(pending["predicted_probability"]),
        )
        pending["recorded"] = True

    def _observe_received_offer(self, index: int, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None:
            return
        signature = (int(state.step), tuple(offer))
        if self._last_seen_offer.get(index) == signature:
            return
        # Receiving a counteroffer implies that our previous proposal was not
        # accepted. This is public protocol evidence for the response model.
        self._record_pending_response(
            index, accepted=False, relative_time=self._relative_time(state)
        )
        self.beliefs[index].observe_edge_offer(offer, self._relative_time(state))
        self._last_seen_offer[index] = signature

    def _plan(
        self, index: int, state: SAOState
    ) -> tuple[GlobalCandidateScore, list[GlobalCandidateScore]]:
        if self.planner is None or not self.beliefs:
            raise RuntimeError("ANL 2025 framework agent has not been initialized")
        return self.planner.choose(
            thread_index=index,
            relative_time=self._relative_time(state),
            agreements=self.agreements,
            completed=self.completed,
            outcomes_by_thread=self.outcomes_by_thread,
            beliefs=self.beliefs,
        )

    def _trace_action(
        self,
        *,
        event: str,
        index: int,
        state: SAOState,
        chosen: GlobalCandidateScore,
        candidates: list[GlobalCandidateScore],
        received_offer: Any | None = None,
        acceptance: dict[str, float] | None = None,
    ) -> None:
        if self.planner is None:
            raise RuntimeError("Planner is unavailable")
        self.framework_trace.append(
            {
                "event": event,
                "thread_index": index,
                "step": int(state.step),
                "relative_time": round(self._relative_time(state), 6),
                "agreements_so_far": [
                    _outcome_json(outcome) for outcome in self.agreements
                ],
                "belief": self.beliefs[index].to_json(),
                "planner": self.planner.to_json(),
                "received_offer": _outcome_json(received_offer),
                "acceptance": acceptance,
                "chosen": chosen.to_json(),
                "candidates": [candidate.to_json() for candidate in candidates],
            }
        )

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        del dest
        index = self._thread_index[negotiator_id]
        # Fallback for mechanisms that request another proposal without first
        # invoking respond() on the counteroffer.
        self._record_pending_response(
            index, accepted=False, relative_time=self._relative_time(state)
        )
        chosen, candidates = self._plan(index, state)
        self._pending_proposals[index] = {
            "outcome": chosen.outcome,
            "predicted_probability": chosen.edge_acceptance_probability,
            "recorded": False,
        }
        self._trace_action(
            event="propose",
            index=index,
            state=state,
            chosen=chosen,
            candidates=candidates,
        )
        return chosen.outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        del source
        index = self._thread_index[negotiator_id]
        self._observe_received_offer(index, state)
        chosen, candidates = self._plan(index, state)
        accepted = False
        acceptance: dict[str, float] | None = None
        if state.current_offer is not None:
            if self.planner is None:
                raise RuntimeError("Planner is unavailable")
            accepted, acceptance = self.planner.should_accept(
                received_offer=state.current_offer,
                planned=chosen,
                thread_index=index,
                relative_time=self._relative_time(state),
                agreements=self.agreements,
                completed=self.completed,
                outcomes_by_thread=self.outcomes_by_thread,
                beliefs=self.beliefs,
            )
        self._trace_action(
            event="accept" if accepted else "reject",
            index=index,
            state=state,
            chosen=chosen,
            candidates=candidates,
            received_offer=state.current_offer,
            acceptance=acceptance,
        )
        return (
            ResponseType.ACCEPT_OFFER if accepted else ResponseType.REJECT_OFFER
        )

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        index = self._thread_index[negotiator_id]
        pending = self._pending_proposals.get(index)
        accepted_our_proposal = bool(
            pending is not None
            and state.agreement is not None
            and tuple(pending["outcome"]) == tuple(state.agreement)
        )
        self._record_pending_response(
            index,
            accepted=accepted_our_proposal,
            relative_time=self._relative_time(state),
        )
        self.agreements[index] = state.agreement
        self.completed[index] = True
        self.framework_trace.append(
            {
                "event": "thread_end",
                "thread_index": index,
                "step": int(state.step),
                "agreement": _outcome_json(state.agreement),
                "agreements_so_far": [
                    _outcome_json(outcome) for outcome in self.agreements
                ],
                "belief": self.beliefs[index].to_json(),
            }
        )
