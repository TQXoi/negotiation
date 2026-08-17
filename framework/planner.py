"""Executable candidate planner whose score explicitly consumes belief."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .schemas import CandidateAction, CanonicalState, OpponentBelief, ScoredCandidate


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class PlannerConfig:
    """Environment-independent planner weights.

    Environment adapters define utilities and candidate feasibility.  These
    weights therefore operate on normalized utility rather than dollars or a
    benchmark-specific score.
    """

    agreement_bonus: float = 0.06
    information_weight: float = 0.18
    quit_risk_weight: float = 0.22
    uncertainty_risk_weight: float = 0.06
    continuation_weight: float = 0.10
    early_accept_regret_weight: float = 0.55
    minimum_probe_utility: float = 0.12
    max_information_share: float = 0.20


class BeliefUsablePlanner:
    """Rank feasible actions under a factorized opponent belief."""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        candidate_list = list(candidates)
        best_available_utility = max(
            (item.own_utility_normalized for item in candidate_list if item.action_type == "offer"),
            default=0.0,
        )
        scored = [
            self._score(state, belief, candidate, best_available_utility)
            for candidate in candidate_list
        ]
        if not scored:
            raise ValueError("Environment adapter produced no candidate actions.")
        return sorted(scored, key=lambda item: (item.score, item.candidate.own_utility), reverse=True)

    def _score(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidate: CandidateAction,
        best_available_utility: float,
    ) -> ScoredCandidate:
        progress = state.progress
        p_accept, acceptance_diagnostics = self._acceptance_probability(belief, candidate)
        p_quit = self._quit_probability(belief, candidate, progress)
        normalized_utility = candidate.own_utility_normalized

        if candidate.action_type == "accept":
            p_accept = 1.0
            p_quit = 0.0
        elif candidate.action_type == "quit":
            p_accept = 0.0
            p_quit = 0.0

        exploitation = p_accept * max(-1.0, normalized_utility)
        if candidate.action_type in {"offer", "accept"}:
            exploitation += p_accept * self.config.agreement_bonus
        if candidate.action_type == "accept":
            # Accept is certain agreement but gives up the option to make a
            # better feasible counteroffer.  Penalize that regret early and let
            # it vanish near the deadline.
            accept_regret = max(0.0, best_available_utility - normalized_utility)
            exploitation -= (
                self.config.early_accept_regret_weight
                * (1.0 - progress)
                * accept_regret
            )

        uncertainty = 1.0 - 0.5 * (
            belief.preference.reliability + belief.response_policy.reliability
        )
        probe_budget_ok = normalized_utility >= self.config.minimum_probe_utility
        information = 0.0
        if candidate.action_type in {"offer", "ask"} and probe_budget_ok:
            information = (
                self.config.information_weight
                * (1.0 - progress)
                * uncertainty
                * _clamp(candidate.information_gain)
            )
            information = min(information, self.config.max_information_share * max(0.0, normalized_utility))

        continuation = 0.0
        if candidate.action_type in {"offer", "ask", "reject"}:
            continuation = (
                self.config.continuation_weight
                * (1.0 - progress)
                * (1.0 - p_accept)
                * max(0.0, normalized_utility)
            )

        risk_penalty = self.config.quit_risk_weight * p_quit * (0.35 + 0.65 * progress)
        risk_penalty += self.config.uncertainty_risk_weight * uncertainty * max(0.0, -candidate.feasibility_margin)
        if normalized_utility < 0:
            risk_penalty += abs(normalized_utility) * 2.0
        if candidate.action_type == "quit":
            outside_normalized = state.own_outside_option / max(1e-9, state.own_value_scale)
            exploitation = outside_normalized
            risk_penalty = 0.03 if progress < 0.85 else 0.0

        score = exploitation + continuation + information - risk_penalty
        return ScoredCandidate(
            candidate=candidate,
            p_accept=_clamp(p_accept),
            p_quit=_clamp(p_quit),
            exploitation_value=exploitation + continuation,
            exploration_value=information,
            risk_penalty=risk_penalty,
            score=score,
            diagnostics={
                **acceptance_diagnostics,
                "belief_preference_reliability": belief.preference.reliability,
                "belief_policy_reliability": belief.response_policy.reliability,
                "regime_change_probability": belief.regime_change_probability,
                "progress": progress,
                "probe_budget_ok": probe_budget_ok,
            },
        )

    @staticmethod
    def _acceptance_probability(
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        compatibility = _clamp(candidate.opponent_value_proxy)
        preference = belief.preference
        policy = belief.response_policy
        structural = 1.0 / (
            1.0 + math.exp(-9.0 * (compatibility - preference.reservation_ratio_mean))
        )
        empirical = policy.acceptance_rate(compatibility)
        factorized = policy.reliability * empirical + (1.0 - policy.reliability) * structural

        # Reliability gating is the key safeguard inherited from V8: when the
        # learned belief is weak, fall back toward the adapter's protocol-local
        # prior rather than letting an uncertain posterior control the action.
        gate = _clamp(0.5 * (preference.reliability + policy.reliability))
        p_accept = gate * factorized + (1.0 - gate) * _clamp(candidate.base_acceptance)
        return p_accept, {
            "compatibility": compatibility,
            "structural_p_accept": structural,
            "empirical_p_accept": empirical,
            "adapter_base_p_accept": _clamp(candidate.base_acceptance),
            "belief_influence_gate": gate,
        }

    @staticmethod
    def _quit_probability(
        belief: OpponentBelief,
        candidate: CandidateAction,
        progress: float,
    ) -> float:
        policy = belief.response_policy
        incompatibility = 1.0 - _clamp(candidate.opponent_value_proxy)
        base = policy.quit_rate * (0.45 + 0.55 * progress)
        impatience = (1.0 - policy.patience) * progress * 0.35
        return _clamp(base + incompatibility * progress * 0.25 + impatience)


@dataclass(frozen=True)
class ProposalBackedTerminalPlannerConfig(PlannerConfig):
    """Conservative accept/continue boundary using only public proposals."""

    late_progress: float = 0.90
    stalled_progress: float = 0.65
    stalled_rejection_count: int = 3
    minimum_accept_surplus: float = 0.02
    late_counter_ev_slack: float = 0.04
    stalled_counter_ev_slack: float = 0.015


class ProposalBackedTerminalPlanner(BeliefUsablePlanner):
    """Prefer a guaranteed IR proposal when continued bargaining is low-value.

    An adapter creates an ``accept`` candidate only for the exact outstanding
    counterparty proposal.  This planner therefore does not infer acceptance
    from free text or inspect private opponent utility.  It compares that
    guaranteed focal utility with the predicted EV of another counteroffer,
    using a conservative acceptance cap after repeated formal failures.
    """

    config: ProposalBackedTerminalPlannerConfig

    def __init__(self, config: ProposalBackedTerminalPlannerConfig | None = None):
        super().__init__(config or ProposalBackedTerminalPlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        accepts = [row for row in ranked if row.candidate.action_type == "accept"]
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        if not accepts or not offers:
            return ranked

        progress = state.progress
        streak = max(0, int(belief.response_policy.repeated_rejection_streak))
        late = progress >= self.config.late_progress
        stalled = (
            progress >= self.config.stalled_progress
            and streak >= self.config.stalled_rejection_count
        )
        if not late and not stalled:
            return ranked

        outside = state.own_outside_option / max(1e-9, state.own_value_scale)
        accept = max(accepts, key=lambda row: row.candidate.own_utility_normalized)
        accept_value = accept.candidate.own_utility_normalized
        if accept_value < outside + self.config.minimum_accept_surplus:
            # The inherited agreement bonus can otherwise rank a zero-surplus
            # accept above a small positive counteroffer.  The guard is also a
            # safety boundary: never terminate at (approximately) outside
            # value while a buyer-IR offer remains available.
            best_offer_score = max(row.score for row in offers)
            accept.score = min(accept.score, best_offer_score - 1e-6)
            accept.diagnostics.update({
                "proposal_backed_guard_considered": True,
                "proposal_backed_guard_override": False,
                "proposal_backed_guard_reason": "insufficient_surplus",
            })
            return sorted(
                ranked,
                key=lambda item: (item.score, item.candidate.own_utility),
                reverse=True,
            )

        conservative_cap = 1.0 / (2.0 + streak) if stalled else 1.0
        counter_evs = []
        for offer in offers:
            p_accept = min(offer.p_accept, conservative_cap)
            utility = offer.candidate.own_utility_normalized
            counter_evs.append(p_accept * utility + (1.0 - p_accept) * outside)
        best_counter_ev = max(counter_evs, default=outside)
        slack = (
            self.config.late_counter_ev_slack
            if late else self.config.stalled_counter_ev_slack
        )
        should_accept = accept_value + slack >= best_counter_ev
        accept.diagnostics.update({
            "proposal_backed_guard_considered": True,
            "proposal_backed_guard_override": should_accept,
            "proposal_backed_guard_reason": "late" if late else "stalled",
            "proposal_backed_accept_value": accept_value,
            "proposal_backed_best_counter_ev": best_counter_ev,
            "proposal_backed_counter_acceptance_cap": conservative_cap,
            "proposal_backed_rejection_streak": streak,
            "proposal_backed_slack": slack,
        })
        if should_accept and ranked[0] is not accept:
            accept.score = ranked[0].score + 1e-6
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )


@dataclass(frozen=True)
class BeliefGroundedArbitratorConfig(ProposalBackedTerminalPlannerConfig):
    """Activation and safety policy for constrained candidate arbitration."""

    minimum_rejection_streak: int = 2
    minimum_progress: float = 0.55
    maximum_output_tokens: int = 160
    safe_improvement_enabled: bool = False
    score_gap_base: float = 0.01
    score_gap_progress: float = 0.01
    utility_loss_base: float = 0.025
    utility_loss_progress: float = 0.025
    minimum_acceptance_gain: float = 0.04
    maximum_acceptance_loss: float = 0.03


class BeliefGroundedCandidateArbitrator(ProposalBackedTerminalPlanner):
    """Let an LLM choose only among already-safe canonical candidates.

    This is deliberately an *arbitrator*, not a free-form planner.  The base
    proposal-backed score is computed first.  Only after repeated public
    rejection or meaningful deadline pressure can the model select an existing
    candidate id using the continuous opponent belief and normalized decision
    statistics.  Invalid output falls back exactly to the base top-1.
    """

    config: BeliefGroundedArbitratorConfig

    def __init__(self, client: Any, config: Optional[BeliefGroundedArbitratorConfig] = None):
        super().__init__(config or BeliefGroundedArbitratorConfig())
        self.client = client

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        base = ranked[0]
        streak = max(0, int(belief.response_policy.repeated_rejection_streak))
        activated = (
            streak >= self.config.minimum_rejection_streak
            or state.progress >= self.config.minimum_progress
        )
        diagnostics: Dict[str, Any] = {
            "arbitrator_enabled": True,
            "arbitrator_activated": activated,
            "arbitrator_valid": False,
            "arbitrator_override": False,
            "arbitrator_base_candidate_id": base.candidate.candidate_id,
            "arbitrator_selected_candidate_id": base.candidate.candidate_id,
            "arbitrator_reason": "activation_gate_closed",
            "arbitrator_rejection_streak": streak,
            "arbitrator_progress": state.progress,
        }
        if not activated or self.client is None:
            if self.client is None:
                diagnostics["arbitrator_reason"] = "no_model_client"
            base.diagnostics.update(diagnostics)
            return ranked

        # Every item was already made legal and buyer-IR-valid by the adapter.
        # Early quit is removed because a language model should not expand the
        # base planner's termination authority before the deadline.
        eligible = [
            row for row in ranked
            if row.candidate.action_type != "quit" or state.progress >= 0.90
        ]
        summaries = []
        for row in eligible:
            candidate = row.candidate
            summaries.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "action_type": candidate.action_type,
                    "offer": candidate.offer.to_dict() if candidate.offer else None,
                    "own_utility_normalized": round(candidate.own_utility_normalized, 6),
                    "opponent_value_proxy": round(candidate.opponent_value_proxy, 6),
                    "base_p_accept": round(row.p_accept, 6),
                    "base_p_quit": round(row.p_quit, 6),
                    "base_score": round(row.score, 6),
                    "information_gain": round(candidate.information_gain, 6),
                    "source": candidate.metadata.get("proposal_source")
                    or candidate.metadata.get("term_profile")
                    or candidate.rationale,
                }
            )
        preference = belief.preference
        policy = belief.response_policy
        belief_summary = {
            "reservation_ratio_interval": [
                preference.reservation_ratio_low,
                preference.reservation_ratio_high,
            ],
            "reservation_ratio_mean": preference.reservation_ratio_mean,
            "preference_reliability": preference.reliability,
            "response_policy_reliability": policy.reliability,
            "repeated_rejection_streak": streak,
            "rejected_compatibility_max": policy.rejected_compatibility_max,
            "quit_rate": policy.quit_rate,
            "patience": policy.patience,
            "regime_change_probability": belief.regime_change_probability,
        }
        prompt = f"""You are a constrained negotiation candidate arbitrator.
Choose exactly one candidate_id from the supplied safe candidate table. Do not invent an offer.
Maximize the buyer's expected final utility, balancing current surplus, evidence-grounded agreement
probability, information value, and deadline/no-deal risk. Treat opponent claims as uncertain; use the
continuous belief and observed rejection streak. Do not choose a higher-concession action merely to
get agreement when a lower candidate remains plausibly viable. Do not reveal private reasoning.

Turn: {state.turn}/{state.max_turns}
Belief summary: {json.dumps(belief_summary, ensure_ascii=False)}
Safe candidates: {json.dumps(summaries, ensure_ascii=False, default=str)}

Return only JSON: {{"candidate_id": "one exact id from the table"}}
"""
        diagnostics["arbitrator_attempted"] = True
        try:
            raw = self.client.generate(
                prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.config.maximum_output_tokens,
            ).strip()
        except Exception as exc:
            diagnostics.update(
                {
                    "arbitrator_reason": "model_error",
                    "arbitrator_error": f"{type(exc).__name__}: {exc}",
                }
            )
            base.diagnostics.update(diagnostics)
            return ranked
        diagnostics["arbitrator_raw"] = raw[:1000]
        chosen_id: Optional[str] = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                value, _ = decoder.raw_decode(raw[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("candidate_id"), str):
                chosen_id = value["candidate_id"].strip()
                break
        by_id = {row.candidate.candidate_id: row for row in eligible}
        chosen = by_id.get(chosen_id or "")
        if chosen is None:
            diagnostics.update(
                {
                    "arbitrator_reason": "invalid_candidate_id",
                    "arbitrator_requested_candidate_id": chosen_id,
                }
            )
            base.diagnostics.update(diagnostics)
            return ranked

        original_base_score = float(base.score)
        original_chosen_score = float(chosen.score)
        score_gap = max(0.0, original_base_score - original_chosen_score)
        utility_loss = (
            base.candidate.own_utility_normalized
            - chosen.candidate.own_utility_normalized
        )
        acceptance_gain = chosen.p_accept - base.p_accept
        allowed_score_gap = (
            self.config.score_gap_base
            + self.config.score_gap_progress * state.progress
        )
        allowed_utility_loss = (
            self.config.utility_loss_base
            + self.config.utility_loss_progress * state.progress
        )
        safe_checks = {
            "terminal_base_protected": base.candidate.action_type not in {"accept", "quit"},
            "offer_to_offer": (
                base.candidate.action_type == "offer"
                and chosen.candidate.action_type == "offer"
            ),
            "score_gap_ok": score_gap <= allowed_score_gap + 1e-12,
            "utility_loss_ok": utility_loss <= allowed_utility_loss + 1e-12,
            "concession_has_acceptance_gain": (
                utility_loss <= 0.0
                or acceptance_gain >= max(self.config.minimum_acceptance_gain, utility_loss)
            ),
            "utility_gain_preserves_acceptance": (
                utility_loss > 0.0
                or acceptance_gain >= -self.config.maximum_acceptance_loss
            ),
        }
        diagnostics.update(
            {
                "safe_improvement_enabled": self.config.safe_improvement_enabled,
                "safe_improvement_checks": safe_checks,
                "safe_improvement_score_gap": score_gap,
                "safe_improvement_allowed_score_gap": allowed_score_gap,
                "safe_improvement_utility_loss": utility_loss,
                "safe_improvement_allowed_utility_loss": allowed_utility_loss,
                "safe_improvement_acceptance_gain": acceptance_gain,
                "arbitrator_requested_candidate_id": chosen.candidate.candidate_id,
            }
        )
        if self.config.safe_improvement_enabled and not all(safe_checks.values()):
            diagnostics.update(
                {
                    "arbitrator_valid": True,
                    "arbitrator_override": False,
                    "arbitrator_selected_candidate_id": base.candidate.candidate_id,
                    "arbitrator_reason": "safe_improvement_rejected",
                }
            )
            base.diagnostics.update(diagnostics)
            return ranked

        diagnostics.update(
            {
                "arbitrator_valid": True,
                "arbitrator_override": chosen is not base,
                "arbitrator_selected_candidate_id": chosen.candidate.candidate_id,
                "arbitrator_reason": "valid_selection",
                "arbitrator_original_base_score": original_base_score,
                "arbitrator_original_chosen_score": original_chosen_score,
            }
        )
        chosen.diagnostics.update(diagnostics)
        if chosen is not base:
            chosen.score = base.score + 1e-6
            ranked = sorted(
                ranked,
                key=lambda item: (item.score, item.candidate.own_utility),
                reverse=True,
            )
        return ranked


@dataclass(frozen=True)
class VerifiedResponseFrontierPlannerConfig(ProposalBackedTerminalPlannerConfig):
    """Public-response frontier layered on the proposal-backed terminal guard.

    The frontier describes *observable response policy*, not private utility.
    In price bargaining an adapter may expose a normalized price coordinate in
    candidate metadata.  Other environments fall back to their monotone
    opponent-value proxy.
    """

    frontier_gate_rate: float = 0.90
    frontier_margin: float = 0.012
    minimum_jump_early: float = 0.025
    minimum_jump_late: float = 0.100
    additional_jump_per_rejection: float = 0.006
    maximum_required_jump: float = 0.18
    insufficient_separation_penalty: float = 0.34
    terminal_probe_progress: float = 0.90


class VerifiedResponseFrontierPlanner(ProposalBackedTerminalPlanner):
    """Stop repeating offers in a formally rejected response region.

    A counter/reject tells us that the tested action was not accepted *then*.
    It does not reveal the opponent's reservation utility.  This planner uses
    that distinction directly: it moves to an available, buyer-IR successor
    while keeping the latent preference posterior untouched.
    """

    config: VerifiedResponseFrontierPlannerConfig

    def __init__(self, config: VerifiedResponseFrontierPlannerConfig | None = None):
        super().__init__(config or VerifiedResponseFrontierPlannerConfig())

    @staticmethod
    def _response_coordinate(candidate: CandidateAction) -> float:
        explicit = candidate.metadata.get("response_coordinate")
        if isinstance(explicit, (int, float)):
            return _clamp(float(explicit))
        return _clamp(candidate.opponent_value_proxy)

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        policy = belief.response_policy
        frontier = _clamp(policy.rejected_compatibility_max)
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        if belief.direct_response_count <= 0 or frontier <= 0.0 or not offers:
            return ranked

        coordinates = {
            row.candidate.candidate_id: self._response_coordinate(row.candidate)
            for row in offers
        }
        maximum_coordinate = max(coordinates.values())
        successor_available = maximum_coordinate > frontier + self.config.frontier_margin
        if not successor_available:
            return ranked

        streak = max(1, int(policy.repeated_rejection_streak))
        required_jump = (
            self.config.minimum_jump_early
            + (self.config.minimum_jump_late - self.config.minimum_jump_early)
            * state.progress
            + self.config.additional_jump_per_rejection * max(0, streak - 1)
        )
        required_jump = min(self.config.maximum_required_jump, required_jump)
        # Candidate grids are intentionally small.  Target the best reachable
        # separation instead of disabling the frontier when the ideal jump is
        # not represented exactly.
        target = min(maximum_coordinate, frontier + required_jump)
        gate = 1.0 - math.exp(-self.config.frontier_gate_rate * belief.direct_response_count)
        outside = state.own_outside_option / max(1e-9, state.own_value_scale)

        for row in offers:
            coordinate = coordinates[row.candidate.candidate_id]
            insufficient = coordinate < target - 1e-9
            penalty = 0.0
            if insufficient:
                separation_fraction = _clamp(
                    (target - coordinate) / max(required_jump, 1e-9)
                )
                penalty = (
                    self.config.insufficient_separation_penalty
                    * gate
                    * (0.45 + 0.55 * state.progress)
                    * max(0.35, separation_fraction)
                )
                row.score -= penalty
                row.risk_penalty += penalty
            elif state.progress >= self.config.terminal_probe_progress:
                # Near the deadline a safe active offer has non-negative
                # response value relative to quitting.  This prevents the
                # response frontier from ending in a voluntary zero-value quit.
                response_floor = (
                    row.p_accept * row.candidate.own_utility_normalized
                    + (1.0 - row.p_accept) * outside
                )
                row.score = max(row.score, response_floor)
                row.exploitation_value = max(row.exploitation_value, response_floor)
                row.diagnostics["verified_frontier_terminal_response_floor"] = response_floor
            row.diagnostics.update(
                {
                    "verified_response_frontier": frontier,
                    "verified_response_coordinate": coordinate,
                    "verified_frontier_target": target,
                    "verified_frontier_required_jump": required_jump,
                    "verified_frontier_maximum_coordinate": maximum_coordinate,
                    "verified_frontier_successor_available": successor_available,
                    "verified_frontier_insufficient_separation": insufficient,
                    "verified_frontier_penalty": penalty,
                }
            )

        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )


@dataclass(frozen=True)
class PublicProposalFeasibilityPlannerConfig(VerifiedResponseFrontierPlannerConfig):
    """Early exact-proposal guard using public counterparty commitments only."""

    minimum_exact_repeat_count: int = 2
    minimum_price_plateau_count: int = 2
    minimum_direct_responses: int = 2
    minimum_rejection_streak: int = 2
    proposal_counter_ev_slack: float = 0.030


class PublicProposalFeasibilityPlanner(VerifiedResponseFrontierPlanner):
    """Balance the rejected-response frontier with a public proposal anchor.

    The exact outstanding counterparty proposal is the strongest public signal
    that a complete package is executable.  After repeated formal evidence,
    this guard compares its deterministic focal value with a conservatively
    capped counteroffer EV.  It never inspects counterparty private utility;
    private feasibility remains evaluator-only.
    """

    config: PublicProposalFeasibilityPlannerConfig

    def __init__(self, config: PublicProposalFeasibilityPlannerConfig | None = None):
        super().__init__(config or PublicProposalFeasibilityPlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        accepts = [row for row in ranked if row.candidate.action_type == "accept"]
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        if not accepts or not offers or ranked[0].candidate.action_type == "accept":
            return ranked

        frontier_active = any(
            "verified_response_frontier" in row.diagnostics for row in offers
        )
        if not frontier_active:
            return ranked

        accept = max(accepts, key=lambda row: row.candidate.own_utility_normalized)
        metadata = accept.candidate.metadata
        exact_repeats = max(0, int(metadata.get("public_proposal_exact_repeat_count") or 0))
        price_plateau = max(0, int(metadata.get("public_proposal_price_plateau_count") or 0))
        proposal_count = max(0, int(metadata.get("public_proposal_observation_count") or 0))
        credible = (
            exact_repeats >= self.config.minimum_exact_repeat_count
            or price_plateau >= self.config.minimum_price_plateau_count
        )
        streak = max(0, int(belief.response_policy.repeated_rejection_streak))
        enough_response_evidence = (
            belief.direct_response_count >= self.config.minimum_direct_responses
            and streak >= self.config.minimum_rejection_streak
        )
        if not credible or not enough_response_evidence:
            return ranked

        outside = state.own_outside_option / max(1e-9, state.own_value_scale)
        accept_value = accept.candidate.own_utility_normalized
        if accept_value < outside + self.config.minimum_accept_surplus:
            accept.diagnostics.update({
                "public_proposal_guard_considered": True,
                "public_proposal_guard_override": False,
                "public_proposal_guard_reason": "insufficient_buyer_surplus",
                "public_proposal_exact_repeat_count": exact_repeats,
                "public_proposal_price_plateau_count": price_plateau,
                "public_proposal_observation_count": proposal_count,
            })
            return ranked

        conservative_cap = 1.0 / (2.0 + streak)
        counter_evs = []
        for offer in offers:
            p_accept = min(offer.p_accept, conservative_cap)
            utility = offer.candidate.own_utility_normalized
            counter_evs.append(p_accept * utility + (1.0 - p_accept) * outside)
        best_counter_ev = max(counter_evs, default=outside)
        should_accept = (
            accept_value + self.config.proposal_counter_ev_slack >= best_counter_ev
        )
        accept.diagnostics.update({
            "public_proposal_guard_considered": True,
            "public_proposal_guard_override": should_accept,
            "public_proposal_guard_reason": "credible_repeated_public_proposal",
            "public_proposal_accept_value": accept_value,
            "public_proposal_best_counter_ev": best_counter_ev,
            "public_proposal_counter_acceptance_cap": conservative_cap,
            "public_proposal_rejection_streak": streak,
            "public_proposal_exact_repeat_count": exact_repeats,
            "public_proposal_price_plateau_count": price_plateau,
            "public_proposal_observation_count": proposal_count,
            "public_proposal_slack": self.config.proposal_counter_ev_slack,
        })
        if should_accept:
            accept.score = ranked[0].score + 1e-6
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )


@dataclass(frozen=True)
class DeadlineAwarePlannerConfig(PlannerConfig):
    """V2 additions expressed in normalized, environment-independent units."""

    rejected_region_weight: float = 0.24
    repeated_action_weight: float = 0.12
    direct_evidence_gate_rate: float = 0.55
    late_information_decay_start: float = 0.55


class DeadlineAwareBeliefUsablePlanner(BeliefUsablePlanner):
    """Planner V2: respect direct failures and stop probing near a deadline."""

    config: DeadlineAwarePlannerConfig

    def __init__(self, config: DeadlineAwarePlannerConfig | None = None):
        super().__init__(config or DeadlineAwarePlannerConfig())

    def _score(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidate: CandidateAction,
        best_available_utility: float,
    ) -> ScoredCandidate:
        scored = super()._score(state, belief, candidate, best_available_utility)
        if candidate.action_type not in {"offer", "ask"}:
            return scored

        policy = belief.response_policy
        progress = state.progress
        compatibility = _clamp(candidate.opponent_value_proxy)
        direct_confidence = 1.0 - math.exp(
            -self.config.direct_evidence_gate_rate * belief.direct_response_count
        )
        rejected_penalty = 0.0
        if (
            policy.rejected_compatibility_max > 0.0
            and compatibility <= policy.rejected_compatibility_max + 1e-9
        ):
            # A formal reject/counter is direct evidence about the tested
            # region. The penalty grows near the deadline but never forbids an
            # informative low-cost probe early in the episode.
            rejected_penalty = (
                self.config.rejected_region_weight
                * direct_confidence
                * (0.25 + 0.75 * progress)
            )

        novelty = _clamp(candidate.metadata.get("action_novelty", 1.0))
        repeated_penalty = (
            self.config.repeated_action_weight
            * min(1.0, policy.repeated_rejection_streak / 3.0)
            * (1.0 - novelty)
            * progress
        )

        information_decay = 1.0
        if progress > self.config.late_information_decay_start:
            span = max(1e-9, 1.0 - self.config.late_information_decay_start)
            information_decay = max(
                0.0, 1.0 - (progress - self.config.late_information_decay_start) / span
            )
        removed_information = scored.exploration_value * (1.0 - information_decay)
        added_risk = rejected_penalty + repeated_penalty
        scored.exploration_value *= information_decay
        scored.risk_penalty += added_risk
        scored.score -= removed_information + added_risk
        scored.diagnostics.update(
            {
                "direct_evidence_confidence": direct_confidence,
                "rejected_region_penalty": rejected_penalty,
                "repeated_action_penalty": repeated_penalty,
                "action_novelty": novelty,
                "information_decay": information_decay,
            }
        )
        return scored

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        p_accept, diagnostics = super()._acceptance_probability(belief, candidate)
        policy = belief.response_policy
        compatibility = _clamp(candidate.opponent_value_proxy)
        direct_gate = 1.0 - math.exp(
            -self.config.direct_evidence_gate_rate * belief.direct_response_count
        )
        if (
            policy.rejected_compatibility_max > 0.0
            and compatibility <= policy.rejected_compatibility_max + 1e-9
        ):
            empirical_cap = 1.0 / (2.0 + max(1, policy.repeated_rejection_streak))
            p_accept = (1.0 - direct_gate) * p_accept + direct_gate * min(
                diagnostics["empirical_p_accept"], empirical_cap
            )
            diagnostics["direct_rejection_cap"] = empirical_cap
        diagnostics["direct_evidence_gate"] = direct_gate
        return _clamp(p_accept), diagnostics


@dataclass(frozen=True)
class BehavioralPlannerConfig(DeadlineAwarePlannerConfig):
    """V4: utility belief is a feasibility model, not an acceptance shortcut."""

    rejected_region_weight: float = 0.0
    repeated_action_weight: float = 0.20
    direct_evidence_gate_rate: float = 0.42


class BehavioralBeliefUsablePlanner(DeadlineAwareBeliefUsablePlanner):
    """Use a monotone behavioral posterior for accept/reject prediction.

    An acceptance at a low compatibility supports all higher-compatibility
    actions; a rejection at a high compatibility supports all lower actions.
    This partial-order pooling is usable for any adapter that maps candidates
    to a normalized opponent-value proxy.
    """

    config: BehavioralPlannerConfig

    def __init__(self, config: BehavioralPlannerConfig | None = None):
        super().__init__(config or BehavioralPlannerConfig())

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        compatibility = _clamp(candidate.opponent_value_proxy)
        policy = belief.response_policy
        index = min(4, max(0, int(compatibility * 5)))
        accept_evidence = sum(
            max(0.0, policy.accept_alpha[bin_id] - 1.0)
            for bin_id in range(index + 1)
        )
        reject_evidence = sum(
            max(0.0, policy.accept_beta[bin_id] - 1.0)
            for bin_id in range(index, 5)
        )
        monotone_empirical = (1.0 + accept_evidence) / (
            2.0 + accept_evidence + reject_evidence
        )
        direct_gate = 1.0 - math.exp(
            -self.config.direct_evidence_gate_rate * belief.direct_response_count
        )
        base = _clamp(candidate.base_acceptance)
        p_accept = (1.0 - direct_gate) * base + direct_gate * monotone_empirical
        return _clamp(p_accept), {
            "compatibility": compatibility,
            "adapter_base_p_accept": base,
            "monotone_behavioral_p_accept": monotone_empirical,
            "behavioral_accept_evidence": accept_evidence,
            "behavioral_reject_evidence": reject_evidence,
            "direct_evidence_gate": direct_gate,
            "belief_influence_gate": direct_gate,
            "structural_p_accept": 0.0,
            "empirical_p_accept": monotone_empirical,
        }


@dataclass(frozen=True)
class LookaheadPlannerConfig(BehavioralPlannerConfig):
    """V5 weights for utility feasibility and a two-step fallback."""

    information_weight: float = 0.06
    max_information_share: float = 0.08
    continuation_weight: float = 0.0
    quit_risk_weight: float = 0.15
    repeated_action_weight: float = 0.14
    lookahead_discount: float = 0.72
    maximum_lookahead_bonus: float = 0.24
    feasibility_floor: float = 0.20


class LookaheadBeliefUsablePlanner(BehavioralBeliefUsablePlanner):
    """Robust short-horizon planner for learned factorized beliefs.

    Immediate acceptance is capped by utility feasibility. For each offer the
    planner also values the best more-compatible next action after rejection.
    This avoids early over-concession while retaining deadline recovery.
    """

    config: LookaheadPlannerConfig

    def __init__(self, config: LookaheadPlannerConfig | None = None):
        super().__init__(config or LookaheadPlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        candidates = list(candidates)
        base_rank = super().rank(state, belief, candidates)
        offer_rows = [row for row in base_rank if row.candidate.action_type == "offer"]
        for row in offer_rows:
            current = row.candidate
            successors = [
                other for other in offer_rows
                if other.candidate.opponent_value_proxy > current.opponent_value_proxy + 1e-6
            ]
            future_best = max(
                (
                    other.p_accept
                    * max(0.0, other.candidate.own_utility_normalized + self.config.agreement_bonus)
                    for other in successors
                ),
                default=0.0,
            )
            continuation = (
                (1.0 - row.p_accept)
                * (1.0 - row.p_quit)
                * self.config.lookahead_discount
                * max(0.0, 1.0 - state.progress)
                * future_best
            )
            continuation = min(self.config.maximum_lookahead_bonus, continuation)
            row.exploitation_value += continuation
            row.score += continuation
            row.diagnostics.update({
                "lookahead_future_best": future_best,
                "lookahead_continuation": continuation,
                "lookahead_successor_count": len(successors),
            })
        return sorted(
            base_rank,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        behavioral, diagnostics = super()._acceptance_probability(belief, candidate)
        compatibility = _clamp(candidate.opponent_value_proxy)
        pref = belief.preference
        low = _clamp(pref.reservation_ratio_low, 0.0, 1.5)
        high = max(low + 1e-6, _clamp(pref.reservation_ratio_high, 0.0, 1.5))
        utility_feasible = _clamp((compatibility - low) / (high - low))
        feasibility_gate = _clamp(pref.reliability)
        feasibility_multiplier = (
            1.0 - feasibility_gate
            + feasibility_gate
            * (self.config.feasibility_floor + (1.0 - self.config.feasibility_floor) * utility_feasible)
        )
        p_accept = behavioral * feasibility_multiplier
        diagnostics.update({
            "utility_feasibility_probability": utility_feasible,
            "utility_feasibility_gate": feasibility_gate,
            "utility_feasibility_multiplier": feasibility_multiplier,
            "behavioral_p_accept_before_feasibility": behavioral,
        })
        return _clamp(p_accept), diagnostics


class TerminalSafeLookaheadPlanner(LookaheadBeliefUsablePlanner):
    """V5.2 fixes terminal dominance and point-mass feasibility beliefs."""

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        if state.progress < 1.0:
            return ranked
        outside = state.own_outside_option / max(1e-9, state.own_value_scale)
        for row in ranked:
            if row.candidate.action_type != "offer":
                continue
            utility = row.candidate.own_utility_normalized
            terminal_value = row.p_accept * utility + (1.0 - row.p_accept) * outside
            # No future quit-risk penalty exists after the final action. A
            # non-negative offer with p(accept)>0 weakly dominates quitting.
            row.score = max(row.score, terminal_value)
            row.exploitation_value = max(row.exploitation_value, terminal_value)
            row.diagnostics.update({
                "terminal_dominance_applied": True,
                "terminal_expected_value": terminal_value,
            })
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        p_accept, diagnostics = super()._acceptance_probability(belief, candidate)
        pref = belief.preference
        if abs(pref.reservation_ratio_high - pref.reservation_ratio_low) > 1e-6:
            return p_accept, diagnostics
        compatibility = _clamp(candidate.opponent_value_proxy)
        feasible = 1.0 if compatibility + 1e-9 >= pref.reservation_ratio_mean else 0.0
        behavioral = diagnostics["behavioral_p_accept_before_feasibility"]
        gate = _clamp(pref.reliability)
        multiplier = (
            1.0 - gate
            + gate
            * (self.config.feasibility_floor + (1.0 - self.config.feasibility_floor) * feasible)
        )
        diagnostics.update({
            "utility_feasibility_probability": feasible,
            "utility_feasibility_multiplier": multiplier,
            "point_mass_feasibility": True,
        })
        return _clamp(behavioral * multiplier), diagnostics


@dataclass(frozen=True)
class OptionAwarePlannerConfig(LookaheadPlannerConfig):
    """V5.3 adds epistemic option value before accepting an ask."""

    uncertain_accept_option_weight: float = 0.08


class DominanceAndOptionPlanner(TerminalSafeLookaheadPlanner):
    """V5.3 removes voluntary dominated quits and protects counter options."""

    config: OptionAwarePlannerConfig

    def __init__(self, config: OptionAwarePlannerConfig | None = None):
        super().__init__(config or OptionAwarePlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        outside = state.own_outside_option / max(1e-9, state.own_value_scale)
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        for row in offers:
            utility = row.candidate.own_utility_normalized
            offer_floor = row.p_accept * utility + (1.0 - row.p_accept) * outside
            # In a no-cost offer protocol, rejection/quit yields the same
            # outside option as voluntary quit. A feasible offer therefore
            # weakly dominates quitting at every round, not only the last.
            row.score = max(row.score, offer_floor)
            row.exploitation_value = max(row.exploitation_value, offer_floor)
            row.diagnostics.update({
                "offer_dominance_floor_applied": True,
                "offer_dominance_floor": offer_floor,
            })

        if offers and state.progress < 1.0:
            uncertainty = 1.0 - _clamp(belief.response_policy.reliability)
            option_penalty = (
                self.config.uncertain_accept_option_weight
                * (1.0 - state.progress)
                * uncertainty
            )
            for row in ranked:
                if row.candidate.action_type != "accept":
                    continue
                row.score -= option_penalty
                row.risk_penalty += option_penalty
                row.diagnostics.update({
                    "uncertain_accept_option_penalty": option_penalty,
                    "accept_policy_uncertainty": uncertainty,
                })
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )


@dataclass(frozen=True)
class BehavioralFrontierPlannerConfig(OptionAwarePlannerConfig):
    """V5.5 uses formal rejection as policy evidence, never utility truth.

    ``opponent_value_proxy`` is an adapter-provided monotone scalar.  If an
    action with proxy ``x`` was formally rejected, repeating an otherwise
    equivalent action at or below ``x`` has low immediate response value.  The
    rejected frontier must not be copied into the latent reservation belief:
    strategic opponents routinely reject offers they could profitably accept.
    """

    frontier_gate_rate: float = 0.90
    frontier_acceptance_discount: float = 0.82
    frontier_score_penalty: float = 0.36
    frontier_margin: float = 0.012


class BehavioralFrontierPlanner(DominanceAndOptionPlanner):
    """Consume the response-policy rejection frontier during planning.

    This planner is environment independent.  It only assumes that higher
    ``opponent_value_proxy`` means weakly better for the counterparty.  An
    adapter that cannot provide that ordering should leave direct response
    evidence at zero or use another planner.
    """

    config: BehavioralFrontierPlannerConfig

    def __init__(self, config: BehavioralFrontierPlannerConfig | None = None):
        super().__init__(config or BehavioralFrontierPlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        frontier = _clamp(belief.response_policy.rejected_compatibility_max)
        if belief.direct_response_count <= 0 or frontier <= 0.0:
            return ranked

        gate = 1.0 - math.exp(
            -self.config.frontier_gate_rate * belief.direct_response_count
        )
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        has_successor = any(
            row.candidate.opponent_value_proxy
            > frontier + self.config.frontier_margin
            for row in offers
        )
        for row in offers:
            compatibility = _clamp(row.candidate.opponent_value_proxy)
            behind_frontier = compatibility <= frontier + 1e-9
            penalty = 0.0
            if behind_frontier and has_successor:
                # Repeating a formally rejected region is a response-policy
                # error, not evidence that the opponent's true reservation is
                # at the frontier.  Penalize it increasingly near the deadline
                # while preserving a non-zero chance of strategic acceptance.
                penalty = (
                    self.config.frontier_score_penalty
                    * gate
                    * (0.40 + 0.60 * state.progress)
                )
                row.score -= penalty
                row.risk_penalty += penalty
            row.diagnostics.update(
                {
                    "behavioral_frontier": frontier,
                    "behavioral_frontier_gate": gate,
                    "behind_rejected_frontier": behind_frontier,
                    "frontier_successor_available": has_successor,
                    "behavioral_frontier_penalty": penalty,
                }
            )
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        p_accept, diagnostics = super()._acceptance_probability(belief, candidate)
        frontier = _clamp(belief.response_policy.rejected_compatibility_max)
        behind_frontier = (
            belief.direct_response_count > 0
            and frontier > 0.0
            and _clamp(candidate.opponent_value_proxy) <= frontier + 1e-9
        )
        gate = 1.0 - math.exp(
            -self.config.frontier_gate_rate * belief.direct_response_count
        )
        multiplier = 1.0
        if behind_frontier:
            multiplier = 1.0 - gate * self.config.frontier_acceptance_discount
            p_accept *= multiplier
        diagnostics.update(
            {
                "behavioral_frontier": frontier,
                "behavioral_frontier_gate": gate,
                "behind_rejected_frontier": behind_frontier,
                "frontier_acceptance_multiplier": multiplier,
            }
        )
        return _clamp(p_accept), diagnostics


class AspirationAwareFrontierPlanner(BehavioralFrontierPlanner):
    """V6 planner: use utility for feasibility and aspiration for response.

    The inherited feasibility term still consumes ``PreferenceBelief``.  This
    additional response-policy gate predicts whether the same feasible offer
    is likely to be accepted *now*.  It is deliberately a soft multiplier so
    an uncertain aspiration posterior cannot make an action impossible.
    """

    aspiration_floor: float = 0.18

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        p_accept, diagnostics = super()._acceptance_probability(belief, candidate)
        policy = belief.response_policy
        compatibility = _clamp(candidate.opponent_value_proxy)
        target = _clamp(policy.aspiration_ratio_mean, 0.0, 1.25)
        aspiration_accept = 1.0 / (1.0 + math.exp(-10.0 * (compatibility - target)))
        gate = _clamp(policy.aspiration_reliability)
        multiplier = 1.0 - gate + gate * (
            self.aspiration_floor + (1.0 - self.aspiration_floor) * aspiration_accept
        )
        diagnostics.update(
            {
                "aspiration_ratio_mean": target,
                "aspiration_acceptance_probability": aspiration_accept,
                "aspiration_gate": gate,
                "aspiration_multiplier": multiplier,
                "p_accept_before_aspiration": p_accept,
            }
        )
        return _clamp(p_accept * multiplier), diagnostics


class AspirationProbeFrontierPlanner(BehavioralFrontierPlanner):
    """V6.1 uses aspiration to probe, not to redefine utility feasibility.

    Offers near an uncertain current aspiration boundary have high response
    information.  A small, capped early bonus selects such probes while the
    inherited frontier and normalized own utility retain control of safety and
    exploitation.  This avoids V6.0's failure mode where pessimistic response
    prediction directly caused over-concession.
    """

    probe_weight: float = 0.045
    probe_bandwidth: float = 0.14
    maximum_probe_bonus: float = 0.025

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        policy = belief.response_policy
        if belief.direct_response_count <= 0:
            return ranked
        target = _clamp(policy.aspiration_ratio_mean, 0.0, 1.25)
        interval_width = _clamp(
            policy.aspiration_ratio_high - policy.aspiration_ratio_low,
            0.0,
            1.0,
        )
        uncertainty = max(interval_width, 1.0 - _clamp(policy.aspiration_reliability))
        for row in ranked:
            if row.candidate.action_type != "offer":
                continue
            compatibility = _clamp(row.candidate.opponent_value_proxy)
            distance = (compatibility - target) / max(self.probe_bandwidth, 1e-9)
            boundary_proximity = math.exp(-0.5 * distance * distance)
            bonus = (
                self.probe_weight
                * (1.0 - state.progress)
                * uncertainty
                * boundary_proximity
            )
            bonus = min(self.maximum_probe_bonus, bonus)
            row.exploration_value += bonus
            row.score += bonus
            row.diagnostics.update(
                {
                    "aspiration_probe_target": target,
                    "aspiration_interval_width": interval_width,
                    "aspiration_probe_proximity": boundary_proximity,
                    "aspiration_probe_bonus": bonus,
                }
            )
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )


class PosteriorIntegratedFrontierPlanner(BehavioralFrontierPlanner):
    """V5.8 integrates utility feasibility over the full latent posterior.

    The compressed posterior mean and 10/90 interval remain useful diagnostics,
    but they can erase multimodality after one censored response.  Candidate
    feasibility therefore uses posterior mass directly whenever available.
    """

    def _acceptance_probability(
        self,
        belief: OpponentBelief,
        candidate: CandidateAction,
    ) -> tuple[float, Dict[str, float]]:
        weights = belief.latent_profile_weights
        ratios = belief.latent_profile_reservation_ratios
        pref = belief.preference
        point_mass = abs(pref.reservation_ratio_high - pref.reservation_ratio_low) <= 1e-6
        if not point_mass and (not weights or set(weights) != set(ratios)):
            return super()._acceptance_probability(belief, candidate)

        behavioral, diagnostics = BehavioralBeliefUsablePlanner._acceptance_probability(
            self, belief, candidate
        )
        compatibility = _clamp(candidate.opponent_value_proxy)
        if point_mass:
            utility_feasible = 1.0 if compatibility + 1e-9 >= pref.reservation_ratio_mean else 0.0
            feasibility_source = "point_mass_intervention"
        else:
            total = sum(max(0.0, value) for value in weights.values())
            utility_feasible = sum(
                max(0.0, weight)
                for profile_id, weight in weights.items()
                if ratios[profile_id] <= compatibility + 1e-9
            ) / max(1e-12, total)
            feasibility_source = "latent_profile_posterior_cdf"
        feasibility_gate = _clamp(pref.reliability)
        feasibility_multiplier = (
            1.0
            - feasibility_gate
            + feasibility_gate
            * (
                self.config.feasibility_floor
                + (1.0 - self.config.feasibility_floor) * utility_feasible
            )
        )
        p_accept = behavioral * feasibility_multiplier

        frontier = _clamp(belief.response_policy.rejected_compatibility_max)
        behind_frontier = (
            belief.direct_response_count > 0
            and frontier > 0.0
            and compatibility <= frontier + 1e-9
        )
        frontier_gate = 1.0 - math.exp(
            -self.config.frontier_gate_rate * belief.direct_response_count
        )
        frontier_multiplier = 1.0
        if behind_frontier:
            frontier_multiplier = (
                1.0 - frontier_gate * self.config.frontier_acceptance_discount
            )
            p_accept *= frontier_multiplier
        diagnostics.update(
            {
                "utility_feasibility_probability": utility_feasible,
                "utility_feasibility_source": feasibility_source,
                "utility_feasibility_gate": feasibility_gate,
                "utility_feasibility_multiplier": feasibility_multiplier,
                "behavioral_p_accept_before_feasibility": behavioral,
                "behavioral_frontier": frontier,
                "behavioral_frontier_gate": frontier_gate,
                "behind_rejected_frontier": behind_frontier,
                "frontier_acceptance_multiplier": frontier_multiplier,
            }
        )
        return _clamp(p_accept), diagnostics


@dataclass(frozen=True)
class ActiveFrontierPlannerConfig(BehavioralFrontierPlannerConfig):
    """V5.6 makes a rejected-frontier probe informative before time expires."""

    minimum_probe_jump_early: float = 0.025
    minimum_probe_jump_late: float = 0.100
    additional_jump_per_rejection: float = 0.010
    insufficient_probe_penalty: float = 0.28


class ActiveFrontierPlanner(BehavioralFrontierPlanner):
    """Choose a deadline-adaptive probe beyond the rejected policy region.

    Small consecutive price changes often query effectively the same response
    region while consuming a whole interaction turn.  The required separation
    grows with elapsed time and the number of formal responses.  This is an
    information/survival rule over public behavior; it never reads or changes
    the opponent's private utility posterior.
    """

    config: ActiveFrontierPlannerConfig

    def __init__(self, config: ActiveFrontierPlannerConfig | None = None):
        super().__init__(config or ActiveFrontierPlannerConfig())

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> List[ScoredCandidate]:
        ranked = super().rank(state, belief, candidates)
        frontier = _clamp(belief.response_policy.rejected_compatibility_max)
        if belief.direct_response_count <= 0 or frontier <= 0.0:
            return ranked

        required_jump = (
            self.config.minimum_probe_jump_early
            + (
                self.config.minimum_probe_jump_late
                - self.config.minimum_probe_jump_early
            )
            * state.progress
            + self.config.additional_jump_per_rejection
            * max(0, belief.direct_response_count - 1)
        )
        minimum_informative_proxy = min(1.0, frontier + required_jump)
        gate = 1.0 - math.exp(
            -self.config.frontier_gate_rate * belief.direct_response_count
        )
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        informative_exists = any(
            row.candidate.opponent_value_proxy >= minimum_informative_proxy - 1e-9
            for row in offers
        )
        for row in offers:
            proxy = _clamp(row.candidate.opponent_value_proxy)
            insufficient = proxy < minimum_informative_proxy - 1e-9
            penalty = 0.0
            if informative_exists and insufficient:
                penalty = (
                    self.config.insufficient_probe_penalty
                    * gate
                    * (0.50 + 0.50 * state.progress)
                )
                row.score -= penalty
                row.risk_penalty += penalty
            row.diagnostics.update(
                {
                    "active_probe_required_jump": required_jump,
                    "active_probe_minimum_proxy": minimum_informative_proxy,
                    "active_probe_informative_successor_exists": informative_exists,
                    "active_probe_insufficient_separation": insufficient,
                    "active_probe_penalty": penalty,
                }
            )
        return sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )
