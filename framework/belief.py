"""Persistent and reliability-gated opponent belief updates."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .schemas import CanonicalOffer, CanonicalState, NegotiationObservation, OpponentBelief


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


class TextGenerator(Protocol):
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str: ...


@dataclass
class BeliefStore:
    """Beliefs keyed by session and counterparty, never by call order."""

    beliefs: Dict[Tuple[str, str], OpponentBelief] = field(default_factory=dict)
    processed_observations: set[Tuple[str, str, str]] = field(default_factory=set)

    def get(self, session_id: str, counterparty_id: str) -> OpponentBelief:
        key = (session_id, counterparty_id)
        if key not in self.beliefs:
            self.beliefs[key] = OpponentBelief(counterparty_id=counterparty_id)
        return self.beliefs[key]

    def unseen(
        self,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> List[NegotiationObservation]:
        result: List[NegotiationObservation] = []
        for observation in observations:
            key = (state.session_id, observation.counterparty_id, observation.observation_id)
            if key in self.processed_observations:
                continue
            self.processed_observations.add(key)
            if observation.counterparty_id == state.counterparty_id:
                result.append(observation)
        return result

    def reset_session(self, session_id: str) -> None:
        self.beliefs = {key: value for key, value in self.beliefs.items() if key[0] != session_id}
        self.processed_observations = {key for key in self.processed_observations if key[0] != session_id}


class BroadPriorBeliefStore(BeliefStore):
    """V5.4 prior with support for low-reservation and tight opponents."""

    def get(self, session_id: str, counterparty_id: str) -> OpponentBelief:
        key = (session_id, counterparty_id)
        if key not in self.beliefs:
            belief = OpponentBelief(counterparty_id=counterparty_id)
            belief.preference.reservation_ratio_low = 0.05
            belief.preference.reservation_ratio_mean = 0.55
            belief.preference.reservation_ratio_high = 0.98
            self.beliefs[key] = belief
        return self.beliefs[key]


class StructuredBeliefUpdater:
    """Update preference and response beliefs from protocol-level evidence.

    Formal accept/reject/counter actions receive more weight than natural
    language claims.  Semantic evidence can refine issue preferences but is
    capped by an evidence-reliability gate.
    """

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        for observation in observations:
            self._update_one(belief, state, observation)

    def _update_one(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observation: NegotiationObservation,
    ) -> None:
        preference = belief.preference
        policy = belief.response_policy
        evidence_strength = 0.0
        observed_ratio = self._price_ratio(observation.offer, state)
        response_ratio = self._price_ratio(observation.response_to_offer, state)

        if observation.response_type in {"accept", "reject", "counter"} and response_ratio is not None:
            predicted = self._prior_acceptance_probability(belief, response_ratio)
            outcome = 1.0 if observation.response_type == "accept" else 0.0
            surprise = abs(outcome - predicted)
            self._update_change_point(belief, surprise)
            index = min(4, max(0, int(response_ratio * 5)))
            if observation.response_type == "accept":
                policy.accept_alpha[index] += 1.0
                policy.accepted_compatibility_min = (
                    response_ratio
                    if policy.accepted_compatibility_min is None
                    else min(policy.accepted_compatibility_min, response_ratio)
                )
                policy.repeated_rejection_streak = 0
                policy.last_rejected_compatibility = None
                preference.reservation_ratio_high = min(preference.reservation_ratio_high, response_ratio)
                preference.reservation_ratio_mean = 0.75 * preference.reservation_ratio_mean + 0.25 * response_ratio
            else:
                policy.accept_beta[index] += 1.0
                policy.rejected_compatibility_max = max(
                    policy.rejected_compatibility_max, response_ratio
                )
                if (
                    policy.last_rejected_compatibility is not None
                    and abs(policy.last_rejected_compatibility - response_ratio) <= 0.025
                ):
                    policy.repeated_rejection_streak += 1
                else:
                    policy.repeated_rejection_streak = 1
                policy.last_rejected_compatibility = response_ratio
                preference.reservation_ratio_low = max(
                    preference.reservation_ratio_low,
                    min(response_ratio, preference.reservation_ratio_high) * 0.82,
                )
                if observation.response_type == "counter":
                    policy.counter_rate = 0.8 * policy.counter_rate + 0.2
            belief.direct_response_count += 1
            evidence_strength = 1.0

        if observation.response_type in {"offer", "counter"} and observed_ratio is not None:
            # A seller ask is normally an upper bound, not a direct disclosure
            # of reservation cost.  The conservative multiplier avoids treating
            # the first anchor as ground truth.
            inferred_upper = _clamp(observed_ratio, 0.05, 1.5)
            preference.reservation_ratio_high = min(preference.reservation_ratio_high, inferred_upper)
            soft_center = max(preference.reservation_ratio_low, inferred_upper * 0.82)
            preference.reservation_ratio_mean = 0.85 * preference.reservation_ratio_mean + 0.15 * soft_center
            evidence_strength = max(evidence_strength, 0.55)

        if observation.response_type == "quit":
            policy.quit_rate = 0.7 * policy.quit_rate + 0.3
            policy.patience = max(0.0, policy.patience - 0.15)
            evidence_strength = 1.0

        if observation.offer is not None:
            self._update_issue_preferences(belief, observation)

        if evidence_strength > 0:
            belief.evidence_count += 1
            direct_reliability = 1.0 - math.exp(-belief.direct_response_count / 4.0)
            total_reliability = 1.0 - math.exp(-belief.evidence_count / 6.0)
            preference.reliability = min(0.90, 0.15 + 0.65 * total_reliability)
            policy.reliability = min(0.92, 0.10 + 0.78 * direct_reliability)
            belief.evidence.append(
                {
                    "observation_id": observation.observation_id,
                    "turn": observation.turn,
                    "type": observation.response_type,
                    "source": "structured_action",
                    "strength": evidence_strength,
                }
            )
            belief.evidence = belief.evidence[-30:]

        StructuredBeliefUpdater._repair_bounds(belief)

    @staticmethod
    def _price_ratio(offer: Optional[CanonicalOffer], state: CanonicalState) -> Optional[float]:
        if offer is None or offer.price is None or state.own_value_scale <= 0:
            return None
        return _clamp(float(offer.price) / float(state.own_value_scale), 0.0, 1.5)

    @staticmethod
    def _prior_acceptance_probability(belief: OpponentBelief, compatibility: float) -> float:
        preference = belief.preference
        policy = belief.response_policy
        structural = 1.0 / (1.0 + math.exp(-10.0 * (compatibility - preference.reservation_ratio_mean)))
        empirical = policy.acceptance_rate(_clamp(compatibility))
        return policy.reliability * empirical + (1.0 - policy.reliability) * structural

    @staticmethod
    def _update_change_point(belief: OpponentBelief, surprise: float) -> None:
        if surprise >= 0.65:
            belief.regime_change_probability = min(0.95, 0.65 * belief.regime_change_probability + 0.35 * surprise)
        else:
            belief.regime_change_probability *= 0.75
        belief.new_regime_weight = belief.regime_change_probability
        belief.old_regime_weight = 1.0 - belief.new_regime_weight
        if belief.regime_change_probability >= 0.60:
            # Temper, rather than erase, old evidence.  This is the explicit
            # old/new regime mixture used in switch experiments.
            belief.preference.reliability *= 0.65
            belief.response_policy.reliability *= 0.55

    @staticmethod
    def _update_issue_preferences(
        belief: OpponentBelief,
        observation: NegotiationObservation,
    ) -> None:
        assert observation.offer is not None
        offer = observation.offer
        response_type = observation.response_type
        weight = 1.0 if response_type in {"offer", "counter", "accept"} else 0.25
        for issue, value in offer.discrete_terms.items():
            scores = belief.preference.issue_option_scores.setdefault(issue, {})
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            scores[key] = scores.get(key, 0.0) + weight
        continuous_deltas = observation.metadata.get("continuous_deltas") or {}
        for issue, delta in continuous_deltas.items():
            numeric = _number(delta)
            if numeric is None or abs(numeric) <= 1e-12:
                continue
            previous = belief.preference.issue_directions.get(issue, 0.0)
            belief.preference.issue_directions[issue] = 0.8 * previous + 0.2 * (1.0 if numeric > 0 else -1.0)

    @staticmethod
    def _repair_bounds(belief: OpponentBelief) -> None:
        preference = belief.preference
        if preference.reservation_ratio_low > preference.reservation_ratio_high:
            middle = (preference.reservation_ratio_low + preference.reservation_ratio_high) / 2.0
            preference.reservation_ratio_low = max(0.0, middle - 0.05)
            preference.reservation_ratio_high = min(1.5, middle + 0.05)
        preference.reservation_ratio_mean = min(
            preference.reservation_ratio_high,
            max(preference.reservation_ratio_low, preference.reservation_ratio_mean),
        )


class CensoredBehaviorBeliefUpdater(StructuredBeliefUpdater):
    """Separate latent utility from observable bargaining behavior.

    A rejection is strong response-policy evidence but only weak, censored
    utility evidence because strategic sellers reject profitable offers. A
    counteroffer level and its movement are tracked as asking policy instead of
    being copied into the reservation posterior.
    """

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        for observation in observations:
            preference = belief.preference
            policy = belief.response_policy
            old_low = preference.reservation_ratio_low
            old_mean = preference.reservation_ratio_mean
            old_high = preference.reservation_ratio_high
            previous_ask = policy.last_observed_ask

            # Reuse protocol bookkeeping, acceptance bins, reliability, quit,
            # and change-point logic. Then replace the overly strong utility
            # inference with the censored update below.
            super().update(belief, state, [observation])
            tested = self._price_ratio(observation.response_to_offer, state)
            ask = self._price_ratio(observation.offer, state)

            if observation.response_type in {"reject", "counter"} and tested is not None:
                strategic = _clamp(0.72 - 0.32 * state.progress)
                if observation.response_type == "counter":
                    strategic = min(0.88, strategic + 0.10)
                policy.strategic_rejection_probability = (
                    0.75 * policy.strategic_rejection_probability + 0.25 * strategic
                )
                soft_lower = tested * (0.58 + 0.22 * (1.0 - strategic))
                preference.reservation_ratio_low = max(old_low, soft_lower)
                weak_target = max(old_mean, soft_lower)
                preference.reservation_ratio_mean = 0.92 * old_mean + 0.08 * weak_target

            if observation.response_type in {"offer", "counter"} and ask is not None:
                policy.minimum_observed_ask = (
                    ask if policy.minimum_observed_ask is None
                    else min(policy.minimum_observed_ask, ask)
                )
                policy.maximum_observed_ask = (
                    ask if policy.maximum_observed_ask is None
                    else max(policy.maximum_observed_ask, ask)
                )
                if previous_ask is not None:
                    concession = previous_ask - ask
                    policy.ask_concession_ema = 0.65 * policy.ask_concession_ema + 0.35 * concession
                policy.last_observed_ask = ask

                # The ask is a soft upper bound on reservation. Rigidity is a
                # policy feature and only weakly shifts the utility center.
                preference.reservation_ratio_high = min(old_high, ask)
                low = preference.reservation_ratio_low
                high = max(low, preference.reservation_ratio_high)
                rigidity = _clamp(0.5 - 4.0 * policy.ask_concession_ema)
                interval_target = low + (0.30 + 0.18 * rigidity) * (high - low)
                preference.reservation_ratio_mean = (
                    0.82 * preference.reservation_ratio_mean + 0.18 * interval_target
                )

            self._repair_bounds(belief)


class StrategicCensoredBeliefUpdater(CensoredBehaviorBeliefUpdater):
    """V5.4 treats rejection chiefly as policy, not utility, evidence.

    A strategic reject/counter has fractional likelihood weight in the
    acceptance model. It does not move the utility interval. Observable asks
    may still provide an upper bound, and later verified language may refine
    the broad prior.
    """

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        for observation in observations:
            pref = belief.preference
            old_utility = (
                pref.reservation_ratio_low,
                pref.reservation_ratio_mean,
                pref.reservation_ratio_high,
            )
            before_beta = list(belief.response_policy.accept_beta)
            super().update(belief, state, [observation])

            if observation.response_type in {"reject", "counter"}:
                tested = self._price_ratio(observation.response_to_offer, state)
                if tested is not None:
                    index = min(4, max(0, int(tested * 5)))
                    raw_increment = belief.response_policy.accept_beta[index] - before_beta[index]
                    weight = 0.35 if observation.response_type == "reject" else 0.20
                    belief.response_policy.accept_beta[index] = (
                        before_beta[index] + weight * max(0.0, raw_increment)
                    )

                # Do not infer cost from a strategic refusal. Preserve only an
                # ask-derived upper bound when a formal counter is present.
                pref.reservation_ratio_low = old_utility[0]
                pref.reservation_ratio_mean = old_utility[1]
                pref.reservation_ratio_high = old_utility[2]
                ask = self._price_ratio(observation.offer, state)
                if observation.response_type == "counter" and ask is not None:
                    pref.reservation_ratio_high = min(pref.reservation_ratio_high, ask)
                self._repair_bounds(belief)


class LatentProfileMixtureBeliefUpdater(StrategicCensoredBeliefUpdater):
    """V5.7 soft Bayesian utility x response-policy inference.

    The fixed hypothesis library lives in normalized compatibility space and
    is therefore shared by price, contract, allocation, and multi-issue
    adapters.  A rejection is a likelihood observation, never a hard utility
    bound.  The library contains no evaluator profile identifiers or private
    values, and a small hazard keeps the posterior recoverable under regime
    change or model misspecification.
    """

    RESERVATION_RATIOS = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
    POLICY_TYPES = {
        "patient": (0.92, 0.02, 0.20),
        "neutral": (0.78, 0.08, 0.25),
        "firm": (0.48, 0.28, 0.75),
    }
    ACCEPTANCE_READINESS = (1.0,)

    def __init__(
        self,
        *,
        likelihood_tau: float = 0.05,
        hazard: float = 0.02,
        readiness_progress_power: float = 1.0,
        readiness_terminal_fraction: float = 1.0,
        quit_progress_weight: float = 0.32,
        quit_finality_weight: float = 0.22,
        counter_progress_decay: float = 0.35,
        likelihood_power: float = 1.0,
    ):
        self.likelihood_tau = max(0.01, float(likelihood_tau))
        self.hazard = _clamp(hazard)
        self.readiness_progress_power = max(0.05, float(readiness_progress_power))
        self.readiness_terminal_fraction = _clamp(readiness_terminal_fraction)
        self.quit_progress_weight = max(0.0, float(quit_progress_weight))
        self.quit_finality_weight = max(0.0, float(quit_finality_weight))
        self.counter_progress_decay = _clamp(counter_progress_decay)
        self.likelihood_power = max(0.05, float(likelihood_power))

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        observations = list(observations)
        super().update(belief, state, observations)
        self._ensure_profiles(belief)
        for observation in observations:
            tested = self._price_ratio(observation.response_to_offer, state)
            outcome = self._outcome(observation.response_type)
            if tested is None or outcome is None:
                continue
            progress = _clamp(observation.turn / max(1, state.max_turns))
            self._bayes_update(belief, tested, outcome, progress)
        self._project_moments(belief)

    @classmethod
    def _profile_rows(cls) -> List[Tuple[str, float, float, float, float, float]]:
        rows: List[Tuple[str, float, float, float, float, float]] = []
        for ratio in cls.RESERVATION_RATIOS:
            for policy_name, (counter, quit_bias, finality) in cls.POLICY_TYPES.items():
                for readiness in cls.ACCEPTANCE_READINESS:
                    suffix = (
                        ""
                        if len(cls.ACCEPTANCE_READINESS) == 1 and readiness == 1.0
                        else f"_readiness_{readiness:.2f}"
                    )
                    rows.append(
                        (
                            f"reservation_{ratio:.2f}_{policy_name}{suffix}",
                            ratio,
                            counter,
                            quit_bias,
                            finality,
                            readiness,
                        )
                    )
        return rows

    @classmethod
    def _ensure_profiles(cls, belief: OpponentBelief) -> None:
        rows = cls._profile_rows()
        belief.latent_profile_reservation_ratios = {
            profile_id: ratio for profile_id, ratio, _, _, _, _ in rows
        }
        if set(belief.latent_profile_weights) == {row[0] for row in rows}:
            total = sum(max(0.0, value) for value in belief.latent_profile_weights.values())
            if total > 0:
                belief.latent_profile_weights = {
                    key: max(0.0, value) / total
                    for key, value in belief.latent_profile_weights.items()
                }
                return
        uniform = 1.0 / len(rows)
        belief.latent_profile_weights = {row[0]: uniform for row in rows}

    @staticmethod
    def _outcome(response_type: str) -> Optional[str]:
        if response_type == "accept":
            return "accept"
        if response_type in {"reject", "counter"}:
            return "counter"
        if response_type == "quit":
            return "quit"
        return None

    def _response_distribution(
        self,
        compatibility: float,
        reservation: float,
        counter_propensity: float,
        quit_bias: float,
        finality: float,
        acceptance_readiness: float,
        progress: float,
    ) -> Dict[str, float]:
        delta = (compatibility - reservation) / self.likelihood_tau
        if delta >= 0:
            accept = 1.0 / (1.0 + math.exp(-delta))
        else:
            exp_delta = math.exp(delta)
            accept = exp_delta / (1.0 + exp_delta)
        # A rational seller may strategically counter a profitable offer to
        # extract more surplus. Readiness is a latent policy variable, not part
        # of reservation utility, and rises toward one near the deadline.
        progress_credit = (
            self.readiness_terminal_fraction
            * _clamp(progress) ** self.readiness_progress_power
        )
        readiness = _clamp(
            acceptance_readiness + (1.0 - acceptance_readiness) * progress_credit
        )
        accept *= readiness
        remaining = 1.0 - accept
        quit_share = _clamp(
            quit_bias
            + self.quit_progress_weight * progress
            + self.quit_finality_weight * finality,
            0.01,
            0.95,
        )
        counter_share = _clamp(
            counter_propensity * (1.0 - self.counter_progress_decay * progress),
            0.01,
            max(0.01, 0.99 - quit_share),
        )
        raw = {
            "accept": accept,
            "counter": remaining * counter_share,
            "quit": remaining * (1.0 - counter_share),
        }
        total = sum(raw.values())
        return {key: value / max(1e-12, total) for key, value in raw.items()}

    def _bayes_update(
        self,
        belief: OpponentBelief,
        compatibility: float,
        outcome: str,
        progress: float,
    ) -> None:
        rows = self._profile_rows()
        uniform = 1.0 / len(rows)
        unnormalized: Dict[str, float] = {}
        predictive = 0.0
        for profile_id, reservation, counter, quit_bias, finality, readiness in rows:
            prior = (
                (1.0 - self.hazard) * belief.latent_profile_weights[profile_id]
                + self.hazard * uniform
            )
            likelihood = self._response_distribution(
                compatibility,
                reservation,
                counter,
                quit_bias,
                finality,
                readiness,
                progress,
            )[outcome] ** self.likelihood_power
            value = prior * max(1e-6, likelihood)
            unnormalized[profile_id] = value
            predictive += value
        belief.latent_profile_weights = {
            key: value / max(1e-12, predictive)
            for key, value in unnormalized.items()
        }
        belief.evidence.append(
            {
                "source": "latent_profile_likelihood",
                "type": outcome,
                "tested_compatibility": compatibility,
                "predictive_probability": predictive,
                "hazard": self.hazard,
            }
        )
        belief.evidence = belief.evidence[-30:]

    def _project_moments(self, belief: OpponentBelief) -> None:
        rows = self._profile_rows()
        weights = belief.latent_profile_weights
        if not weights:
            return
        ratio_mass: Dict[float, float] = {}
        expected_counter = 0.0
        expected_quit = 0.0
        expected_finality = 0.0
        entropy = 0.0
        for profile_id, ratio, counter, quit_bias, finality, _ in rows:
            weight = weights[profile_id]
            ratio_mass[ratio] = ratio_mass.get(ratio, 0.0) + weight
            expected_counter += weight * counter
            expected_quit += weight * quit_bias
            expected_finality += weight * finality
            if weight > 0:
                entropy -= weight * math.log2(weight)
        mean = sum(ratio * mass for ratio, mass in ratio_mass.items())

        def quantile(level: float) -> float:
            cumulative = 0.0
            for ratio in sorted(ratio_mass):
                cumulative += ratio_mass[ratio]
                if cumulative + 1e-12 >= level:
                    return ratio
            return max(ratio_mass)

        maximum_entropy = math.log2(len(rows))
        concentration = 1.0 - entropy / max(1e-9, maximum_entropy)
        evidence_gate = 1.0 - math.exp(-belief.direct_response_count / 3.0)
        reliability = _clamp(0.10 + 0.82 * evidence_gate * (0.35 + 0.65 * concentration))
        preference = belief.preference
        preference.reservation_ratio_low = quantile(0.10)
        preference.reservation_ratio_mean = mean
        preference.reservation_ratio_high = quantile(0.90)
        preference.reliability = reliability
        policy = belief.response_policy
        policy.counter_rate = expected_counter
        policy.quit_rate = expected_quit
        policy.patience = 1.0 - expected_finality
        policy.reliability = max(policy.reliability, reliability * 0.92)
        belief.latent_profile_entropy_bits = entropy
        self._repair_bounds(belief)


class StrategicReadinessMixtureBeliefUpdater(LatentProfileMixtureBeliefUpdater):
    """V5.9 adds latent strategic acceptance readiness.

    Different hypotheses can share the same reservation utility while varying
    in their willingness to accept a profitable offer early. This prevents a
    strategic counter from being explained only by an inflated reservation.
    """

    ACCEPTANCE_READINESS = (0.25, 0.50, 0.75, 1.00)

    @classmethod
    def from_calibration_json(
        cls,
        path: str | Path,
        *,
        hazard: float = 0.02,
    ) -> "StrategicReadinessMixtureBeliefUpdater":
        """Load only allowlisted likelihood parameters from a frozen artifact."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        params = payload.get("params") or {}
        allowed = {
            "likelihood_tau",
            "readiness_progress_power",
            "readiness_terminal_fraction",
            "quit_progress_weight",
            "quit_finality_weight",
            "counter_progress_decay",
        }
        kwargs = {
            key: float(value)
            for key, value in params.items()
            if key in allowed and isinstance(value, (int, float))
        }
        kwargs["likelihood_power"] = float(payload.get("likelihood_power", 1.0))
        kwargs["hazard"] = hazard
        return cls(**kwargs)


class ReservationAspirationMixtureBeliefUpdater(LatentProfileMixtureBeliefUpdater):
    """V6 posterior that separates feasible utility from strategic aspiration.

    ``reservation`` is the counterparty's durable individual-rationality
    boundary. ``aspiration_margin`` is a response-policy target above that
    boundary.  A hard bargainer can therefore counter a profitable offer
    without forcing the reservation posterior upward.  The representation is
    defined in normalized opponent-value space and is not price-specific.

    Aspiration decays slowly for a firm policy and faster for a patient one.
    At the terminal decision it collapses to reservation, preserving the
    individual-rationality interpretation of that variable.
    """

    ASPIRATION_MARGINS = (0.00, 0.25, 0.50, 0.75)

    @classmethod
    def _profile_rows(cls) -> List[Tuple[str, float, float, float, float, float]]:
        rows: List[Tuple[str, float, float, float, float, float]] = []
        for ratio in cls.RESERVATION_RATIOS:
            for policy_name, (counter, quit_bias, finality) in cls.POLICY_TYPES.items():
                for margin in cls.ASPIRATION_MARGINS:
                    rows.append(
                        (
                            f"reservation_{ratio:.2f}_{policy_name}_aspiration_{margin:.2f}",
                            ratio,
                            counter,
                            quit_bias,
                            finality,
                            margin,
                        )
                    )
        return rows

    @classmethod
    def _ensure_profiles(cls, belief: OpponentBelief) -> None:
        super()._ensure_profiles(belief)
        belief.latent_profile_aspiration_margins = {
            profile_id: margin
            for profile_id, _, _, _, _, margin in cls._profile_rows()
        }

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        # ``_project_moments`` is called by the parent after every batch.  The
        # current decision progress is needed to project a time-varying target.
        self._projection_progress = state.progress
        super().update(belief, state, observations)

    @staticmethod
    def _aspiration_threshold(
        reservation: float,
        margin: float,
        finality: float,
        progress: float,
    ) -> float:
        # Firm policies (large finality) preserve their target until late;
        # patient policies concede more smoothly.  No benchmark-specific turn
        # count or currency appears in this equation.
        exponent = 1.0 + 4.0 * _clamp(finality)
        remaining_target = 1.0 - _clamp(progress) ** exponent
        return _clamp(reservation + margin * remaining_target, 0.0, 1.25)

    def _response_distribution(
        self,
        compatibility: float,
        reservation: float,
        counter_propensity: float,
        quit_bias: float,
        finality: float,
        aspiration_margin: float,
        progress: float,
    ) -> Dict[str, float]:
        target = self._aspiration_threshold(
            reservation, aspiration_margin, finality, progress
        )
        delta = (compatibility - target) / self.likelihood_tau
        if delta >= 0:
            accept = 1.0 / (1.0 + math.exp(-delta))
        else:
            exp_delta = math.exp(delta)
            accept = exp_delta / (1.0 + exp_delta)
        remaining = 1.0 - accept
        below_reservation = _clamp(
            (reservation - compatibility) / max(self.likelihood_tau, 1e-9)
        )
        quit_share = _clamp(
            quit_bias
            + 0.18 * progress
            + 0.16 * finality
            + 0.12 * below_reservation,
            0.01,
            0.95,
        )
        counter_share = _clamp(
            counter_propensity * (1.0 - 0.20 * progress),
            0.01,
            max(0.01, 0.99 - quit_share),
        )
        raw = {
            "accept": accept,
            "counter": remaining * counter_share,
            "quit": remaining * (1.0 - counter_share),
        }
        total = sum(raw.values())
        return {key: value / max(1e-12, total) for key, value in raw.items()}

    def _project_moments(self, belief: OpponentBelief) -> None:
        super()._project_moments(belief)
        rows = self._profile_rows()
        weights = belief.latent_profile_weights
        if not weights:
            return
        progress = _clamp(getattr(self, "_projection_progress", 0.0))
        target_mass: Dict[float, float] = {}
        expected_margin = 0.0
        for profile_id, ratio, _, _, finality, margin in rows:
            weight = weights[profile_id]
            target = self._aspiration_threshold(ratio, margin, finality, progress)
            target_mass[target] = target_mass.get(target, 0.0) + weight
            expected_margin += weight * margin

        def quantile(level: float) -> float:
            cumulative = 0.0
            for target in sorted(target_mass):
                cumulative += target_mass[target]
                if cumulative + 1e-12 >= level:
                    return target
            return max(target_mass)

        policy = belief.response_policy
        policy.aspiration_ratio_low = quantile(0.10)
        policy.aspiration_ratio_mean = sum(
            target * mass for target, mass in target_mass.items()
        )
        policy.aspiration_ratio_high = quantile(0.90)
        policy.aspiration_margin_mean = expected_margin
        policy.aspiration_reliability = belief.preference.reliability
        policy.strategic_rejection_probability = _clamp(
            expected_margin / max(self.ASPIRATION_MARGINS)
        )


class CensoredReservationAspirationBeliefUpdater(
    ReservationAspirationMixtureBeliefUpdater
):
    """V6.1 partial-identification update for strategically censored actions.

    Reject/counter reveals that an offer was not selected, but generally does
    not reveal that it was below reservation.  Its utility likelihood is
    therefore tempered.  A formal counteroffer is a soft upper bound on
    reservation and is modeled separately from the response event.  This
    prevents repeated hardball counters from manufacturing a precise, high
    utility threshold while retaining useful bracketing evidence.
    """

    COUNTER_UTILITY_POWER = 0.18
    QUIT_UTILITY_POWER = 0.45
    ASK_BOUND_POWER = 0.75

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: Iterable[NegotiationObservation],
    ) -> None:
        observations = list(observations)
        self._projection_progress = state.progress
        # Preserve response-policy/frontier bookkeeping, but do not invoke the
        # uncensored latent likelihood from the parent class.
        StrategicCensoredBeliefUpdater.update(self, belief, state, observations)
        self._ensure_profiles(belief)
        for observation in observations:
            tested = self._price_ratio(observation.response_to_offer, state)
            outcome = self._outcome(observation.response_type)
            if tested is None or outcome is None:
                continue
            progress = _clamp(observation.turn / max(1, state.max_turns))
            ask = self._price_ratio(observation.offer, state)
            self._censored_bayes_update(
                belief,
                tested,
                outcome,
                observation.response_type,
                progress,
                ask,
            )
        self._project_moments(belief)

    def _censored_bayes_update(
        self,
        belief: OpponentBelief,
        compatibility: float,
        outcome: str,
        response_type: str,
        progress: float,
        ask_ratio: Optional[float],
    ) -> None:
        rows = self._profile_rows()
        uniform = 1.0 / len(rows)
        unnormalized: Dict[str, float] = {}
        predictive = 0.0
        for profile_id, reservation, counter, quit_bias, finality, margin in rows:
            prior = (
                (1.0 - self.hazard) * belief.latent_profile_weights[profile_id]
                + self.hazard * uniform
            )
            response_likelihood = self._response_distribution(
                compatibility,
                reservation,
                counter,
                quit_bias,
                finality,
                margin,
                progress,
            )[outcome]
            if response_type in {"reject", "counter"}:
                likelihood = max(1e-6, response_likelihood) ** self.COUNTER_UTILITY_POWER
            elif response_type == "quit":
                likelihood = max(1e-6, response_likelihood) ** self.QUIT_UTILITY_POWER
            else:
                likelihood = max(1e-6, response_likelihood)

            # An ask is compatible with any lower reservation.  It is not a
            # point disclosure, so use a soft one-sided likelihood.
            if response_type == "counter" and ask_ratio is not None:
                delta = (ask_ratio - reservation) / max(self.likelihood_tau, 1e-9)
                if delta >= 0:
                    upper_bound = 1.0 / (1.0 + math.exp(-delta))
                else:
                    exp_delta = math.exp(delta)
                    upper_bound = exp_delta / (1.0 + exp_delta)
                likelihood *= max(1e-6, upper_bound) ** self.ASK_BOUND_POWER

            value = prior * max(1e-6, likelihood)
            unnormalized[profile_id] = value
            predictive += value
        belief.latent_profile_weights = {
            key: value / max(1e-12, predictive)
            for key, value in unnormalized.items()
        }
        belief.evidence.append(
            {
                "source": "censored_reservation_aspiration_likelihood",
                "type": response_type,
                "tested_compatibility": compatibility,
                "formal_ask_upper_bound": ask_ratio,
                "counter_utility_power": self.COUNTER_UTILITY_POWER,
                "ask_bound_power": self.ASK_BOUND_POWER,
                "predictive_probability": predictive,
                "hazard": self.hazard,
            }
        )
        belief.evidence = belief.evidence[-30:]


class SemanticBeliefUpdater:
    """Optional LLM evidence extractor whose influence is explicitly bounded."""

    def __init__(self, client: TextGenerator, max_tokens: int = 700):
        self.client = client
        self.max_tokens = max_tokens

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: List[NegotiationObservation],
    ) -> Optional[Dict[str, Any]]:
        text_observations = [item for item in observations if item.text.strip()]
        if not text_observations:
            return None
        payload = [
            {
                "turn": item.turn,
                "response_type": item.response_type,
                "text": item.text[:1200],
                "offer": item.offer.to_dict() if item.offer else None,
            }
            for item in text_observations[-4:]
        ]
        prompt = f"""You extract bounded opponent-preference evidence from negotiation dialogue.

The environment may involve price, contracts, products, allocations, or several
counterparties.  Extract evidence only; do not choose an action and do not infer
private values with false precision.

Issues: {json.dumps([issue.__dict__ for issue in state.issues], ensure_ascii=False, default=str)}
New observations: {json.dumps(payload, ensure_ascii=False, default=str)}

Return one JSON object:
{{
  "reservation_ratio_range": [low, high] or null,
  "issue_signals": [
    {{"issue": "name", "preferred_value": "value or null", "direction": -1|0|1, "confidence": 0-1}}
  ],
  "patience": 0-1 or null,
  "quit_risk": 0-1 or null,
  "evidence": ["short observation grounded in the text"]
}}

The reservation ratio is opponent reservation divided by the focal agent's
value scale.  Omit it unless the language plus formal offer gives evidence.
Language such as 'final' or 'below cost' is weak evidence by itself.
"""
        raw = self.client.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
        parsed = self._json_object(raw)
        if parsed is None:
            return {"parsed": False, "raw": raw[:500]}
        gate = min(0.22, 0.05 + 0.035 * belief.direct_response_count)
        self._merge(parsed, belief, gate)
        belief.semantic_evidence_count += 1
        return {"parsed": True, "gate": gate, "proposal": parsed, "raw": raw[:800]}

    @staticmethod
    def _json_object(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text or ""):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _merge(parsed: Dict[str, Any], belief: OpponentBelief, gate: float) -> None:
        preference = belief.preference
        interval = parsed.get("reservation_ratio_range")
        if isinstance(interval, list) and len(interval) == 2:
            low = _number(interval[0])
            high = _number(interval[1])
            if low is not None and high is not None:
                low, high = sorted((_clamp(low, 0.0, 1.5), _clamp(high, 0.0, 1.5)))
                preference.reservation_ratio_low = (1.0 - gate) * preference.reservation_ratio_low + gate * low
                preference.reservation_ratio_high = (1.0 - gate) * preference.reservation_ratio_high + gate * high
        for signal in parsed.get("issue_signals") or []:
            if not isinstance(signal, dict):
                continue
            issue = str(signal.get("issue") or "").strip()
            confidence = _clamp(_number(signal.get("confidence")) or 0.0)
            if not issue or confidence <= 0:
                continue
            local_gate = gate * confidence
            preferred = signal.get("preferred_value")
            if preferred is not None:
                scores = preference.issue_option_scores.setdefault(issue, {})
                key = json.dumps(preferred, ensure_ascii=False, sort_keys=True)
                scores[key] = scores.get(key, 0.0) + local_gate
            direction = _number(signal.get("direction"))
            if direction is not None:
                old = preference.issue_directions.get(issue, 0.0)
                preference.issue_directions[issue] = (1.0 - local_gate) * old + local_gate * _clamp(direction, -1, 1)
        patience = _number(parsed.get("patience"))
        if patience is not None:
            belief.response_policy.patience = (1.0 - gate) * belief.response_policy.patience + gate * _clamp(patience)
        quit_risk = _number(parsed.get("quit_risk"))
        if quit_risk is not None:
            belief.response_policy.quit_rate = (1.0 - gate) * belief.response_policy.quit_rate + gate * _clamp(quit_risk)
        for evidence in (parsed.get("evidence") or [])[:4]:
            belief.evidence.append({"source": "semantic", "text": str(evidence)[:300], "strength": gate})
        preference.reliability = min(0.92, preference.reliability + gate * 0.10)
        StructuredBeliefUpdater._repair_bounds(belief)


class CenteredSemanticBeliefUpdater(SemanticBeliefUpdater):
    """V2 updater that moves the posterior mean with a bounded interval gate."""

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: List[NegotiationObservation],
    ) -> Optional[Dict[str, Any]]:
        before = belief.preference.reservation_ratio_mean
        result = super().update(belief, state, observations)
        if not result or not result.get("parsed"):
            return result
        proposal = result.get("proposal") or {}
        interval = proposal.get("reservation_ratio_range")
        if isinstance(interval, list) and len(interval) == 2:
            low = _number(interval[0])
            high = _number(interval[1])
            if low is not None and high is not None:
                low, high = sorted((_clamp(low, 0.0, 1.5), _clamp(high, 0.0, 1.5)))
                gate = min(0.22, max(0.0, float(result.get("gate") or 0.0)))
                center = (low + high) / 2.0
                preference = belief.preference
                preference.reservation_ratio_mean = (1.0 - gate) * before + gate * center
                StructuredBeliefUpdater._repair_bounds(belief)
                result["mean_update"] = {
                    "before": before,
                    "interval_center": center,
                    "after": preference.reservation_ratio_mean,
                    "gate": gate,
                }
        return result


class VerifiedSemanticBeliefUpdater(SemanticBeliefUpdater):
    """Admit public-language evidence only after behavioral corroboration.

    V5 still calls an LLM to interpret dialogue. A verbal reservation claim can
    affect utility only after two formal responses and only where its interval
    overlaps the structured posterior. This is intermediate between trusting
    claims directly (V2) and never using them (V3/V4).
    """

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: List[NegotiationObservation],
    ) -> Optional[Dict[str, Any]]:
        shadow = copy.deepcopy(belief)
        result = super().update(shadow, state, observations)
        if not result or not result.get("parsed"):
            return result

        result["applied_to_planner_belief"] = False
        result["verification"] = "insufficient_formal_evidence"
        proposal = result.get("proposal") or {}
        interval = proposal.get("reservation_ratio_range")
        if belief.direct_response_count < 2 or not isinstance(interval, list) or len(interval) != 2:
            return result
        low = _number(interval[0])
        high = _number(interval[1])
        if low is None or high is None:
            result["verification"] = "invalid_interval"
            return result
        low, high = sorted((_clamp(low, 0.0, 1.5), _clamp(high, 0.0, 1.5)))
        pref = belief.preference
        overlap_low = max(low, pref.reservation_ratio_low)
        overlap_high = min(high, pref.reservation_ratio_high)
        if overlap_low > overlap_high:
            result["verification"] = "contradicts_structured_interval"
            return result

        gate = min(0.10, 0.025 + 0.015 * belief.direct_response_count)
        center = 0.5 * (overlap_low + overlap_high)
        before = pref.reservation_ratio_mean
        pref.reservation_ratio_mean = (1.0 - gate) * before + gate * center
        pref.reservation_ratio_low = (1.0 - gate) * pref.reservation_ratio_low + gate * overlap_low
        pref.reservation_ratio_high = (1.0 - gate) * pref.reservation_ratio_high + gate * overlap_high
        pref.reliability = min(0.90, pref.reliability + 0.04 * gate)
        belief.semantic_evidence_count += 1
        belief.evidence.append({
            "source": "verified_semantic",
            "strength": gate,
            "interval": [low, high],
            "corroborated_interval": [overlap_low, overlap_high],
        })
        StructuredBeliefUpdater._repair_bounds(belief)
        result.update({
            "applied_to_planner_belief": True,
            "verification": "behaviorally_corroborated_overlap",
            "verified_gate": gate,
            "mean_update": {"before": before, "after": pref.reservation_ratio_mean},
        })
        return result


class ShadowSemanticBeliefUpdater(SemanticBeliefUpdater):
    """Extract an LLM belief proposal without mutating the behavioral belief.

    This is the conservative V3 gate: semantic claims remain fully logged and
    can later be admitted by a calibrated verifier, while formal actions remain
    the planner-controlling evidence in the current episode.
    """

    def update(
        self,
        belief: OpponentBelief,
        state: CanonicalState,
        observations: List[NegotiationObservation],
    ) -> Optional[Dict[str, Any]]:
        shadow = OpponentBelief(counterparty_id=belief.counterparty_id)
        # Start from the current posterior so the extractor sees the same
        # evidence gate, but merge only into the copy.
        shadow.preference = copy.deepcopy(belief.preference)
        shadow.response_policy = copy.deepcopy(belief.response_policy)
        shadow.evidence_count = belief.evidence_count
        shadow.direct_response_count = belief.direct_response_count
        shadow.semantic_evidence_count = belief.semantic_evidence_count
        shadow.evidence = list(belief.evidence)
        result = super().update(shadow, state, observations)
        if result is not None:
            result["applied_to_planner_belief"] = False
            result["gate_reason"] = "await_behavioral_verification"
            result["shadow_preference"] = shadow.preference.__dict__.copy()
        return result
