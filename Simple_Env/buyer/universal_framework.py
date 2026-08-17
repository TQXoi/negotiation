"""Simple Env adapter for the cross-environment universal framework."""

from __future__ import annotations

import math
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    format_action_message,
    parse_action,
    validate_buyer_action,
)
from experiments.model_clients import ModelClient
from framework import (
    ActiveFrontierPlanner,
    AspirationAwareFrontierPlanner,
    AspirationProbeFrontierPlanner,
    BehavioralFrontierPlanner,
    BehavioralBeliefUsablePlanner,
    BroadPriorBeliefStore,
    CandidateAction,
    CanonicalOffer,
    CanonicalState,
    ConservativeAWRPlanner,
    CensoredBehaviorBeliefUpdater,
    CensoredReservationAspirationBeliefUpdater,
    CenteredSemanticBeliefUpdater,
    DeadlineAwareBeliefUsablePlanner,
    DominanceAndOptionPlanner,
    IssueSpec,
    LookaheadBeliefUsablePlanner,
    LatentProfileMixtureBeliefUpdater,
    NegotiationObservation,
    OpponentBelief,
    PosteriorIntegratedFrontierPlanner,
    ReservationAspirationMixtureBeliefUpdater,
    ShadowSemanticBeliefUpdater,
    StrategicCensoredBeliefUpdater,
    StrategicReadinessMixtureBeliefUpdater,
    TerminalSafeLookaheadPlanner,
    TrainableDecisionBoundaryPlanner,
    UniversalNegotiationEngine,
    VerifiedSemanticBeliefUpdater,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class SimpleEnvAdapter:
    """Translate single-price RLVR negotiation into canonical actions.

    This class is intentionally limited to environment semantics.  Belief
    update and candidate ranking live in the shared ``framework`` package
    package, which prevents Simple/AmazonHistoryPrice-specific prompt rules
    from becoming the purported universal method.
    """

    def __init__(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        framework_version: str = "v1",
        observation_namespace: str = "",
        repeated_opponent: bool = False,
        cross_session_progress: float = 0.0,
    ):
        self.scenario = scenario
        self.history = history
        self.framework_version = framework_version
        self.observation_namespace = observation_namespace
        self.repeated_opponent = bool(repeated_opponent)
        self.cross_session_progress = _clamp(cross_session_progress)

    def state(self, round_id: int, max_turns: int, belief_mode: str) -> CanonicalState:
        observations = self._observations()
        metadata: Dict[str, Any] = {
            "quantity": self.scenario.quantity,
            "framework_version": f"universal_belief_planner_{self.framework_version}",
            "repeated_opponent": self.repeated_opponent,
            "cross_session_progress": self.cross_session_progress,
            "counterparty_count": 1,
        }
        if belief_mode in {"oracle", "oracle_utility", "adversarial_wrong"}:
            seller_cost = getattr(self.scenario, "seller_cost", None)
            if seller_cost is not None and self.scenario.buyer_budget:
                metadata["oracle_belief"] = {
                    "reservation_ratio": float(seller_cost) / float(self.scenario.buyer_budget)
                }
        return CanonicalState(
            session_id=str(self.scenario.item_id),
            environment_id="simple_price_negotiation",
            self_id="buyer",
            role="buyer",
            counterparty_id="seller",
            turn=round_id,
            max_turns=max_turns,
            issues=[IssueSpec(name="price", kind="price", own_weight=-1.0)],
            own_value_scale=float(self.scenario.buyer_budget),
            own_outside_option=0.0,
            reference_value=float(self.scenario.reference_price),
            observations=observations,
            metadata=metadata,
        )

    def candidates(self, state: CanonicalState, belief: OpponentBelief) -> List[CandidateAction]:
        budget = float(self.scenario.buyer_budget)
        reference = float(self.scenario.reference_price)
        seller_offer = self._last_price("seller")
        last_buyer = self._last_price("buyer")
        low = max(0.01, min(reference * 0.34, budget * 0.34))
        high = budget * (0.74 + 0.22 * state.progress)
        if seller_offer is not None:
            high = min(high, seller_offer * 0.995)
        if last_buyer is not None:
            low = max(low, last_buyer)
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        prices = {round(low + (high - low) * ratio, 2) for ratio in ratios}
        if self.framework_version in {"v2", "v3", "v4", "v5", "v5_1", "v5_2", "v5_3", "v5_4", "v5_5", "v5_6", "v5_7", "v5_8", "v5_9", "v6_0", "v6_1"} and belief.direct_response_count:
            # Add belief-targeted frontier points without treating the belief as
            # a hard constraint. The environment adapter still enforces budget
            # and protocol feasibility below.
            preference = belief.preference
            belief_ratios = {
                preference.reservation_ratio_low,
                preference.reservation_ratio_mean,
                preference.reservation_ratio_high,
                0.5 * (
                    preference.reservation_ratio_mean
                    + preference.reservation_ratio_high
                ),
            }
            prices.update(round(budget * ratio, 2) for ratio in belief_ratios)
            for key in preference.issue_option_scores.get("price", {}):
                try:
                    prices.add(round(float(str(key).strip('"')), 2))
                except ValueError:
                    continue
        prices = sorted(prices)
        candidates: List[CandidateAction] = []
        for index, price in enumerate(prices):
            if price <= 0 or price > budget or (seller_offer is not None and price >= seller_offer):
                continue
            if (
                self.framework_version in {"v5_1", "v5_2", "v5_3", "v5_4", "v5_5", "v5_6", "v5_7", "v5_8", "v5_9", "v6_0", "v6_1"}
                and last_buyer is not None
                and price + 1e-9 < last_buyer
            ):
                # A belief-targeted frontier point must not bypass protocol
                # monotonicity after the seller has rejected a higher bid.
                continue
            opponent_proxy = _clamp(price / max(1e-9, budget))
            base_acceptance = self._base_acceptance(price, seller_offer, budget)
            information_gain = _clamp(1.0 - abs(opponent_proxy - belief.preference.reservation_ratio_mean) * 4.0)
            own_utility = budget - price
            novelty = 1.0
            if last_buyer is not None:
                novelty = _clamp(abs(price - last_buyer) / max(0.01, 0.08 * budget))
            candidates.append(
                CandidateAction(
                    candidate_id=f"offer_{index}_{price:g}",
                    action_type="offer",
                    counterparty_id=state.counterparty_id,
                    offer=CanonicalOffer(price=price),
                    own_utility=own_utility,
                    own_utility_normalized=own_utility / max(1e-9, budget),
                    opponent_value_proxy=opponent_proxy,
                    base_acceptance=base_acceptance,
                    information_gain=information_gain,
                    feasibility_margin=own_utility / max(1e-9, budget),
                    rationale="price candidate on a bounded concession frontier",
                    metadata={"action_novelty": novelty},
                )
            )
        if seller_offer is not None and seller_offer <= budget:
            own_utility = budget - seller_offer
            candidates.append(
                CandidateAction(
                    candidate_id=f"accept_{seller_offer:g}",
                    action_type="accept",
                    counterparty_id=state.counterparty_id,
                    offer=CanonicalOffer(price=seller_offer),
                    own_utility=own_utility,
                    own_utility_normalized=own_utility / max(1e-9, budget),
                    opponent_value_proxy=1.0,
                    base_acceptance=1.0,
                    feasibility_margin=own_utility / max(1e-9, budget),
                    rationale="accept the exact outstanding seller offer",
                )
            )
        candidates.append(
            CandidateAction(
                candidate_id="quit",
                action_type="quit",
                counterparty_id=state.counterparty_id,
                offer=None,
                own_utility=0.0,
                own_utility_normalized=0.0,
                opponent_value_proxy=0.0,
                base_acceptance=0.0,
                rationale="exercise outside option",
            )
        )
        return candidates

    def render_locked(self, state, belief, selected, language_client):
        candidate = selected.candidate
        persuasive_text = self._persuasive_text(candidate.action_type, language_client, state, belief)
        if candidate.action_type == "accept" and candidate.offer and candidate.offer.price is not None:
            raw = format_action_message(
                "buyer", "DEAL", candidate.offer.price, self.scenario, persuasive_text
            )
        elif candidate.action_type == "offer" and candidate.offer and candidate.offer.price is not None:
            raw = format_action_message(
                "buyer", "BUY", candidate.offer.price, self.scenario, persuasive_text
            )
        elif candidate.action_type == "quit":
            raw = "Thought: The remaining expected surplus is not positive.\nTalk: I cannot make this work.\nAction: [QUIT]"
        else:
            raw = "Thought: I need a better proposal.\nTalk: Please provide a materially better quote.\nAction: [REJECT]"
        return parse_action("buyer", raw)

    def validate_locked(self, state, selected, rendered_action):
        normalized_history = [
            {**item, "action": self._parsed(item).to_dict()}
            for item in self.history
        ]
        action, validation = validate_buyer_action(
            rendered_action,
            self.scenario,
            normalized_history,
            state.turn,
            state.max_turns,
        )
        locked_price = selected.candidate.offer.price if selected.candidate.offer else None
        expected_action = {
            "offer": "offer",
            "accept": "accept",
            "quit": "walk_away",
            "reject": "reject",
            "ask": "reject",
        }[selected.candidate.action_type]
        lock_ok = action.action == expected_action and (
            locked_price is None
            or (action.price is not None and abs(float(action.price) - float(locked_price)) <= 1e-6)
        )
        if not action.valid or not lock_ok:
            action = self.render_locked(state, OpponentBelief(state.counterparty_id), selected, None)
            action, validation = validate_buyer_action(
                action, self.scenario, normalized_history, state.turn, state.max_turns
            )
        return action, {**validation, "action_lock_ok": lock_ok, "locked_candidate_id": selected.candidate.candidate_id}

    def _observations(self) -> List[NegotiationObservation]:
        observations: List[NegotiationObservation] = []
        prior_buyer: Optional[CanonicalOffer] = None
        for index, item in enumerate(self.history):
            role = str(item.get("role") or "")
            parsed = self._parsed(item)
            if role == "buyer":
                if parsed.price is not None:
                    prior_buyer = CanonicalOffer(price=float(parsed.price))
                continue
            if role != "seller":
                continue
            response_type = "message"
            if parsed.action == "offer":
                response_type = "counter" if prior_buyer is not None else "offer"
            elif parsed.action == "accept":
                response_type = "accept"
            elif parsed.action in {"reject", "walk_away"}:
                response_type = "quit" if parsed.action == "walk_away" else "reject"
            offer = CanonicalOffer(price=float(parsed.price)) if parsed.price is not None else None
            observations.append(
                NegotiationObservation(
                    observation_id=(
                        f"{self.observation_namespace + '_' if self.observation_namespace else ''}seller_{index}_"
                        f"{item.get('round', index)}_{parsed.action}_{parsed.price}"
                    ),
                    turn=int(item.get("round") or index + 1),
                    actor_id="seller",
                    counterparty_id="seller",
                    response_type=response_type,
                    offer=offer,
                    text=(
                        self._public_message(str(item.get("message") or parsed.raw or ""))
                        if self.framework_version in {"v5", "v5_1", "v5_2", "v5_3", "v5_4", "v5_5", "v5_6", "v5_7", "v5_8", "v5_9", "v6_0", "v6_1", "v6_2", "v6_3_cross_environment_awr"}
                        else str(item.get("message") or parsed.raw or "")
                    ),
                    response_to_offer=prior_buyer,
                )
            )
        return observations

    @staticmethod
    def _public_message(message: str) -> str:
        """Expose only public Talk/Action, never a seller scratchpad."""
        talk = re.search(r"(?:^|\n)Talk:\s*(.*?)(?=\nAction:|\Z)", message, re.DOTALL)
        action = re.search(r"(?:^|\n)Action:\s*(.*)", message, re.DOTALL)
        parts: List[str] = []
        if talk:
            parts.append("Talk: " + talk.group(1).strip())
        if action:
            parts.append("Action: " + action.group(1).strip())
        return "\n".join(parts) if parts else message

    @staticmethod
    def _base_acceptance(price: float, seller_offer: Optional[float], budget: float) -> float:
        if seller_offer is None:
            return _clamp(0.12 + 0.35 * price / max(1e-9, budget), 0.05, 0.55)
        gap = (seller_offer - price) / max(1e-9, budget)
        return _clamp(0.72 - 2.5 * gap, 0.04, 0.90)

    @staticmethod
    def _persuasive_text(action_type: str, client, state, belief) -> str:
        fallback = {
            "offer": "This is a serious offer and I can close promptly if you can work with it.",
            "accept": "I can close on your stated offer now.",
            "quit": "I cannot make the remaining terms work.",
        }.get(action_type, "Please improve the proposal.")
        if client is None:
            return fallback
        version = str(state.metadata.get("framework_version") or "")
        prompt = f"""Write one concise buyer negotiation sentence for a {action_type} action.
Do not include numbers, prices, brackets, tags, private values, beliefs, or analysis.
The structured action will be appended by a trusted renderer. Turn {state.turn}/{state.max_turns}.
If this is an offer, make clear it is a counteroffer (not an agreement) and ask
for one concrete sign of flexibility. Do not claim terms are already agreed.
"""
        try:
            text = client.generate(
                prompt,
                temperature=0.0 if version.endswith(("_v3", "_v4", "_v5", "_v5_1", "_v5_2", "_v5_3", "_v5_4", "_v5_5", "_v5_6", "_v5_7", "_v5_8", "_v5_9", "_v6_0", "_v6_1")) else 0.4,
                top_p=1.0,
                max_tokens=90,
            ).strip()
        except Exception:
            return fallback
        if not text or re.search(r"\d|\[|\]", text):
            return fallback
        return text[:400]

    @staticmethod
    def _parsed(item: Dict[str, Any]) -> ParsedAction:
        action = item.get("action")
        if isinstance(action, dict):
            return ParsedAction(
                role=str(action.get("role") or item.get("role") or ""),
                action=str(action.get("action") or ""),
                price=action.get("price"),
                raw=str(action.get("raw") or item.get("message") or ""),
                valid=bool(action.get("valid", True)),
                item_spec=action.get("item_spec"),
            )
        return parse_action(str(item.get("role") or ""), str(item.get("message") or ""))

    def _last_price(self, role: str) -> Optional[float]:
        for item in reversed(self.history):
            if item.get("role") != role:
                continue
            parsed = self._parsed(item)
            if parsed.price is not None:
                return float(parsed.price)
        return None


class UniversalFrameworkBuyer:
    """Simple Env buyer exposing the standard ``act`` interface."""

    belief_mode = "learned"
    framework_version = "v1"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_{self.framework_version}_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            semantic_client=semantic_client,
            language_client=client,
            belief_mode=self.belief_mode,
        )

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> Tuple[ParsedAction, Dict[str, Any]]:
        adapter = SimpleEnvAdapter(scenario, history, self.framework_version)
        state = adapter.state(round_id, max_turns, self.belief_mode)
        decision = self.engine.decide(state, adapter)
        return decision.rendered_action, decision.trace()


class UniversalFrameworkFrozenBuyer(UniversalFrameworkBuyer):
    belief_mode = "frozen"


class UniversalFrameworkWrongBeliefBuyer(UniversalFrameworkBuyer):
    belief_mode = "wrong"


class UniversalFrameworkShuffledBeliefBuyer(UniversalFrameworkBuyer):
    belief_mode = "shuffled"


class UniversalFrameworkOracleBeliefBuyer(UniversalFrameworkBuyer):
    belief_mode = "oracle"


class UniversalFrameworkV2Buyer(UniversalFrameworkBuyer):
    """V2 keeps V1 executable while adding deadline/direct-evidence handling."""

    framework_version = "v2"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v2_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=DeadlineAwareBeliefUsablePlanner(),
            semantic_updater=CenteredSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV2FrozenBuyer(UniversalFrameworkV2Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV2WrongBeliefBuyer(UniversalFrameworkV2Buyer):
    belief_mode = "wrong"


class UniversalFrameworkV2ShuffledBeliefBuyer(UniversalFrameworkV2Buyer):
    belief_mode = "shuffled"


class UniversalFrameworkV2OracleBeliefBuyer(UniversalFrameworkV2Buyer):
    belief_mode = "oracle"


class UniversalFrameworkV3Buyer(UniversalFrameworkBuyer):
    """V3: structured posterior plus logged, behavior-gated LLM proposals."""

    framework_version = "v3"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v3_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=DeadlineAwareBeliefUsablePlanner(),
            semantic_updater=ShadowSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV3FrozenBuyer(UniversalFrameworkV3Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV3WrongBeliefBuyer(UniversalFrameworkV3Buyer):
    belief_mode = "wrong"


class UniversalFrameworkV3ShuffledBeliefBuyer(UniversalFrameworkV3Buyer):
    belief_mode = "shuffled"


class UniversalFrameworkV3OracleBeliefBuyer(UniversalFrameworkV3Buyer):
    belief_mode = "oracle"


class UniversalFrameworkV4Buyer(UniversalFrameworkBuyer):
    """V4 factorizes utility feasibility from behavioral response prediction."""

    framework_version = "v4"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v4_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=BehavioralBeliefUsablePlanner(),
            semantic_updater=ShadowSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV4FrozenBuyer(UniversalFrameworkV4Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV4WrongPolicyBuyer(UniversalFrameworkV4Buyer):
    belief_mode = "wrong_policy"


class UniversalFrameworkV4ShuffledFullBuyer(UniversalFrameworkV4Buyer):
    belief_mode = "shuffled_full"


class UniversalFrameworkV4OracleUtilityBuyer(UniversalFrameworkV4Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV5Buyer(UniversalFrameworkBuyer):
    """V5: censored utility belief, verified language, lookahead planning."""

    framework_version = "v5"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=LookaheadBeliefUsablePlanner(),
            structured_updater=CensoredBehaviorBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV5FrozenBuyer(UniversalFrameworkV5Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV5WrongPolicyBuyer(UniversalFrameworkV5Buyer):
    belief_mode = "wrong_policy"


class UniversalFrameworkV5ShuffledFullBuyer(UniversalFrameworkV5Buyer):
    belief_mode = "shuffled_full"


class UniversalFrameworkV5OracleUtilityBuyer(UniversalFrameworkV5Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV51Buyer(UniversalFrameworkV5Buyer):
    """V5.1 preserves V5 while enforcing monotone price concessions."""

    framework_version = "v5_1"


class UniversalFrameworkV51FrozenBuyer(UniversalFrameworkV51Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV51WrongPolicyBuyer(UniversalFrameworkV51Buyer):
    belief_mode = "wrong_policy"


class UniversalFrameworkV51ShuffledFullBuyer(UniversalFrameworkV51Buyer):
    belief_mode = "shuffled_full"


class UniversalFrameworkV51OracleUtilityBuyer(UniversalFrameworkV51Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV52Buyer(UniversalFrameworkV5Buyer):
    """V5.2: monotone frontier plus terminal-safe lookahead."""

    framework_version = "v5_2"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_2_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=TerminalSafeLookaheadPlanner(),
            structured_updater=CensoredBehaviorBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV52FrozenBuyer(UniversalFrameworkV52Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV52WrongPolicyBuyer(UniversalFrameworkV52Buyer):
    belief_mode = "wrong_policy_after_evidence"


class UniversalFrameworkV52ShuffledFullBuyer(UniversalFrameworkV52Buyer):
    belief_mode = "shuffled_full_after_evidence"


class UniversalFrameworkV52OracleUtilityBuyer(UniversalFrameworkV52Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV53Buyer(UniversalFrameworkV52Buyer):
    """V5.3: no dominated voluntary quit and uncertainty-aware acceptance."""

    framework_version = "v5_3"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_3_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=DominanceAndOptionPlanner(),
            structured_updater=CensoredBehaviorBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV53FrozenBuyer(UniversalFrameworkV53Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV53WrongPolicyBuyer(UniversalFrameworkV53Buyer):
    belief_mode = "wrong_policy_after_evidence"


class UniversalFrameworkV53ShuffledFullBuyer(UniversalFrameworkV53Buyer):
    belief_mode = "shuffled_full_after_evidence"


class UniversalFrameworkV53OracleUtilityBuyer(UniversalFrameworkV53Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV54Buyer(UniversalFrameworkV53Buyer):
    """V5.4: broad utility prior and strategically censored responses."""

    framework_version = "v5_4"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_4_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=DominanceAndOptionPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicCensoredBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV54FrozenBuyer(UniversalFrameworkV54Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV54WrongPolicyBuyer(UniversalFrameworkV54Buyer):
    belief_mode = "wrong_policy_after_evidence"


class UniversalFrameworkV54ShuffledFullBuyer(UniversalFrameworkV54Buyer):
    belief_mode = "shuffled_full_after_evidence"


class UniversalFrameworkV54OracleUtilityBuyer(UniversalFrameworkV54Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV55Buyer(UniversalFrameworkV54Buyer):
    """V5.5: keep V5.4 belief and consume the rejected policy frontier."""

    framework_version = "v5_5"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_5_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=BehavioralFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicCensoredBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV55FrozenBuyer(UniversalFrameworkV55Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV55NoFrontierBuyer(UniversalFrameworkV55Buyer):
    belief_mode = "no_frontier_after_evidence"


class UniversalFrameworkV55WrongPolicyBuyer(UniversalFrameworkV55Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV55ShuffledFullBuyer(UniversalFrameworkV55Buyer):
    belief_mode = "shuffled_full_frontier_after_evidence"


class UniversalFrameworkV55OracleUtilityBuyer(UniversalFrameworkV55Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV56Buyer(UniversalFrameworkV55Buyer):
    """V5.6: deadline-adaptive active probes beyond the policy frontier."""

    framework_version = "v5_6"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_6_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=ActiveFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicCensoredBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV56FrozenBuyer(UniversalFrameworkV56Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV56NoFrontierBuyer(UniversalFrameworkV56Buyer):
    belief_mode = "no_frontier_after_evidence"


class UniversalFrameworkV56WrongPolicyBuyer(UniversalFrameworkV56Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV56ShuffledFullBuyer(UniversalFrameworkV56Buyer):
    belief_mode = "shuffled_full_frontier_after_evidence"


class UniversalFrameworkV56OracleUtilityBuyer(UniversalFrameworkV56Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV57Buyer(UniversalFrameworkV55Buyer):
    """V5.7: soft latent utility x response-policy mixture posterior."""

    framework_version = "v5_7"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_7_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=BehavioralFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=LatentProfileMixtureBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV57FrozenBuyer(UniversalFrameworkV57Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV57NoFrontierBuyer(UniversalFrameworkV57Buyer):
    belief_mode = "no_frontier_after_evidence"


class UniversalFrameworkV57WrongPolicyBuyer(UniversalFrameworkV57Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV57ShuffledFullBuyer(UniversalFrameworkV57Buyer):
    belief_mode = "shuffled_full_frontier_after_evidence"


class UniversalFrameworkV57ShuffledMixtureBuyer(UniversalFrameworkV57Buyer):
    """Causal control that permutes the learned latent utility posterior."""

    belief_mode = "shuffled_latent_profile_after_evidence"


class UniversalFrameworkV57OracleUtilityBuyer(UniversalFrameworkV57Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV58Buyer(UniversalFrameworkV57Buyer):
    """V5.8: planner integrates feasibility over the full mixture posterior."""

    framework_version = "v5_8"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_8_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=PosteriorIntegratedFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=LatentProfileMixtureBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV58FrozenBuyer(UniversalFrameworkV58Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV58NoFrontierBuyer(UniversalFrameworkV58Buyer):
    belief_mode = "no_frontier_after_evidence"


class UniversalFrameworkV58WrongPolicyBuyer(UniversalFrameworkV58Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV58ShuffledFullBuyer(UniversalFrameworkV58Buyer):
    belief_mode = "shuffled_full_frontier_after_evidence"


class UniversalFrameworkV58OracleUtilityBuyer(UniversalFrameworkV58Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV59Buyer(UniversalFrameworkV57Buyer):
    """V5.9: factor reservation from strategic acceptance readiness."""

    framework_version = "v5_9"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v5_9_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=BehavioralFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicReadinessMixtureBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV59FrozenBuyer(UniversalFrameworkV59Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV59WrongPolicyBuyer(UniversalFrameworkV59Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV59ShuffledMixtureBuyer(UniversalFrameworkV59Buyer):
    belief_mode = "shuffled_latent_profile_after_evidence"


class UniversalFrameworkV59OracleUtilityBuyer(UniversalFrameworkV59Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV60Buyer(UniversalFrameworkV57Buyer):
    """V6.0: factor durable reservation from time-varying aspiration."""

    framework_version = "v6_0"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v6_0_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=AspirationAwareFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=ReservationAspirationMixtureBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV60FrozenBuyer(UniversalFrameworkV60Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV60NoAspirationPlannerBuyer(UniversalFrameworkV60Buyer):
    """Ablation: learn the same posterior but hide aspiration from planner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = "universal_framework_v6_0_no_aspiration_planner"
        self.engine.planner = BehavioralFrontierPlanner()


class UniversalFrameworkV60WrongPolicyBuyer(UniversalFrameworkV60Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV60ShuffledMixtureBuyer(UniversalFrameworkV60Buyer):
    belief_mode = "shuffled_latent_profile_after_evidence"


class UniversalFrameworkV60OracleUtilityBuyer(UniversalFrameworkV60Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV61Buyer(UniversalFrameworkV57Buyer):
    """V6.1: censored utility evidence plus aspiration-targeted probes."""

    framework_version = "v6_1"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k, continuous_planner_params_json
        self.variant = f"universal_framework_v6_1_{self.belief_mode}"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=AspirationProbeFrontierPlanner(),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=CensoredReservationAspirationBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV61FrozenBuyer(UniversalFrameworkV61Buyer):
    belief_mode = "frozen"


class UniversalFrameworkV61NoProbeBuyer(UniversalFrameworkV61Buyer):
    """Ablation: same censored belief with the frozen V5.5 planner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = "universal_framework_v6_1_no_probe"
        self.engine.planner = BehavioralFrontierPlanner()


class UniversalFrameworkV61WrongPolicyBuyer(UniversalFrameworkV61Buyer):
    belief_mode = "wrong_policy_frontier_after_evidence"


class UniversalFrameworkV61ShuffledMixtureBuyer(UniversalFrameworkV61Buyer):
    belief_mode = "shuffled_latent_profile_after_evidence"


class UniversalFrameworkV61OracleUtilityBuyer(UniversalFrameworkV61Buyer):
    belief_mode = "oracle_utility"


class UniversalFrameworkV62FinetunedPlannerBuyer(UniversalFrameworkV59Buyer):
    """V6.2: calibrated belief plus a listwise fine-tuned candidate-Q planner."""

    framework_version = "v6_2"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k
        if not continuous_planner_params_json:
            raise ValueError(
                "universal_framework_v6_2_finetuned_planner requires "
                "--continuous-planner-params-json planner_v2_checkpoint.json"
            )
        checkpoint = Path(continuous_planner_params_json)
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        calibration = payload.get("belief_calibration_json")
        updater = (
            StrategicReadinessMixtureBeliefUpdater.from_calibration_json(calibration)
            if calibration else StrategicReadinessMixtureBeliefUpdater()
        )
        self.variant = "universal_framework_v6_2_finetuned_planner"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=TrainableDecisionBoundaryPlanner(
                checkpoint,
                require_direct_evidence_for_override=True,
                minimum_override_advantage=0.0,
                minimum_continue_over_accept_advantage=0.02,
            ),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=updater,
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )


class UniversalFrameworkV62UnguardedPlannerBuyer(UniversalFrameworkV62FinetunedPlannerBuyer):
    """Ablation: fine-tuned Q network without the residual-policy safety head."""

    def __init__(self, *args, **kwargs):
        checkpoint = kwargs.get("continuous_planner_params_json")
        if checkpoint is None and len(args) >= 7:
            checkpoint = args[6]
        super().__init__(*args, **kwargs)
        self.variant = "universal_framework_v6_2_finetuned_planner_unguarded"
        self.engine.planner = TrainableDecisionBoundaryPlanner(checkpoint)


class UniversalFrameworkV63ConservativeAWRBuyer(UniversalFrameworkV59Buyer):
    """V6.3: cross-environment conservative AWR residual policy.

    The checkpoint may override V5.9's BehavioralFrontierPlanner only when the
    ensemble lower-confidence advantage passes the frozen safety gate.  Low
    confidence and out-of-distribution states therefore retain the base action.
    """

    framework_version = "v6_3_cross_environment_awr"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        del max_tokens, temperature, top_p, candidate_offer_k
        if not continuous_planner_params_json:
            raise ValueError(
                "universal_framework_v6_3_conservative_awr requires "
                "--continuous-planner-params-json cross_environment_awr_checkpoint.json"
            )
        checkpoint = Path(continuous_planner_params_json)
        self.variant = "universal_framework_v6_3_conservative_awr"
        semantic_client = planner_client or client
        self.engine = UniversalNegotiationEngine(
            planner=ConservativeAWRPlanner(checkpoint),
            belief_store=BroadPriorBeliefStore(),
            structured_updater=StrategicReadinessMixtureBeliefUpdater(),
            semantic_updater=VerifiedSemanticBeliefUpdater(semantic_client),
            language_client=client,
            belief_mode=self.belief_mode,
        )
        self._repeated_episode_index = 0

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> Tuple[ParsedAction, Dict[str, Any]]:
        if round_id == 1 and not history:
            self._repeated_episode_index += 1
        # ``item_id`` remains the opponent-memory key in the shared engine;
        # this namespace only distinguishes otherwise identical observations
        # from different episodes against that repeated opponent.
        adapter = SimpleEnvAdapter(
            scenario,
            history,
            self.framework_version,
            observation_namespace=str(self._repeated_episode_index),
            repeated_opponent=self._repeated_episode_index > 1,
            cross_session_progress=min(1.0, max(0, self._repeated_episode_index - 1) / 19.0),
        )
        state = adapter.state(round_id, max_turns, self.belief_mode)
        decision = self.engine.decide(state, adapter)
        return decision.rendered_action, decision.trace()


class UniversalFrameworkV64RewardLabeledLCBBuyer(UniversalFrameworkV63ConservativeAWRBuyer):
    """V6.4 deployment of the cross-environment reward-labelled checkpoint.

    No Simple-specific policy rule is added here.  The named class prevents
    results from this real-trajectory checkpoint being mixed with the older
    synthetic-only V6.3 checkpoint.  Simple keeps its frozen behavioral base;
    only the conservative residual checkpoint changes.
    """

    framework_version = "v6_4_reward_labeled_lcb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = "universal_framework_v6_4_reward_labeled_lcb"


class UniversalFrameworkV65EvidenceGatedLCBBuyer(UniversalFrameworkV64RewardLabeledLCBBuyer):
    """V6.5: V6.4 with a universal direct-evidence activation gate."""

    framework_version = "v6_5_evidence_gated_lcb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = "universal_framework_v6_5_evidence_gated_lcb"
        self.engine.planner.require_direct_evidence = True
