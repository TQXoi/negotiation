"""Canonical schemas shared by all negotiation environments.

The schemas intentionally describe utilities, issues, counterparties, and
responses instead of benchmark tags such as ``[BUY]`` or ``<contract>``.
Those protocol details belong in environment adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


ActionType = Literal["offer", "accept", "reject", "quit", "ask"]
ResponseType = Literal["offer", "counter", "accept", "reject", "quit", "message"]


@dataclass
class IssueSpec:
    """One negotiable issue as seen by the focal agent.

    ``own_values`` stores exact focal-side utility contributions for discrete
    options.  Opponent weights are never placed here: they remain uncertain in
    :class:`OpponentBelief`.
    """

    name: str
    kind: Literal["price", "continuous", "discrete", "allocation"]
    domain: Any = None
    own_weight: float = 0.0
    own_values: Dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class CanonicalOffer:
    """Protocol-free offer representation."""

    price: Optional[float] = None
    continuous_terms: Dict[str, float] = field(default_factory=dict)
    discrete_terms: Dict[str, Any] = field(default_factory=dict)
    allocations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NegotiationObservation:
    """A public event converted by an environment adapter."""

    observation_id: str
    turn: int
    actor_id: str
    counterparty_id: str
    response_type: ResponseType
    offer: Optional[CanonicalOffer] = None
    text: str = ""
    response_to_offer: Optional[CanonicalOffer] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalState:
    """Decision state consumed by the universal framework."""

    session_id: str
    environment_id: str
    self_id: str
    role: str
    counterparty_id: str
    turn: int
    max_turns: int
    issues: List[IssueSpec]
    own_value_scale: float
    own_outside_option: float = 0.0
    reference_value: Optional[float] = None
    observations: List[NegotiationObservation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        return min(1.0, max(0.0, self.turn / max(1, self.max_turns)))


@dataclass
class ResponsePolicyBelief:
    """Factorized response-policy belief.

    Acceptance evidence is stored in coarse compatibility bins.  This makes
    the representation usable for price, multi-issue, and allocation offers.
    """

    accept_alpha: List[float] = field(default_factory=lambda: [1.0] * 5)
    accept_beta: List[float] = field(default_factory=lambda: [1.0] * 5)
    counter_rate: float = 0.5
    quit_rate: float = 0.1
    patience: float = 0.6
    reliability: float = 0.15
    # V2 diagnostics. V1 does not consume them, so its scoring is preserved.
    rejected_compatibility_max: float = 0.0
    accepted_compatibility_min: Optional[float] = None
    repeated_rejection_streak: int = 0
    last_rejected_compatibility: Optional[float] = None
    # V5 keeps observable asking policy separate from latent reservation
    # utility. An ask is an anchor, not a truthful declaration of cost.
    minimum_observed_ask: Optional[float] = None
    maximum_observed_ask: Optional[float] = None
    last_observed_ask: Optional[float] = None
    ask_concession_ema: float = 0.0
    strategic_rejection_probability: float = 0.55
    # V6 separates a durable reservation boundary from the opponent's current
    # strategic target.  Aspiration is a response-policy variable: it may sit
    # well above reservation and decay as the deadline approaches.  Keeping
    # these fields here prevents a strategic counter from being mislabeled as
    # evidence that mutually beneficial agreement is impossible.
    aspiration_ratio_low: float = 0.30
    aspiration_ratio_mean: float = 0.70
    aspiration_ratio_high: float = 1.00
    aspiration_margin_mean: float = 0.25
    aspiration_reliability: float = 0.10

    def acceptance_rate(self, compatibility: float) -> float:
        index = min(4, max(0, int(float(compatibility) * 5)))
        a = self.accept_alpha[index]
        b = self.accept_beta[index]
        return a / max(1e-9, a + b)


@dataclass
class PreferenceBelief:
    """Opponent preference belief separated from response policy."""

    reservation_ratio_low: float = 0.30
    reservation_ratio_mean: float = 0.62
    reservation_ratio_high: float = 0.95
    issue_option_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    issue_directions: Dict[str, float] = field(default_factory=dict)
    reliability: float = 0.10


@dataclass
class OpponentBelief:
    """Persistent, per-counterparty belief with regime-change diagnostics."""

    counterparty_id: str
    preference: PreferenceBelief = field(default_factory=PreferenceBelief)
    response_policy: ResponsePolicyBelief = field(default_factory=ResponsePolicyBelief)
    evidence_count: int = 0
    direct_response_count: int = 0
    semantic_evidence_count: int = 0
    regime_change_probability: float = 0.0
    old_regime_weight: float = 1.0
    new_regime_weight: float = 0.0
    # V5.7 generic latent utility x response-policy posterior. Keys describe
    # hypotheses in normalized opponent-value space; no environment units or
    # evaluator truth are stored here.
    latent_profile_weights: Dict[str, float] = field(default_factory=dict)
    latent_profile_reservation_ratios: Dict[str, float] = field(default_factory=dict)
    # Optional normalized policy parameter for mixture implementations that
    # explicitly factor reservation utility from bargaining aspiration.
    latent_profile_aspiration_margins: Dict[str, float] = field(default_factory=dict)
    latent_profile_entropy_bits: Optional[float] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAction:
    """A feasible action supplied by an adapter and scored by the core."""

    candidate_id: str
    action_type: ActionType
    counterparty_id: str
    offer: Optional[CanonicalOffer]
    own_utility: float
    own_utility_normalized: float
    opponent_value_proxy: float
    base_acceptance: float
    information_gain: float = 0.0
    feasibility_margin: float = 0.0
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredCandidate:
    candidate: CandidateAction
    p_accept: float
    p_quit: float
    exploitation_value: float
    exploration_value: float
    risk_penalty: float
    score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidate"] = self.candidate.to_dict()
        return data


@dataclass
class FrameworkDecision:
    state: CanonicalState
    belief: OpponentBelief
    ranked_candidates: List[ScoredCandidate]
    selected: ScoredCandidate
    rendered_action: Any
    validation: Dict[str, Any]
    semantic_update: Optional[Dict[str, Any]] = None

    def trace(self) -> Dict[str, Any]:
        return {
            "framework_version": self.state.metadata.get(
                "framework_version", "universal_belief_planner_v1"
            ),
            "environment_id": self.state.environment_id,
            "session_id": self.state.session_id,
            "counterparty_id": self.state.counterparty_id,
            "turn": self.state.turn,
            "belief": self.belief.to_dict(),
            "ranked_candidates": [item.to_dict() for item in self.ranked_candidates],
            "selected_candidate_id": self.selected.candidate.candidate_id,
            "validation": self.validation,
            "semantic_update": self.semantic_update,
        }
