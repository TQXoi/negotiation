"""Identifiable opponent-belief components for Simple Env.

These components are deliberately independent from the language-model client so
their causal behavior can be unit tested before an expensive benchmark run.
"""

from .oracle import SellerBehaviorModel, categorical_js_divergence
from .adaptive import (
    BehaviorHypothesis,
    ChangePointBehaviorBelief,
    ChangePointUpdate,
    default_behavior_hypotheses,
    expected_information_gain_bits,
    normalize_public_outcome,
)
from .probe import ActiveProbePlanner, ProbeScore
from .profiles import UtilityProfile, UtilityProfilePair, build_profile_suite
from .regime import RegimeMixtureBelief, RegimeUpdate

__all__ = [
    "ActiveProbePlanner",
    "BehaviorHypothesis",
    "ChangePointBehaviorBelief",
    "ChangePointUpdate",
    "ProbeScore",
    "RegimeMixtureBelief",
    "RegimeUpdate",
    "SellerBehaviorModel",
    "UtilityProfile",
    "UtilityProfilePair",
    "build_profile_suite",
    "categorical_js_divergence",
    "default_behavior_hypotheses",
    "expected_information_gain_bits",
    "normalize_public_outcome",
]
