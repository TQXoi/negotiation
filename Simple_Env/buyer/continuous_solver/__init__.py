"""Continual-resolving buyer components for Simple_Env.

This package is intentionally separated from the current prompt-only buyer
variants. It provides a clean exploration surface for persistent opponent
belief, belief updates, EV-based planning, and final message generation.
"""

from .pipeline import ContinualResolvingPipeline

__all__ = ["ContinualResolvingPipeline"]
