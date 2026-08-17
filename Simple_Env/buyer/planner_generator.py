from .base import RLVRBuyer


class PlannerGeneratorBuyer(RLVRBuyer):
    """Prompt planner + final generator baseline with no separate belief model."""

    variant_name = "planner_generator"
