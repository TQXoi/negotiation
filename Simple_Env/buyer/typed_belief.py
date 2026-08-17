from .base import RLVRBuyer


class TypedBeliefBuyer(RLVRBuyer):
    """Typed evidence-grounded belief buyer.

    Internally uses the paper-aligned runner's `typed_evidence_belief`
    implementation: structured seller reservation range, confidence,
    accept-probability curve, and evidence fields.
    """

    variant_name = "typed_evidence_belief"
