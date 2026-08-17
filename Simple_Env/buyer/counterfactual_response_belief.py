from .base import RLVRBuyer


class CounterfactualResponseBeliefBuyer(RLVRBuyer):
    """Counterfactual response belief + risk-aware planner buyer.

    The buyer enumerates k candidate offers, asks a response model to predict
    seller accept/counter/walk-away behavior for each, and lets the planner
    pick the offer with the best expected buyer surplus.
    """

    variant_name = "counterfactual_offer_model"
