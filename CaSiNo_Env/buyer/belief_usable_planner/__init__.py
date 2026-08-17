"""Belief-usable planner buyer for CaSiNo_Env.

This package separates four pieces:

1. mathematical belief update,
2. candidate action generation,
3. belief-conditioned candidate scoring,
4. final dialogue generation.

The current v0.1 is deterministic and training-free. The trace format is
designed so that the scorer and final chooser can later be trained with SFT,
RFT, or pairwise preference learning.
"""
