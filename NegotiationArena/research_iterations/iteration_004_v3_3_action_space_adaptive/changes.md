# Code changes

- Added `ActionSpaceAdaptiveCommitmentBeliefPlannerAgent` (`framework_v3_3`).
- Refactored V3.2's frontier decision behind `_allow_ungated_exploit()`.
- Scalar buy/sell negotiations keep all candidates response-supported.
- Combinatorial resource negotiations retain one broad exploit candidate at all
  proposal decisions; safe/probe/reciprocal/fallback candidates remain gated.
- Kept uncertainty safeguards, commitment floor, locked action realization, and
  strict immediate-offer acceptance unchanged.
- Added continuous and frozen CLI variants and a structural unit test.
