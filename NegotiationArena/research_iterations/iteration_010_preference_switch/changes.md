# Iteration 010 changes and audit trail

- Added evaluator-only dynamic opponent truth; never enters normal prompts or learned posterior.
- Added a moderate private resource-valuation switch at episode 11.
- Added explicit `opponent_preference_switched` episode labels and pre/post summaries.
- Completed smoke (4 episodes, 0 errors) and formal matrix (6 cells, 600 episodes, 0 errors).
- Result: continuous V8 did not beat the historical frozen variant. Audit found the
  switch moved truth toward the 0.5 prior and historical frozen still carried one
  observation per episode into meta memory. Result retained as a negative pilot.
