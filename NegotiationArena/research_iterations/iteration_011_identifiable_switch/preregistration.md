# Iteration 011: identifiable away-from-prior preference switch

The toward-prior pilot (iteration 010) is retained as a negative result but is
not treated as an identifiable calibration test. This final test fixes both
confounds before any new outcome is observed.

- Settings: resource_first and resource_second.
- Methods: framework_v8 and framework_v8_no_cross_episode.
- Conditions: stationary no-switch and switch at episode 11.
- Switch moves opponent theta away from the 0.5 prior while retaining ordering:
  BLUE 5/6 -> .95; RED 1/6 -> .05. Valuation sums remain 3.0.
- Five runs x 20 episodes, seed 20261300.
- The no-cross control resets to the identical uniform prior each episode but
  retains within-episode updates and the identical executable planner.
- Primary: paired reward DiD of V8 versus its own stationary condition, compared
  against no-cross DiD; report the difference between these two DiDs.
- Mechanism: estimate should move away from .5 in the correct direction; compare
  early episode 11-15 versus late 16-20 MAE and action changes.
- No tuning after outcome inspection.
