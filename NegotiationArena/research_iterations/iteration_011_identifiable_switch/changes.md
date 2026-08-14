# Iteration 011 changes and audit trail

- Added `framework_v8_no_cross_episode`, preserving within-episode inference and
  planner while resetting the joint belief to uniform every episode.
- Added away-from-prior switch profile and matched stationary controls.
- First smoke attempt failed before model calls due missing method metadata.
- Second smoke attempt revealed a silent factory fallback after one episode;
  output is retained under `runs/smoke`, and no formal cell used it.
- Fixed factory routing, reran 54 tests, then completed corrected smoke under
  `runs/smoke_v2` (4 episodes, 0 errors; diagnostics present; truth .833 -> .95).
- Completed preregistered formal matrix: 8 cells, 800 episodes, 0 errors.
- Result: Resource-second V8 has a positive within-method reward DiD, but the
  continuous-minus-no-cross interval crosses zero and belief MAE worsens. This
  does not establish correct non-stationary belief adaptation.
