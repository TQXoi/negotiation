# Iteration 010: private-preference switch preregistration

- Freeze V8 code/results from iteration 009 before this test.
- Settings: resource_first, resource_second.
- Methods: framework_v7, framework_v8, framework_v8_frozen.
- Five runs x 20 episodes, seed 20261100, switch starts at episode 11.
- Control: exact method/setting/seed stationary cells from iteration 009.
- Focal utility is invariant. Only opponent private valuation changes moderately,
  retaining the original issue ordering and complementarity.
- Primary estimand: paired run-level difference-in-differences of post-minus-pre
  focal reward, switch minus no-switch.
- Mechanism checks: post-switch belief truth/estimate/MAE, action flips, agreement,
  joint reward, and early (11-15) versus late (16-20) recovery.
- No planner weight or trust-threshold tuning after observing these cells.
