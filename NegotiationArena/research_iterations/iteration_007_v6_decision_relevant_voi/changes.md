# Iteration 007 change record — V6 Decision-Relevant VOI

## Trigger

The frozen formal gate passed at `2026-08-13T02:05:40Z`:

- main: 36/36 cells, 3600/3600 latest episodes, 0 current errors;
- opponent-switch: 20/20 cells, 2000/2000 latest episodes, 0 current errors.

All preregistered diagnostic gates A–D fired. V5 improved acceptance calibration
in all four settings without consistent reward improvement; continuous was worse
than frozen in buyer and resource-first; wrong/shuffled beliefs were not reliably
worse; oracle belief retained positive headroom. The switch matrix did not support
a unique continuous-adaptation effect.

## Additive code changes

Existing V3.3, V4 and V5 classes and their result directories were not replaced.

1. `arena_integration/decision_calibrated_agent.py`
   - added `DecisionRelevantInformationPlannerAgent` (`framework_v6`);
   - retained V5 endogeneity-tempered continuous belief;
   - replaced V4's entropy × sqrt(horizon) probe bonus with deterministic
     ACCEPT/COUNTER/REJECT shadow posterior updates;
   - credited information only when response-identifiability, action-flip
     probability and positive expected future decision value all hold;
   - capped the bonus by both current-deal opportunity budget and posterior
     decision-value range;
   - added complete VOI components to `framework_v6_decisions.jsonl`.
2. `arena_integration/run_opponent_simulation_setting.py`
   - registered continuous/frozen/wrong/shuffled/oracle V6 method variants;
   - added method semantics to each saved config.
3. `tests/test_repeated_integration_unittest.py`
   - added tests for ordered shadow posterior updates;
   - covered ACCEPT, COUNTER and REJECT outcome branches;
   - verified opportunity-cost capping blocks a V4-style low-value probe.
4. `offline_replay.py`
   - reconstructs exact V5 particle posteriors from recorded structured events
     and saved semantic-extractor outputs without an LLM call;
   - validates reconstruction against every saved posterior summary;
   - performs one-step V6 action replay on the fixed V5 trajectory.
5. `run_smoke.sh`
   - isolated new-seed 4 settings × V4/V5/V6 × 2 runs × 20 episodes smoke;
   - append-only/resumable outputs under this iteration directory.

## Verification before online execution

```text
python -m compileall -q arena_integration tests
python -m unittest tests.test_repeated_integration_unittest -v
Ran 42 tests in 0.330s — OK
```

Offline replay reconstructed all 553 decision posteriors with zero summary error.
Across the logged V5 trajectories, the old cross-episode heuristic triggered 21
probes. V6 blocked 20, restored 20 positive observed offers, retained one
decision-relevant probe, and introduced zero infeasible proposals. This is an
off-policy mechanism diagnostic, not an online reward claim.

## Online smoke

Started against the existing Qwen endpoint on `127.0.0.1:8002` with seed
`20260840`. Results are written to `runs/smoke_seed20260840`; the formal
`arena_runs/formal_belief_matrix_qwen30b_5x20_20260812` directory remains
read-only.
