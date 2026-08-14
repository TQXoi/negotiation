# Iteration 009 change log

## Stage 0 plan (before code)

- add `ReliabilityGatedFactorizedBeliefPlannerAgent` as a V7 subclass;
- preserve V7 and all prior method names/results unchanged;
- expose `framework_v8` plus frozen/wrong/shuffled/oracle diagnostics;
- log channel counts, posterior drift and preference/policy trust;
- add integration tests and an offline shadow replay;
- run online smoke only after tests and replay gates pass.

## Stage 0 implementation

- added a role-agnostic anchor/learned posterior mixture;
- preference trust uses direct responses plus weak offer/semantic evidence;
- policy trust uses direct responses only;
- unsupported posterior drift receives a smooth reliability discount;
- oracle preference remains evaluator-only and bypasses only the theta gate;
- wrong/shuffled variants are mediated, so they test safety mitigation;
- added `ungated` and `anchor` same-state action shadows;
- extended summaries with trust and new action-flip metrics;
- added four unit tests; complete integration suite: 51/51 passing.

## Offline replay

Fixed V7 confirmatory trajectories, zero model calls, 541 decisions:

| Setting | Decisions | V8 vs V7 flip | V8 vs anchor flip | Mean theta trust | Mean policy trust |
|---|---:|---:|---:|---:|---:|
| Buyer | 110 | .100 | .027 | .189 | .046 |
| Seller | 102 | .118 | .049 | .150 | .010 |
| Resource first | 182 | .044 | .027 | .339 | .289 |
| Resource second | 147 | .014 | .027 | .337 | .201 |

The gate is more willing to trust evidence in multi-turn resource negotiations,
yet does not collapse every resource decision to the V7 action. Buyer has the
required non-zero protective flips. Replay does not claim counterfactual reward.

## Online smoke (seed 20261020)

12/12 cells, 480/480 episodes, 0 errors. See `smoke_decision.md`.

- Buyer: V8 11.450 vs V7 4.725; both paired runs positive; agreement .95;
- Seller: V8 12.650 vs V7 10.625; paired direction mixed; agreement 1.00;
- Resource first: V8 reward 16.563 vs V7 17.350, but agreement/joint improve
  from .975/35.30 to 1.00/36.65;
- Resource second: V8 18.663 vs V7 20.238 and lower joint; retained as failure.

## Resource safety stress (same smoke seed)

8/8 cells, 320/320 episodes, 0 errors.

| Setting | Contrast | V7 reward/agreement/joint | V8 reward/agreement/joint |
|---|---|---:|---:|
| Resource first | wrong-confident | 10.425/1.00/10.80 | **19.025/.975/37.90** |
| Resource first | shuffled | 18.775/.925/36.80 | **19.388/1.00/39.15** |
| Resource second | wrong-confident | 11.213/.95/25.35 | **20.000/.95/37.90** |
| Resource second | shuffled | 20.988/.975/40.55 | **28.775/1.00/49.90** |

The wrong-belief mitigation is large in both resource roles. It is not explained
by indiscriminate rejection: agreement is unchanged or lower by only one of 40
episodes while joint utility rises substantially.
