# Iteration 004 result: V3.3 action-space-adaptive frontier

## Paper-temperature pilot (`temperature=0.7`)

Output: `arena_runs/dc_bap_v3_3_strict_acceptance_n10_seed20260805`

| Setting | Agreement | Mean focal reward | Mean joint reward |
|---|---:|---:|---:|
| buyer | 100% | 7.20 | 20.00 |
| seller | 90% | 17.80 | 18.00 |
| resource-first | 90% | 13.65 | not used for selection |
| resource-second | 100% | 22.25 | not used for selection |

V3.3 recovers the resource-second direction but does not solve scalar bargaining.
The buyer/seller traces show immediate acceptance of weak but positive opening
offers. The planner computes only the current episode's counterfactual value even
though its posterior persists across ten episodes.

## Reproducibility audit

Two serialized `temperature=0` runs with identical prompts and request hashes still
produced different Qwen responses from the current vLLM service. Therefore request
seed and temperature do not provide bitwise repeatability in this serving setup.
Formal claims must use multiple independent runs and confidence intervals; small
single-seed differences are diagnostic only. Full raw calls remain archived under:

- `arena_runs/dc_bap_v3_3_determinism_a_n3_seed20260806`
- `arena_runs/dc_bap_v3_3_determinism_b_n3_seed20260806`
