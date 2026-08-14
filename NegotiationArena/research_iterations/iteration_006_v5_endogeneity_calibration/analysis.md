# Iteration 006 result: V5 endogeneity calibration

## Two independent resource pilots

| Seed | Setting | Agreement | Mean focal reward | Early 1–5 | Late 6–10 |
|---:|---|---:|---:|---:|---:|
| 20260805 | resource-first | 100% | 21.05 | 16.00 | 26.10 |
| 20260805 | resource-second | 90% | 19.00 | 16.00 | 22.00 |
| 20260806 | resource-first | 100% | 18.15 | 20.00 | 16.30 |
| 20260806 | resource-second | 90% | 21.75 | 16.50 | 27.00 |

All 40 episodes were valid and had zero runtime errors. However, V5 does not show
a stable payoff advantage over V4 and loses agreement in both resource-second
runs. The provenance-aware idea remains scientifically motivated, but these fixed
tempering strengths are too conservative for promotion.

Decision: preserve V5 as a negative-result calibration ablation. Do not tune its
strengths further on these pilot seeds. V4 is the candidate method for formal
multi-run evaluation; V3.3 is the no-cross-episode-VOI ablation and V5 tests whether
naive provenance tempering helps.

Raw outputs:

- `arena_runs/dc_bap_v5_resource_serial_n10_seed20260805`
- `arena_runs/dc_bap_v5_resource_serial_n10_seed20260806`
