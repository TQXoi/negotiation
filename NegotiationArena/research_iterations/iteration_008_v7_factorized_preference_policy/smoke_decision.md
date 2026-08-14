# V7 smoke and intervention decision

## Core smoke completion

- 12/12 cells, 480/480 latest episodes, zero errors;
- seed `20260860`, 2 runs × 20 episodes;
- V5, V6 and V7 share model, opponent, prompts, temperature and protocol.

| Setting | V5 reward | V6 reward | V7 reward | V7−V5 paired runs | V7 agreement/joint |
|---|---:|---:|---:|---|---:|
| Buyer | 8.025 | 7.950 | **10.600** | +2.90, +2.25 | 1.00 / 20.00 |
| Seller | **18.450** | 17.050 | 16.475 | −3.00, −0.95 | 1.00 / 20.00 |
| Resource first | 18.100 | 16.000 | **20.150** | +0.175, +3.925 | .975 / 39.70 |
| Resource second | 17.463 | **29.513** | 24.463 | +13.40, +0.60 | 1.00 / 56.35 |

V7 is positive versus V5 in both Buyer runs and both resource-first/resource-second
runs, but negative in both Seller runs. It does not systematically lower
agreement or joint utility. The two-run values remain smoke evidence only.

## Online mechanism

- Buyer: preference first-action flips .25; oracle flips .80; policy flips 0;
- Seller: preference/oracle flips .15; policy flips 0;
- Resource first: wrong-preference 1.0, shuffled .275, oracle 1.0,
  policy-uniform .125, policy-shuffled .175;
- Resource second: wrong-preference .40, shuffled .05, oracle .50; policy flips 0.

Thus V7's preference marginal changes exploitation in all settings. Response-policy
belief changes action only in resource-first and remains high-entropy elsewhere.

## Deployed resource-first interventions

All cells use the same seed as continuous V7 and contain 2×20 episodes.

| Variant | Reward | Agreement | Joint | Correct−variant paired runs |
|---|---:|---:|---:|---|
| correct V7 | 20.150 | .975 | 39.70 | — |
| wrong preference | 11.238 | 1.00 | 11.45 | +6.45, +11.375 |
| shuffled preference | 18.550 | 1.00 | 38.90 | −0.90, +4.10 |
| oracle preference | 19.813 | .975 | 27.25 | +1.125, −0.45 |
| uniform policy | 21.538 | 1.00 | 41.35 | −4.125, +1.35 |
| shuffled policy | 17.013 | 1.00 | 36.95 | +0.075, +6.20 |

Wrong preference causes a large, directionally consistent collapse in focal and
joint utility. Shuffled preference has the expected average direction but is not
consistent across two runs. Shuffled response policy hurts in both runs; uniform
policy does not, showing that learned policy inference has not yet surpassed an
uninformative policy prior. Oracle preference does not improve reward and reduces
joint utility, so truth access alone does not solve response-policy/planner error.

## Gate decision

Proceed to 5×20 confirmatory because:

1. belief-conditioned exploitation flips online in more than two settings;
2. V7 is non-inferior to V5 in three of four smoke settings and positive in both
   runs for Buyer and both resource roles;
3. deployed wrong preference and shuffled policy have directionally harmful
   utility effects in resource-first;
4. agreement/joint utility are not systematically sacrificed.

Do not yet run opponent-switch or claim learned response-policy adaptation.
Confirmatory must include uniform-policy and shuffled-policy controls and retain
the negative Seller result.
