# V8 smoke decision

状态：Stage 2 core smoke 完成，12/12 cells、480/480 episodes、0 errors。

| Setting | V5 reward | V7 reward | V8 reward | V7 agreement/joint | V8 agreement/joint | V8-vs-V7 paired runs | Safe-vs-ungated flip |
|---|---:|---:|---:|---:|---:|---|---:|
| Buyer | 6.350 | 4.725 | **11.450** | 1.00/20.00 | .95/19.00 | +3.000, +10.450 | .159 |
| Seller | 10.725 | 10.625 | **12.650** | 1.00/20.00 | 1.00/20.00 | +9.800, −5.750 | .050 |
| Resource first | 13.663 | **17.350** | 16.563 | .975/35.30 | **1.00/36.65** | +1.425, −3.000 | .033 |
| Resource second | **24.588** | 20.238 | 18.663 | .975/50.75 | .95/42.65 | −.975, −2.175 | .088 |

Gate interpretation:

- Buyer improvement: pass, both paired runs positive;
- at least two settings not lower in aggregate: Buyer and Seller pass, but Seller is unstable;
- resource-first no collapse: pass; agreement and joint improve;
- at least two settings have reliability-gated action flips: pass;
- infrastructure errors: pass;
- resource-second: clear failure condition; must remain in all later reports.

Decision: run the preregistered resource-role wrong/shuffled safety stress before any 5×20 confirmatory. Do not tune the gate from these two-run rewards.
