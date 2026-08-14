# Iteration 007 smoke decision

## Completion

- 12/12 cells complete;
- 480/480 latest episodes;
- zero infrastructure errors;
- independent seed `20260840`;
- 2 runs × 20 episodes per cell;
- V4, V5 and V6 used identical opponent, prompt, model, temperature and protocol.

## Result

| Setting | V4 | V5 | V6 | V6−V4 | V6−V5 |
|---|---:|---:|---:|---:|---:|
| Buyer | 14.325 | 10.575 | 10.600 | −3.725 | +0.025 |
| Seller | 17.225 | 13.750 | 13.100 | −4.125 | −0.650 |
| Resource first | 15.038 | 16.888 | 18.325 | +3.288 | +1.438 |
| Resource second | 20.225 | 19.988 | 20.513 | +0.287 | +0.525 |

The two paired run deltas are heterogeneous and are not used as confirmatory
evidence. V6 improved resource-first in both smoke runs and resource-second over
V5 in both runs, but it did not improve Buyer/Seller.

## Mechanism gate

V6 evaluated 34 accept-versus-probe opportunities online:

- Buyer: 16 evaluated, 16 blocked, 0 executed;
- Seller: 15 evaluated, 15 blocked, 0 executed;
- Resource first: 3 evaluated, 3 blocked, 0 executed;
- Resource second: 0 evaluated, 0 executed.

Thus the implementation successfully removes generic entropy exploration, but
the proposed decision-relevant exploration mechanism is inactive in the actual
online policy. The resource gains are attributable to V5-like exploitation and
stochastic opponent trajectories, not to executed V6 probes.

## Decision

**Do not promote V6 to the 5×20 confirmatory matrix as the paper's main method.**

This is not a post-hoc threshold change. The preregistration requires a stable
positive reward effect and an active, auditable belief-to-action mechanism. V6
fails the latter and is negative versus V4 in both scalar roles. Its useful
contribution is retained as a safety layer and negative finding:

> Generic entropy/IG probing is too permissive, but merely capping it produces a
> conservative policy rather than a stronger belief-using negotiator.

The next iteration must improve belief-conditioned exploitation, not tune the
V6 cap until probes appear.
