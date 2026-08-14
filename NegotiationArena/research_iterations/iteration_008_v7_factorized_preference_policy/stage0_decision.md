# V7 Stage-0 decision

## Verification

- 47 integration tests pass after NumPy vectorization;
- vectorization changes only the numerical kernel, not particles or equations;
- exact source reconstruction was inherited from iteration 007: 553/553 V5
  posterior summaries, zero mismatch;
- V7 replay covered all 553 logged decisions and 399 episode-opening
  intervention points in 91 seconds, with zero model calls;
- no negative-self-utility proposal was introduced.

## Offline one-step action results

| Setting | Decisions | V7 vs V5 flips | V7 proposals | Frozen flip | Wrong-pref flip | Shuffled-pref flip | Oracle-pref flip | Uniform-policy flip | Shuffled-policy flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Buyer | 120 | 25 | 8 | .06 | .06 | .06 | 1.00 | .01 | .02 |
| Seller | 114 | 22 | 21 | .21 | .32 | .23 | .32 | .00 | .00 |
| Resource first | 186 | 20 | 142 | .00 | 1.00 | .23 | 1.00 | .06 | .06 |
| Resource second | 133 | 13 | 45 | .061 | .343 | .111 | .465 | .030 | .020 |

The planner is preference-sensitive in every role and especially in the
combinatorial action space. Policy interventions cause fewer flips because the
replayed policy marginal remains high-entropy; this is a testable prediction,
not evidence that the policy factor is already learned well.

Final replay preference estimates illustrate the remaining inference problem:

- Buyer seller-reservation estimate 42.71 (evaluator truth 43);
- Seller buyer-reservation estimate 57.43 (truth 63);
- Resource first opponent-X-weight estimate .357 (truth .833);
- Resource second opponent-X-weight estimate .607 (truth .167).

Therefore Stage 0 supports online smoke for mechanism testing, but does **not**
support a performance claim. In particular, resource preference inference may
be directionally wrong even though oracle preference changes the action. The
online smoke must retain all four roles and report this failure rather than
tuning on evaluator truth.

## Decision

Proceed to the preregistered new-seed core smoke (V5/V6/V7, 2×20, four
settings). Do not run 5×20 or opponent-switch until:

1. V7 produces nonzero belief-conditioned exploitation flips online;
2. reward/agreement/joint utility do not systematically degrade;
3. targeted deployed wrong/shuffled preference variants show the expected
   utility direction in at least one setting.
