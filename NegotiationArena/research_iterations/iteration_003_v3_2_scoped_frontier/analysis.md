# Iteration 003 result: V3.2 scoped frontier

## Strict paired pilot

- Model: `Qwen3-30B-A3B-Instruct-2507-base`
- Seed: `20260805`
- Episodes: 10 per setting
- Evaluator: an `ACCEPT` is valid only for the immediately preceding opponent trade
- Output: `arena_runs/dc_bap_v3_2_strict_acceptance_n10_seed20260805`

| Setting | Agreement | Mean focal reward | Mean joint reward | Mean turns |
|---|---:|---:|---:|---:|
| buyer | 100% | 12.70 | 20.00 | 2.70 |
| seller | 100% | 15.70 | 20.00 | 2.60 |
| resource-first | 100% | 14.25 | 34.20 | 2.40 |
| resource-second | 90% | 11.10 | 38.40 | 3.60 |

## Interpretation

V3.2 fixed V3.1's monetary-bargaining failure: it no longer lets an unsupported
high-self-utility price bypass the response posterior. It also restores the
resource-first result under the strict evaluator. However, restricting the broad
resource frontier to the first mover's first decision is too narrow for the second
mover. In episode 10 the planner reached `WAIT` while the opponent later attempted
to accept a stale historical counter; the evaluator correctly assigned no deal.

This is a useful negative result. Candidate breadth should not be selected by a
role-specific opening exception. It should follow action-space structure: scalar
price bargaining can safely gate every candidate, whereas combinatorial exchange
needs a broad bundle candidate throughout the dialogue.
