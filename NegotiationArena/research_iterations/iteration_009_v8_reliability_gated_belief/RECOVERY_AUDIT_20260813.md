# V8 confirmatory recovery audit — 2026-08-13 UTC

## Observed state

- The original unified execution session was interrupted.
- No host `run_confirmatory` or episode process remained active.
- Persisted matrix state: 4/28 complete cells, 400/2800 episodes, 0 errors.
- There were no partially completed cells.
- Complete Buyer cells: V5, V7, V8 continuous and V8 frozen, each 100/100.
- Their stdout logs, model-call traces, episode JSONL, configs and summaries are intact.

## Recovery scope

To avoid truncating completed stdout logs, recovery is split into two queues:

1. Buyer only: V8 wrong-confident, shuffled and oracle;
2. Seller, resource-first and resource-second: all seven preregistered methods.

The same model, endpoint, seed, temperature, protocol and 5×20 design are inherited
from `run_confirmatory.sh`. Since recovery begins at a clean cell boundary, no
within-run posterior reconstruction is required.
