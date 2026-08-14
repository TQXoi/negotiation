# V7 confirmatory recovery audit (2026-08-13 UTC)

## Observed state

- Original unified execution session was no longer available.
- No `run_confirmatory` or episode process remained active.
- Monitor state: 18/32 complete cells, 1800/3200 episodes, 0 current errors.
- There were no partially completed cells.
- Buyer and seller: all eight methods complete.
- Resource-first: `framework_v5` and `framework_v7` complete.
- Model service `127.0.0.1:8002` responded normally.

## Recovery decision

The recovery queue contains only the 14 never-started cells:

- resource-first: frozen, wrong-confident, shuffled, oracle, policy-uniform,
  policy-shuffled;
- resource-second: the same six variants plus V5 and V7.

`run_confirmatory_recovery_all.sh` executes this as a 12-cell phase followed by
a two-cell resource-second V5/V7 phase. It intentionally does not touch any
completed cell log.

The run command remains resumable and uses the preregistered seed, run count,
episode count, model endpoint, and protocol settings from `run_confirmatory.sh`.
All per-call traces, episode records, summaries, and code snapshots remain
append-only or versioned in the iteration directory.

## Important correction

`run_confirmatory_remaining.sh` covers 12 cells (six methods across two
settings). The wrapper then runs resource-second V5 and V7 as a two-cell queue.
This split ensures the completed resource-first V5/V7 stdout logs are never
overwritten.
