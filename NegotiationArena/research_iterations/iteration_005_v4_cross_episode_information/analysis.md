# Iteration 005 result: V4 cross-episode information planner

## First serialized pilot (seed 20260805)

| Setting | Continuous reward / agreement | Frozen reward / agreement |
|---|---:|---:|
| buyer | 11.80 / 100% | 13.40 / 100% |
| seller | 19.10 / 100% | 19.20 / 100% |
| resource-first | 25.75 / 90% | 16.00 / 90% |
| resource-second | 28.25 / 100% | 24.00 / 100% |

The price settings usually ended before a second belief update and show no continuous
advantage. Resource settings provide an initial positive signal.

## Second resource seed (20260806)

| Setting | Continuous reward / agreement | Frozen reward / agreement |
|---|---:|---:|
| resource-first | 23.50 / 100% | 20.00 / 100% |
| resource-second | 19.30 / 100% | 28.55 / 100% |

The continuous advantage does not replicate in resource-second. Inspection suggests
that repeated strategic counteroffers are treated as independent preference evidence,
which can make the continuous posterior overconfident. Thus V4 remains an important
planner ablation, but its updater is not yet adequately calibrated.

Raw outputs:

- `arena_runs/dc_bap_v4_strict_serial_n10_seed20260805`
- `arena_runs/dc_bap_v4_frozen_strict_serial_n10_seed20260805`
- `arena_runs/dc_bap_v4_resource_seed20260806`
- `arena_runs/dc_bap_v4_frozen_resource_seed20260806`
