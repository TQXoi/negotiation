# 论文主张与证据台账

**规则：** 每个要写入摘要或主结果的 claim 必须有对应 frozen method、manifest、raw run、配对统计和负结果检查。新方法观察过当前 test trajectory 后，应建立新的独立 test split；不可继续把相同 split 当 confirmatory evidence。

| Claim | 当前证据 | 状态 | 还需什么 |
|---|---|---|---|
| Simple V6.3 高于 CoT/full | `docs/simple/PRIMARY_METRICS.md`；128 × 3；paired cluster bootstrap | **可作为本地确认结果** | 外部 held-out seller/backbone |
| V6.3 AWR 比 V5.9 更强 | +0.0149，CI [-0.0296, 0.0589] | **未证实** | 新 split、更多 rollouts 或作为小幅 point gain |
| CaSiNo chooser 提高 P1 | 100 official test，两个运行中 +约 3.6–4.8 | **有希望，非 paired confirmatory** | 3+ rollouts、P1/joint/agreement CI |
| AgenticPay V60 only_multi_seller 高于 CoT/full | 29 × 3 seeds；BuyerScore 30.7184；CI 不跨 0 | **子集已证实** | Direct CI 跨 0，扩到 all-116/all-231 |
| AgenticPay V60 all-231 通胜 | 旧运行含 25 adapter errors | **不可主张** | 修复后的干净 paired 231 × 3 |
| NegotiationArena 超过 Opponent Simulation | buyer/resource-second 局部胜，seller/resource-first 负 | **不能写成全设置胜** | 冻结一个 variant 四设置报告 |
| 连续 belief 产生收益 | ANL/NA 上 frozen/shuffled 不弱，CaSiNo top-1 低 | **未证实** | action-flip → reward 因果链与 oracle/wrong controls |

以后每轮增加一行：`commit SHA`、任务 split、seller/model snapshot、平均差、95% CI、runner error、是否主表/附录、与原 hypothesis 是否一致。
