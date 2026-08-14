# Iteration 001 — V3 uncertainty safeguard

## Hypothesis

Framework V2 的 posterior entropy 只被记录，没有改变候选排序。将 action-conditional acceptance forecast 向 posterior adverse tail 收缩，并加入 response-support eligibility、outside-option dominance 和 agreement-regret gate，应当：

1. 阻止 Seller 在高不确定下从 43 反锚到 70；
2. 降低 Buyer/Seller 因小幅潜在增益而失约的概率；
3. 阻止 Resource planner 发送 safeguarded value 为负的 proposal；
4. 不增加 belief 或 generator 的 LLM 调用次数。

## Single changed mechanism

保持 V2 belief updater、candidate space、reciprocity ledger、language realizer 不变，只改变 belief uncertainty 在 planner 中的使用方式。

## Canary pass criteria

- 四个 setting 各 1 episode，0 error、100% format success；
- trace 包含 `accept_mean`、`accept_tail_q10`、`accept_safeguarded`、outside option 和 gate reason；
- 不选择 `safeguarded_contingent_value <= 0` 的 proposal。

## Pilot promotion criteria

- 相同 seed 下运行 3 runs × 10 episodes；
- Buyer/Seller agreement 不低于 V2 的 90%；
- 四项无超过 1 utility 的系统性退化，至少一个 V2 failure setting 改善；
- focal calls 不超过 V2 1.2 倍。
