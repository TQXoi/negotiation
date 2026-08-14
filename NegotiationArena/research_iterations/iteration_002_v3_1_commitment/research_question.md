# Iteration 002 — V3.1 reciprocal commitment

## Hypothesis

V3.0 的局部 agreement safeguard 会让 repeated opponent 逐局恶化 opening terms；同时，全局 response-support gate 会把 Resource first-mover 的 exploit frontier 错误替换成低效用 safe frontier。

加入最近五局已实现自身收益的 soft commitment floor，并将 opening exploit 与 safe frontier 分离，应在保持 V3 agreement 改善的同时恢复 Buyer 与 Resource/First 自身收益。

## Single changed mechanism

- V3 belief updater、adverse-tail forecast、outside option、agreement regret 均不变；
- 新增基于 own realized reward 的 commitment floor；
- candidate family selection 中 exploit 使用全 frontier，safe/probe/reciprocal/fallback 仍使用 response-support gate；
- 不新增 LLM 调用。

## Promotion criteria

- 0 error、100% format；
- Buyer reward 明显高于 V3.0，agreement 不低于 90%；
- Resource/First reward 恢复到 V2 附近；
- Seller 与 Resource/Second 不比 V3.0 下降超过 1；
- calls 不超过 V2 的 1.2 倍。
