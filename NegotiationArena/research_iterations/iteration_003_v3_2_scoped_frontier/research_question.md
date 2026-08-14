# Iteration 003 — V3.2 scoped exploit frontier

## Hypothesis

V3.1 Buyer 的 3 次失约来自 `exploit` 在所有环境绕过 response-support gate，导致从 58 直接 counter 到 30。只允许 Resource/First 的首个 opening exploit 使用全 frontier，其他位置全部使用 response-supported frontier，应保留 Resource opening 收益并降低 Buyer commitment 的失败成本。

## Single changed mechanism

只改变 ungated exploit frontier 的作用域。Posterior、uncertainty safeguard、outside option、agreement regret 与 commitment floor 全部不变，不增加 LLM 调用。

## Promotion criteria

- Buyer reward > V3.1 且 agreement ≥ 90%；
- Seller 保持接近 20/100%；
- strict evaluator 下 Resource/Second 保持高收益；
- Resource/First 不出现负效用假 agreement；
- 0 error、100% format。
