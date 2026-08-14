# Iteration 001 analysis

## Canary（1 run × 1 episode × 4 settings）

输出：`arena_runs/dc_bap_v3_canary_20260812/`

| Setting | Error | Format | Agreement | Focal reward | Focal calls |
|---|---:|---:|---:|---:|---:|
| Buyer | 0 | 100% | 100% | 8.0 | 2 |
| Seller | 0 | 100% | 100% | 20.0 | 2 |
| Resource First | 0 | 100% | 100% | 32.5 | 3 |
| Resource Second | 0 | 100% | 100% | 22.5 | 2 |

Trace audit：

- 四个 setting 均生成独立 `framework_v3_decisions.jsonl`；
- 每个候选包含 `accept_mean`、`accept_tail_q10`、`accept_safeguarded`、`uncertainty_weight`；
- planner 包含 outside option、counter failure opportunity cost 和 gate reason；
- 未发送 `safeguarded_contingent_value <= 0` 的 proposal；
- 0 protocol repair failure。Resource Second 的 opponent raw output 有一次确定性格式修复，不影响 focal framework action。

结论：Canary 通过，进入同 seed 的 10-episode paired validation。单局 reward 不作为方法优劣证据。

## 同 seed paired pilot（1 run × 10 episodes）

V2 reference：`arena_runs/framework_v2_1_fix_validation_n10_20260812/`
V3：`arena_runs/dc_bap_v3_paired_n10_seed20260805/`
Resource/Second protocol-fix clean rerun：`arena_runs/dc_bap_v3_paired_n10_seed20260805_protocolfix/`

| Setting | V2 reward/agreement | V3 reward/agreement | Reward Δ | Calls V2→V3 |
|---|---:|---:|---:|---:|
| Buyer | 14.20 / 90% | 9.70 / 100% | -4.50 | 2.2→2.0 |
| Seller | 17.70 / 90% | 17.30 / 100% | -0.40 | 2.2→2.0 |
| Resource First | 16.40 / 100% | 10.15 / 100% | -6.25 | 2.6→1.8 |
| Resource Second | 14.75 / 90% | 15.25 / 100% | +0.50 | 3.8→2.2 |

所有 clean results 均为 10/10 valid、0 error、100% format success、0 repair failure。

### 发现

1. Safeguard 确实消除了 V2 的 Buyer/Seller/Resource Second 失约，并降低调用与轮数；方向有效。
2. Buyer 在 10 局中全部立即接受。对手观察到这一行为后，opening price 从 50 上升至 54–55，形成 repeated-game exploitation。V3 只计算当前局的 agreement opportunity cost，没有计算“接受更差 deal 会训练对手继续恶化”的跨 episode commitment value。
3. Seller 避免了 V2 的 70 ZUP 过度反锚，但从第 2 局起持续接受 60，低于历史已实现的 63；同样缺少历史 payoff commitment floor。
4. Resource First 的全局 response-support eligibility 过度偏向低 self-utility 安全报价，例如从 V2 的 `10X↔10Y` 降到 `9X↔3Y`。对于没有当前可接受 offer 的 opening action，adverse-tail safeguard 不应把 exploit frontier 整体替换成 safe frontier。
5. Resource Second clean rerun改善了 reward 与 agreement，说明 outside option/deadline safeguard 对 second mover 有价值。

### 决定

V3.0 **不晋级**，因为 Buyer 和 Resource First 超过 1 utility 的退化门槛。下一 iteration 保留 V3 uncertainty、outside option 和 regret gate，增加一个独立且可解释的机制：

- 基于历史已实现自身 payoff 的 reciprocal commitment floor，防止对手通过逐局恶化 opening offer 获利；
- Resource opening 的 exploit family 保留全 candidate frontier，而 safe/fallback 继续使用 response-support gate。

不修改 belief updater，不增加 LLM 调用。
