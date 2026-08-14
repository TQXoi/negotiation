# NegotiationArena Framework V2 修复与 Validation 记录（2026-08-12）

## 1. 结论

本轮修复解决了此前 `IndexError: list index out of range`、Buy--Sell 公开动作无法被 belief 模块观察、Resource trade 双方顺序不稳定，以及私有资源字段绕过前置校验等问题。

正式 validation 使用 Qwen3-30B-A3B-Instruct-2507、同模型对手、1 run × 10 episodes、4 个 setting、3 个 method，共 120 episodes。结果为：

- 120/120 episodes 有效，0 个运行错误；
- 12/12 method-setting 组合的 `format_success_rate = 1.0`；
- 0 个 protocol repair failure；
- 四个 setting 均生成了完整 `comparison.json`。

因此，工程问题已修复，可以进入 framework 算法实验。但当前 validation 只有一个 run，每组仅 10 局，只用于工程与初步策略 validation，不能作为论文显著性结果。

## 2. 修复内容

1. Buy--Sell public message 使用规范的 `<other player message>`、`<other player answer>`、`<other player proposed trade>` 标签，使 continuous belief 能真正观察对手行动。
2. Resource trade 在进入原库 parser 前统一为 `Player RED Gives ... | Player BLUE Gives ...`，只调整双方顺序，不修改资源数量。
3. 当 Resource 输出包含完整 trade、但遗漏 `<player answer>` 时，确定性恢复为 `NONE`；Buy--Sell 对应恢复为 `PROPOSAL`。
4. 对 `<my resources>` 执行 `KEY: INTEGER` 语法校验，阻止格式错误进入 `Resources.from_string`。
5. format retry prompt 不再包含容易被模型逐字复制的 `ACCEPT|NONE` 等占位字符串，并补充规范的私有 resources/goals。
6. belief event reader 同时支持规范标签和旧版 malformed trace，保留历史结果兼容性。

## 3. Validation 协议

- Actor/opponent model: `Qwen3-30B-A3B-Instruct-2507-base`
- Endpoint: `http://127.0.0.1:8002/v1`
- Settings: `buyer`, `seller`, `resource_first`, `resource_second`
- Methods: `direct`, `opponent_simulation_paper`, `framework_v2`
- Runs: 1
- Episodes per run: 10
- Max turns: 10
- Opponent Simulation candidates: 5
- Protocol mode: `normalize_retry`
- Seed: 20260805

注意：这是同模型本地公平对比，不是论文中 Gemini-2.5-Flash opponent 配置的数值复现，因此不能把本表绝对值直接与论文表格对齐。

## 4. 正式结果

| Setting | Method | Valid/Error | Format | Agreement | Focal reward | Joint reward | Focal calls/episode |
|---|---|---:|---:|---:|---:|---:|---:|
| Buyer | Direct | 10/0 | 100% | 100% | 13.50 | 20.00 | 1.1 |
| Buyer | Opponent Simulation | 10/0 | 100% | 100% | 12.70 | 20.00 | 11.0 |
| Buyer | Framework V2 | 10/0 | 100% | 90% | 14.20 | 18.00 | 2.2 |
| Seller | Direct | 10/0 | 100% | 100% | 19.60 | 20.00 | 1.1 |
| Seller | Opponent Simulation | 10/0 | 100% | 100% | 19.40 | 20.00 | 10.0 |
| Seller | Framework V2 | 10/0 | 100% | 90% | 17.70 | 18.00 | 2.2 |
| Resource/First | Direct | 10/0 | 100% | 100% | 8.55 | 22.20 | 1.7 |
| Resource/First | Opponent Simulation | 10/0 | 100% | 100% | 6.10 | 22.40 | 23.0 |
| Resource/First | Framework V2 | 10/0 | 100% | 100% | 16.40 | 29.20 | 2.6 |
| Resource/Second | Direct | 10/0 | 100% | 90% | -0.90 | 15.60 | 1.7 |
| Resource/Second | Opponent Simulation | 10/0 | 100% | 100% | 17.45 | 25.00 | 16.9 |
| Resource/Second | Framework V2 | 10/0 | 100% | 90% | 14.75 | 45.40 | 3.8 |

## 5. 结果解释与下一步门槛

工程 validation 通过。按这 10 局的描述性统计，Framework V2 在 Resource/First 上同时明显提高 focal 与 joint reward；在 Resource/Second 上 joint reward 最高、调用成本明显低于 Opponent Simulation，但 focal reward 暂低于 Opponent Simulation。Buyer 的 framework focal reward 略高，但损失一局成交；Seller 则因一次 70 ZUP 过度锚定被拒绝，平均收益低于两条 baseline。这些差异尚未经过多 seed 显著性检验。

因此下一步不是继续修 parser，而是改进 planner 的 risk control：

1. Buyer/Seller 增加 deadline-aware agreement floor，避免已有正 surplus 时为了小幅提升而承担全局失约风险。
2. Seller 对超出已观测 buyer ceiling 的探索报价加入 posterior-tail probability 与 expected regret 门控。
3. Resource planner 将 self reward、joint reward、agreement probability 分开报告并消融，避免只靠高 joint utility 掩盖 focal utility 损失。
4. 正式论文实验至少运行 5 seeds × 20 episodes，并报告 paired bootstrap CI、model calls、tokens 与 wall time。

## 6. 可复查产物

- 正式结果：`arena_runs/framework_v2_1_fix_validation_n10_20260812/`
- 全 setting canary：`arena_runs/framework_v2_1_fix_canary_20260812/`
- Resource/Second 新进程 canary：`arena_runs/framework_v2_1_fix_canary_resource_second_rerun_20260812/`
- 每个 method 目录包含 `config.json`、`episodes.jsonl`、`summary.json`；每个 setting 包含 `comparison.json`。
- 最终测试：22/22 unittest 通过，`compileall` 通过。
