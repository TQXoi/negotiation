# Code changes

- 新增 `UncertaintySafeguardedBeliefPlannerAgent`，保留 V2 类不变。
- 对每个候选保存 posterior mean acceptance、q10 adverse-tail acceptance 与收缩后的 safeguarded acceptance。
- 高不确定条件下，仅让具有最低 response support 的候选进入五类 frontier；规则只使用 action-conditional 概率，不编码固定价格或资源名称。
- 新增 zero outside option dominance。
- 新增 counter failure opportunity cost 与 deadline-aware agreement-regret gate。
- 新增独立 `framework_v3` / `framework_v3_frozen` CLI 方法。
- 新增独立 trace 文件 `framework_v3_decisions.jsonl`。
- 新增三项 V3 单元测试；总测试数由 22 增至 25。

## Pilot 中发现的 protocol hotfix

Resource/Second episode 2 的 opponent format retry 输出 `Player RED Gives 10X | Player BLUE Gives 10Y`。旧 validator 只检查双方片段存在，没有检查 trade 内资源项，导致原 parser 抛出 `IndexError`。

- 严格 validator 现在只接受规范 `RESOURCE: INTEGER`；
- deterministic canonicalizer 将无歧义的 `10X` 规范成 `X: 10`；
- 重复资源或无法完整解析的 residue 被拒绝；
- 新增回归测试，总测试数增至 26。

该修复只影响协议可靠性，不改变 V3 planner；修复前代码仍保存在 `code_snapshot/`，修复后的完整相关文件保存在 `code_snapshot_after_protocol_fix/`。
