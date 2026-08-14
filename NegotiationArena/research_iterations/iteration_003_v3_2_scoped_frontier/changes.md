# Code changes

- 新增 `ScopedCommitmentBeliefPlannerAgent`，保留 V3.0/V3.1。
- `resource_exchange && starts_episode && decision_index == 1` 时 exploit 使用完整 frontier。
- Buyer、Seller、Resource/Second 及 Resource/First 后续轮次的所有 family 均使用 response-supported frontier。
- trace 新增 `frontier_scope`，写入独立 `framework_v3_2_decisions.jsonl`。
- 新增作用域回归测试。
