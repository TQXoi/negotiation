# Iteration 063：V60 全任务与 focal-only multi-buyer 验证

## 目标

本轮冻结 `universal_framework_v60_resilient_multiseller_settlement`，不再修改
planner、seller、环境或官方 scorer，只回答三个泛化问题：

1. V60 在所有含 multi-seller 的 116 个任务上是否可运行并保持协议有效；
2. V60 在 AgenticPay 全部 231 个任务上是否可运行；
3. 在含 multi-buyer 的 4 个 family 中，只替换 Buyer 1 时，V60 是否优于同样
   focal-only 注入的 repo-native、direct prompt 与 CoT prompt。

## 入口修复

- `auto_loop.py` 不再硬编码 `/snap/bin/codex`，而是验证显式环境变量、VS Code
  Codex 扩展二进制和 PATH 候选；不可执行时给出明确错误。
- `AgenticPay_Env.eval` 新增 `--focal-buyer-index`。它是 BuyerAgent 的 1-based
  构造序号；其余 Buyer 使用上游原生 `BuyerAgent`。
- 每条记录写入 `buyer_patch_audit`。focal 模式若不是恰好替换一个 Buyer，运行
  立即失败，防止把 all-buyers 结果误标成 focal-only。
- 保留官方 `buyer_score`，另记录 `focal_buyer_reward`、
  `focal_buyer_selected` 和 `focal_buyer_max_price`；不自行创造新的 focal score。

## 实验矩阵

| 阶段 | task family | 任务数 | Buyer 策略 | 注入范围 |
|---|---|---:|---|---|
| Smoke | 4 个 multi-buyer family 的 Task1 | 4 | V60 | 仅 Buyer 1 |
| all-116 | 4 个含 multi-seller family | 116 | V60 | 所有 Buyer |
| all-231 | AgenticPay 全部 family | 231 | V60 | 所有 Buyer |
| focal formal | 4 个含 multi-buyer family | 116×4 variants | native/direct/CoT/V60 | 仅 Buyer 1 |

固定项：Qwen3-30B-A3B-Instruct-2507、同一 native seller、seed 20260907、
上游环境与 scorer 不变。长实验均支持相同输出目录下 resume。

## 主要指标

- 官方：mean BuyerScore、agreement/valid-deal、timeout/error；
- focal 因果指标：Buyer 1 的 native reward、被选中率；
- 协议审计：每条记录的 buyer instance 总数、framework focal 数和 native
  control 数；
- 分 task family 报告，避免用容易 family-composition 混淆的单个总体均值。

## 通过门槛

- Smoke：4/4 无 runner error，所有记录 `variant_buyer_instances == 1`；
- all-116/all-231：零 adapter/renderer/validator Python error；timeout 作为策略结果
  单独统计，不混入基础设施错误；
- focal formal：V60 与三个 control 使用完全相同的 task/seed/seller，并报告
  paired BuyerScore delta、focal reward delta 与 task-cluster bootstrap 95% CI。

## 运行中发现与修复（2026-09-04）

all-116 和 all-231 均在 `multi_products_multi_seller/Task5–29` 复现了同一
`AttributeError: 'function' object has no attribute 'resolve_selected_seller'`。
原因不是 V60 planner，而是只读轨迹 instrumentation 把上游环境类替换成了
普通 factory function；上游任务在谈判结束时通过类调用静态 seller-routing
方法，因此丢失该属性。

修复后 instrumentation 使用上游环境的真实子类，只在实例初始化后包装
`step`，从而保留继承关系以及 static/class methods。单元测试加入静态方法保持
断言，共 79 tests passed。此前必现错误的 Task5 在线回归已达成协议，
BuyerScore `47.704`，0 error。原始 25 条错误记录完整保留；当前 focal matrix
结束后，将在独立目录重跑该 family 全部 29 个任务，用于构造无基础设施污染的
corrected all-116/all-231 汇总。
