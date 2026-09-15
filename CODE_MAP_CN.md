# 代码焦点：后续只从这些入口修改

`negotiation_final` 保留一些历史 variant，是为了复现已发生的实验；以下是未来代码修改的 focus set。此表也解释为什么代码没有强行按 `environments/` 重新嵌套：顶层 package 名是现有 Python import 和 benchmark runner 的接口。

| 模块 | 重点文件 | 责任 | 下一次合理修改 |
|---|---|---|---|
| Shared schema/engine | `framework/schemas.py`、`framework/engine.py` | canonical state、belief-to-action flow | 明确每条 belief 的 evidence/reliability、统一 adapter contracts |
| Shared belief | `framework/belief.py`、`framework/events.py` | verified update、posterior、abstention | calibration、old/new regime mixture、per-opponent identity |
| Shared planner | `framework/planner.py`、`framework/conservative_awr_planner.py` | legal candidate rank、frontier、probe、terminal safety | 一个冻结的跨环境 scoring policy，reward-first safe residual |
| Simple | `Simple_Env/buyer/universal_framework.py`、`Simple_Env/eval.py` | 单价格 adapter、V5.9/V6.3 | 独立 held-out、多 seller style、frozen/shuffled controls |
| CaSiNo | `CaSiNo_Env/buyer/belief_usable_planner/`、`CaSiNo_Env/eval.py` | 3-issue allocation + chooser | 接 shared belief/planner interface，补 multi-rollout CI |
| AgenticPay | `AgenticPay_Env/buyer/universal_framework.py`、`issue_classifier.py`、`environment/all_tasks.py` | contract、per-seller state、V60 | full family修复验证；后续拆 contract grammar / role gate / router |
| AgenticPay entry | `AgenticPay_Env/eval.py`、`experiments/run_agenticpay_single28_framework.py` | official task runner/BuyerScore | paired focal-only all-116/all-231、API failure resume |
| NegotiationArena | `NegotiationArena/arena_integration/decision_calibrated_agent.py`、`paper_aligned_opponent_simulation.py` | 四 setting 与直接竞争 baseline | 只冻结一个 Ours，统一 shared interface |
| Training | `experiments/train_agenticpay_issue_role_gate.py`、`train_agenticpay_settlement_eligibility_gate.py`、`experiments/negotiation_rl/` | 低成本、可拒绝 gate 与安全 residual | 新 train/dev task split，官方 reward-label policy |
| Regression | `tests/`、`AgenticPay_Env/tests/`、`Simple_Env/tests/`、`NegotiationArena/tests/` | 协议/score/parser/selection 锁定 | 新 topology 每修一 bug 加一个在线或 mock test |

## 历史但仍需保留的代码

- `Simple_Env/buyer/continuous_solver/` 与 `price_policy_v03/`：早期 planner/training 试验及 identifiable utility-profile suite；部分测试/工具仍直接引用，不可贸然移动。
- `AgenticPay_Env/buyer/universal_framework.py` 的 V1–V59：复现历史失败与对照；不要继续修改旧行为。
- `NegotiationArena/negotiationarena/`、`games/`：原 benchmark parser/prompt/runner，对比论文原设置需要，保留 license/attribution。
- `experiments/external_comparisons/`：Simple adapter 仍依赖其 RLVR action parser，不是可以随意删除的“旧论文脚本”。

每次清理代码先用 `rg` 查 import/CLI/test 引用，再在 `negotiation_past/` 保留可恢复副本；不要只按文件名判断废弃。
