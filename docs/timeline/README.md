# Negotiation Belief–Planner Research：核心文档迁移包

生成日期：2026-08-17

本目录整理了从早期 AgenticPay/RL 探索到当前 reward-first universal belief–planner framework 的
关键可编辑 PPT 与核心报告。文件名统一增加日期和序号，按研究演进顺序排列；原文件仍保留在
`/work5/qixint`，本包是便于下载和长期保存的只读副本。

## 目录

- `presentations/`：12 份关键 PPT，优先保留可编辑版本；
- `reports/`：24 份核心 Markdown 报告；
- `MANIFEST.sha256`：所有文件的 SHA-256 校验值；
- `SOURCE_MANIFEST.tsv`：迁移文件名、原路径、时间和说明。

## 时间线

| 阶段 | 日期 | 代表 PPT | 建议先读的报告 | 主要内容 |
|---|---|---|---|---|
| 初始任务与 RL | 06-03–06-09 | `01_initial_agenticpay_rl`, `02_agenticpay_astra_merit` | `02_agenticpay_experiment_summary` | AgenticPay、ASTRA、prompt/RL 初始实验 |
| Latent belief | 06-10–06-21 | `03_latent_belief_report`, `04_agenticpay_framework_final` | `01_latent_belief_deep_research` | 从生成策略转向 opponent belief 与 planner |
| Benchmark/novelty | 06-26–07-01 | `05_belief_novelty_experiments`, `06_unified_benchmark_plan` | `04_benchmark_reproducibility_plan`, `05_opponent_belief_novelty_ideas` | benchmark 选择与 novelty 假设 |
| Simple/continuous planner | 07-09–07-27 | `07_simple_rlvr_update`, `08_continuous_planner_belief_update` | `06_continual_resolving_plan`, `07_continuous_solver_stage`, `08_codebase_overview` | Simple Env、continuous belief、CEM/planner |
| Multi-issue | 07-29–08-03 | `09_agenticpay_multiissue_outlook` | `10_agenticpay_multiissue_outlook`, `11_casino_belief_usable_planner` | AgenticPay multi-issue 与 CaSiNo 迁移 |
| Related work / benchmark choice | 08-03–08-07 | `10_benchmark_novelty_plan` | `12_current_work_benchmark_plan`, `13_accepted_benchmark_survey`, `15_openreview_related_work_novelty` | LLM-Deliberation、NegotiationArena、ANL、相关工作 |
| Universal framework | 08-07–08-14 | `11_universal_framework_results_plan` | `17_novelty_architecture_plan`, `19_negotiationarena_final_report`, `20_universal_framework_plan` | factorized belief、belief-usable planner、跨环境故事 |
| Reward-first | 08-17 | `12_reward_first_framework_results_plan` | `21_framework_migration_summary`, `22_reward_first_ppt_explanation`, `23_current_framework_primary_results`, `24_reward_first_pipeline_ledger` | 只优化 buyer、固定 seller，以 Simple reward 与 AgenticPay BuyerScore 为终极目标 |

## 推荐阅读顺序

若时间有限，建议依次阅读：

1. `reports/2026-07-27_08_codebase_overview_cn.md`
2. `reports/2026-08-05_15_openreview_related_work_novelty_cn.md`
3. `reports/2026-08-14_20_universal_framework_plan_cn.md`
4. `reports/2026-08-17_21_framework_migration_summary_cn.md`
5. `reports/2026-08-17_24_reward_first_pipeline_ledger_cn.md`
6. `presentations/2026-08-17_12_reward_first_framework_results_plan_editable_en.pptx`

## 范围说明

- 没有收入模型权重、虚拟环境、缓存、原始 API 调用或大规模 trajectory；
- 没有收入同一 PPT 的 `backup_before_*` 版本；
- 论文 PDF 不属于本研究自产文档，未收入本包；原始引用可通过报告中的链接重新获取；
- 2026-08-17 晚间正在运行的 Iteration 012/013 尚未完成，本包中的 pipeline ledger 是打包时快照；
  最终结果会另外收入结果迁移包。

## 完整性检查

下载后在本目录运行：

```bash
sha256sum -c MANIFEST.sha256
```
