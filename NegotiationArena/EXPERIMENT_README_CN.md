# Qwen3-30B / GPT-4 Buy-Sell 复现与 Framework 对照

> 本文记录的是早期独立单局 `40/60` protocol。与 2026 Opponent Simulation
> 论文的 `43/63 + 20 repeated episodes + Resource Exchange` 正式对照请改用
> [自然语言方法与实验计划](../../8.5/NATURAL_LANGUAGE_NEGOTIATION_BELIEF_FRAMEWORK_AND_EXPERIMENT_PLAN_V2_CN.md)
> 以及 [Framework V2 与基线复现说明](NEGOTIATIONARENA_FRAMEWORK_V2_BASELINE_REPRODUCTION_CN.md)。
> 新实验使用 `scripts/run_paper_baselines_framework_v2_qwen30b.sh`；两组结果不可直接合并。

本实现先复现原论文最稳定的 Buy/Sell `seller cost=40, buyer WTP=60` 设置。`direct` 是原论文式 structured prompting；`framework_seller` 和 `framework_buyer` 每局只替换一个 focal agent，对手保持 direct，避免双方同时改变造成归因困难。

调用层只使用 Python 标准库 `urllib`，不需要安装 `openai` SDK；同一 client 同时支持本地 vLLM 与 OpenAI 官方 endpoint。

## Qwen3-30B（8002）

先确认服务中的准确模型名：

```bash
curl http://127.0.0.1:8002/v1/models
```

然后运行：

```bash
cd NegotiationArena
NEGOTIATION_QWEN_MODEL='服务返回的模型名' EPISODES=20 \
  bash scripts/run_qwen30b_buysell_comparison.sh
```

正式表至少使用 60 episodes；20 仅用于 smoke/初步方差判断。

## GPT-4

API key 仅从环境变量读取：

```bash
export OPENAI_API_KEY='...'
export NEGOTIATION_GPT4_MODEL='gpt-4o'
EPISODES=60 bash scripts/run_gpt4_buysell_reproduction.sh
```

如需严格历史复现，可把模型名设为仍可访问的 GPT-4 snapshot；现代 `gpt-4o` 结果应标记为 benchmark replication，而不是 2023 snapshot exact reproduction。

## 原论文 cross-model protocol

原论文 Buy/Sell 主实验排除同模型配对；每个有序模型对独立运行。此前 `gpt-4o × gpt-4o` self-play 不是论文中“GPT-4 buyer 平均成交价约 41”的对应实验。当前有 Qwen 与 GPT-4o 时，运行两个有序 cross-model cells：

```bash
export OPENAI_API_KEY='...'
EPISODES=60 bash scripts/run_paper_protocol_qwen_gpt4.sh
```

脚本还恢复了 paper branch 的终止语义：`ACCEPT` 和 `REJECT` 都立即结束；main branch 的通用基类只检查 `ACCEPT`，不应直接用于论文复现。

原始 worktree 固定在 `NegotiationArena_paper`（commit `d35a7a3a`）。归档原结果可用 `arena_integration/summarize_original_archive.py` 独立汇总。

每个 mode 保存 `episodes.jsonl`、持续更新的 `summary.json`、原平台完整 logs，以及 framework 每轮 `belief/candidates/chosen/realized` trace。所有脚本支持 `--resume`，不会覆盖已完成 rollout。
