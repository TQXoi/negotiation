# NegotiationArena：Opponent Simulation 基线复现与 Framework V2 测试说明

> 更新日期：2026-08-09
> 目标：在同一 repeated NegotiationArena protocol 中比较原始 direct baseline、Opponent Simulation 和我们的 belief-usable framework。

## 1. 当前结论

本目录现在可以进行三方、成对且可审计的比较：

| 方法 | runner 名称 | 实现含义 | 每个 focal action 的主要 LLM 调用 |
|---|---|---|---:|
| Direct | `direct` | 原 NegotiationArena action prompt + 论文公开 strategic-brainstorming prompt + repeated outcome reminder | 1 |
| Opponent Simulation（旧） | `opponent_simulation` | 历史本地三阶段近似；一次请求联合模拟全部候选 | 3 |
| Opponent Simulation（论文对齐） | `opponent_simulation_paper` | 5 个独立 actor action samples；每个候选各自做完整 serialized future rollout；按预测 focal reward 选 action | 最多 10 |
| Framework V1 | `framework` | 历史 prompt-only belief/candidate/chooser，保留用于复现实验 | 3 |
| Framework V2 | `framework_v2` | bounded semantic evidence + structured continuous posterior + common candidate frontier + reciprocal contingent planner + action lock | 通常 2 |

论文作者公开了 prompt 和十组 example logs，但没有公开完整 Algorithm-1 runner。因此 `opponent_simulation_paper` 是依据论文与公开 prompt 重构的 **paper-aligned baseline**，不是官方代码的逐行复现。旧 `opponent_simulation` 不删除，防止覆盖旧结果，但不能再作为最接近论文算法的主 baseline。

## 2. 论文 setting 的对齐情况

固定配置如下：

- 每个 setting：10 runs × 20 repeated episodes；
- 每个 episode 最多 10 turns；
- candidate budget `N=5`；
- Buyer–Seller：seller cost=43，buyer WTP=63；
- Resource Exchange：RED=(X25,Y5)，BLUE=(X5,Y25)；RED values=(0.5,2.5)，BLUE values=(2.5,0.5)；
- buyer focal、seller focal 均从第二位行动；resource focal 分别测试 first/second；
- strategic-brainstorming prompt 同时施加给双方，与公开日志一致；
- 每 episode 保留同一 opponent，下一 episode 重置局内 dialogue，但携带过去 deal/reward reminder；
- actor、actual opponent、seed、first mover、protocol mode 必须在三个方法间一致。

论文数值使用的 actual opponent 是 `Gemini-2.5-Flash`。如果 actor 和 actual opponent 都连接本地 Qwen，得到的是 **matched Qwen self-play comparison**，可以比较方法，但不能宣称数值复现论文 Table。

## 3. Framework V2

### 3.1 Continuous belief

V2 不再让 LLM 自由写一个不可校准的 belief JSON。它维护两类显式 particle posterior：

- Buyer–Seller：对手 reservation/floor/ceiling 的离散分布；
- Resource Exchange：对手 X/Y 相对价值权重的离散分布。

每次对手公开行动后使用以下证据更新：

1. 对手是否接受、counter 或退出；
2. 对手提出的结构化 price/bundle；
3. 对手公开话语中的 reservation claim、firmness、concession signal。

第三类证据由 LLM 只做受约束抽取。代码把其可靠性限制在弱证据范围，LLM 不能直接生成 posterior 或 planner score。谈判中的自报 cost/budget 仅被视为可能不真实的 claim。

### 3.2 Belief-usable planner

每个候选都记录：

- exact own utility；
- `P(accept/counter/reject | action, belief)`；
- continuation value；
- downside risk；
- bounded decision value of information；
- concession cost 与 reciprocity ledger 中的 uncompensated concession。

Planner 比较接受当前报价与最佳 counterfactual candidate 的价值。输出 action 后立即锁定 price、bundle 和 action type；LLM language realizer 只能改公开措辞，不能改交易内容。

因此 V2 的关键区别不是“比 Opponent Simulation 多一段 belief 文本”，而是：belief 是候选条件化且可干预的概率对象，planner 的每个分数项可以从 trace 中检查。

## 4. 运行方法

### 4.1 最小 smoke test（建议先运行）

确认 8002 已启动后：

```bash
cd NegotiationArena
RUNS=1 EPISODES_PER_RUN=3 \
OUTPUT_ROOT=arena_runs/paper_baselines_framework_v2_qwen30b_smoke \
bash scripts/run_paper_baselines_framework_v2_qwen30b.sh
```

只先跑关键的 Resource Exchange：

```bash
SETTINGS="resource_first resource_second" RUNS=1 EPISODES_PER_RUN=3 \
OUTPUT_ROOT=arena_runs/paper_baselines_framework_v2_resource_smoke \
bash scripts/run_paper_baselines_framework_v2_qwen30b.sh
```

### 4.2 本地 Qwen 正式 matched comparison

```bash
RUNS=10 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/paper_baselines_framework_v2_qwen30b_n10x20 \
bash scripts/run_paper_baselines_framework_v2_qwen30b.sh
```

上述命令共 4 settings × 3 methods × 200 episodes。`opponent_simulation_paper` 的调用量最大，应在 smoke 验证后再启动正式实验。

### 4.3 使用论文 actual opponent

需要另行提供一个兼容 OpenAI Chat Completions 的 Gemini endpoint：

```bash
export NEGOTIATION_OPPONENT_MODEL=gemini-2.5-flash
export NEGOTIATION_OPPONENT_BASE_URL=https://YOUR_COMPATIBLE_ENDPOINT/v1
export NEGOTIATION_OPPONENT_API_KEY_ENV=GEMINI_COMPAT_API_KEY
export GEMINI_COMPAT_API_KEY=YOUR_KEY
RUNS=10 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/paper_actor_qwen_opponent_gemini_n10x20 \
bash scripts/run_paper_baselines_framework_v2_qwen30b.sh
```

只有这种配置才进入与论文 Qwen row 的数值对齐阶段；仍需在报告中注明作者没有公开完整 runner。

### 4.4 Raw protocol 与 parser-safe protocol

论文复现诊断：

```bash
PROTOCOL_MODE=raw RUNS=1 EPISODES_PER_RUN=3 \
OUTPUT_ROOT=arena_runs/paper_raw_protocol_diagnostic \
bash scripts/run_paper_baselines_framework_v2_qwen30b.sh
```

主比较默认用 `normalize_retry`，三个方法共享同一 protocol layer，并报告 raw invalid、deterministic repair、retry 和 repair failure。格式失败按 agreement=0、reward=0 进入主指标；不能像旧 summary 那样删除失败 episode。

## 5. 输出与判读

每个 setting 生成：

```text
OUTPUT_ROOT/
  buyer|seller|resource_first|resource_second/
    direct/
    opponent_simulation_paper/
    framework_v2/
    comparison.json
```

主指标：

- `completion_rate` 与 `format_success_rate`；
- `all_episode_mean_focal_reward`；
- `all_episode_agreement_rate`；
- 对 direct 的 paired run-level reward delta 与 bootstrap 95% CI；
- early episode 1–5 与 late episode 16–20 reward；
- focal model calls / episode 与 protocol intervention counts。

所有 API 请求、响应和错误保存在 `model_traces/**/calls.jsonl`；V2 决策保存在 `framework_v2_decisions.jsonl`；Opponent Simulation 选择记录在 `oppsim_paper_decisions.jsonl`。

## 6. 当前验证状态

- `python -m unittest tests.test_repeated_integration_unittest -v`：18/18 通过；
- `python -m compileall -q arena_integration negotiationarena games tests`：通过；
- 新脚本 `bash -n`：通过；
- 8002 在线 smoke：尚未完成。2026-08-09 检查 `127.0.0.1:8002` 返回 connection refused，因此没有生成或伪造新的模型结果。

旧目录 `arena_runs/oppsim_qwen30b_formal_n10x20` 的有效 episode 比例过低，而且旧 summary 删除 error rows，不能作为正式结论。新实验必须写到新目录，并以 `comparison.json` 的 all-episode 指标为准。

## 7. 下一组必要 ablation

主三方 smoke 稳定后，再运行：

1. `framework_v2_frozen`：只保留首次局内 update；
2. `framework_v2 --no-semantic-belief`：验证自然语言 evidence 的增量价值；
3. `framework_v2 --no-language-realizer`：验证收益不是语言润色造成；
4. 共享 candidates 下的 oracle/shuffled posterior；
5. opponent-switch episode，测 adaptation lag 和 belief recovery。

前三项已经有 CLI 支持；oracle/shuffled 与 switch track 应在主 baseline smoke 通过后加入，避免同时改变过多实验因素。

## 8. 参考

- Opponent Simulation paper: <https://arxiv.org/abs/2602.19309>
- 作者公开 prompt / examples: <https://github.com/llmnegotiationsubmission/llmnegotiationsubmission>
- NegotiationArena paper/code: <https://github.com/Alab-NII/negotiation-arena>
