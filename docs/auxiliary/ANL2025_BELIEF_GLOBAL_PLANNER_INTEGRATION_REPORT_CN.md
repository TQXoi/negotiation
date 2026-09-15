# ANL2025 Continuous Belief + Global Planner 基础测试实现报告

日期：2026-08-05

## 1. 当前结论

ANL2025 的基础测试代码已经接入官方 `anl-2025-platform`，可以直接测试我们的 framework。实现不是把 ANL2024 的双边 reservation-value planner 原样移植，而是针对 ANL2025 的 sequential multi-deal setting 做了对应分解：

- 每一条 edge negotiation 维护独立、在线更新的 partner preference/acceptance belief；
- planner 直接调用 center 已知的 `CenterUFun`，评估当前交易对最终整组 agreements 的影响；
- 对未来尚未开始的 edge 建立 accept/reject continuation beam，保留交易之间的互补、替代与目标数量关系；
- generator 由 ANL 的结构化 `Outcome` action 取代，因此本实验隔离测量 belief 和 planner，不包含自然语言格式收益；
- 所有正式 variant 均强制使用 `share_ufuns=False`。标准 static/continuous variant 从未获得 edge utility function；oracle 和 shuffled 只用于 evaluator-side causal intervention。

当前三个官方场景 `dinners`、`job_hunt`、`target_quantity` 均已完成 smoke test，所有 baseline 与 framework variants 均无运行错误。新增的 6 个 ANL2025 单元/集成测试全部通过。

## 2. 代码结构

根目录：`/work5/qixint/external_negotiation_envs/ANL_2024_2026`

| 文件 | 作用 |
|---|---|
| `belief_planner_anl/anl2025/belief.py` | 每条 edge 的 continuous preference + latent acceptance-threshold belief |
| `belief_planner_anl/anl2025/planner.py` | 基于最终 center utility 的 global continuation planner |
| `belief_planner_anl/anl2025/agent.py` | 接入官方 `ANL2025Negotiator` 的 center controller |
| `belief_planner_anl/anl2025/runner.py` | 官方场景、baseline、消融、配对种子、评测和逐 episode 落盘 |
| `scripts/run_anl2025_basic.sh` | 一键运行三类官方场景 |
| `tests/test_anl2025_framework.py` | belief、global planning、truth isolation 和官方场景测试 |

## 3. Framework 在 ANL2025 中的对应关系

### 3.1 Environment

采用官方 sequential one-to-many protocol。Center 依次和多个 edge 协商，每条边产生一个 agreement 或 `None`；最终收益不是简单的当前边收益，而是：

\[
U_C(o_1, o_2, \ldots, o_n).
\]

运行器固定：

- `method="sequential"`；
- `share_ufuns=False`；
- 相同场景、随机种子、edge 类型与次序下进行 paired variant comparison；
- edge opponent 从 `Boulware2025`、`Linear2025`、`Conceder2025` 做按 seed 固定的随机排列，所有 paired variants 使用同一 assignment。

### 3.2 Belief model

对第 (i) 条 edge 维护 (B_i^t)：

1. 从对方历史报价中的 issue-value frequency 学习其相对偏好；
2. 维护对方最低可接受 preference score 的离散 posterior；
3. 用对方报价、我方报价被拒绝、我方报价被接受连续更新；
4. 输出候选 deal 的接受概率和 expected information gain。

当前版本是无需训练的结构化 online model，目的是先验证 belief 是否被 planner 使用。后续可用 learned ranker/calibrator 替换 belief 内核，但不改变 ANL adapter 和 global planner 接口。

### 3.3 Global belief-usable planner

对当前第 (i) 条边的候选 (o_i)，planner 比较：

\[
P_i(\mathrm{accept}\mid o_i,B_i^t)
V(o_{<i},o_i)
+
(1-P_i)V(o_{<i},\varnothing)
+
\lambda(1-t)\operatorname{IG}(o_i).
\]

这里的 (V) 不是当前边 utility，而是对未来 edges 的 accept/reject branches 做 beam continuation 后的期望最终 `CenterUFun`。因此 planner 能表示：

- 某项当前 deal 只有和未来 deal 组合时才有价值；
- 已经达到 target quantity 后不应继续以高成本成交；
- 当前 deal 和未来 deal 互为替代；
- 应为更高价值的未来组合保留可行空间。

接受策略采用“chosen deal 的确定性 continuation value”作为早期 aspiration，并随 deadline 向 reject/no-deal continuation value 让步。它不会错误地把 proposal 的接受概率折扣期望直接当作接受门槛。

## 4. 测试 variants

### 官方 baseline

- `boulware2025`
- `linear2025`
- `conceder2025`

### Framework 与消融

- `framework_static_belief_global_planner`：固定无信息 belief + global planner；
- `framework_continuous_belief_local_planner`：continuous belief，但不规划未来 edges；
- `framework_continuous_belief_global_planner`：完整方法；
- `framework_no_information_gain`：完整方法去掉信息价值；
- `framework_oracle_partner_belief`：真实 edge preference，仅作 causal diagnosis；它没有 oracle opponent policy，因此不是严格的收益上界；
- `framework_shuffled_partner_belief`：把 edge truth 循环错配，用于检验 planner 是否真的使用 belief。

推荐论文主表至少报告前三个 baseline、static、local、完整方法；oracle/shuffled 放 causal analysis，no-IG 放 ablation。

## 5. 指标与记录

每个 episode 写入 `episodes.jsonl`，包含：

- center utility；
- 每条 edge utility 与 mean edge utility；
- agreement fraction、full-agreement；
- belief 的 pairwise preference ranking accuracy；
- acceptance-probability Brier score；
- 完整 framework action/belief/candidate trace；
- seed、edge type assignment、场景、协议参数和错误信息。

`summary.json` 另外给出按 variant 聚合的指标，以及相对 static reference 的：

- agreement vector flip rate；
- framework action sequence flip rate；
- paired center utility delta。

同时输出 `summary_by_scenario_family`，避免把 dinners、job-hunt、target-quantity 与 generated utility 不加区分地混成一个数字。每个 variant 会给 center utility 的 95% normal-approximation confidence interval；paired diagnostics 还报告 center-utility win rate。样本很小时该区间只用于程序核对，正式统计应进一步使用 paired bootstrap/Wilcoxon。

`config.json` 保存实验参数和 framework source hash。JSONL 每完成一局立即 flush；中断后可直接给 runner 加 `--resume` 继续。

Resume 有版本保护：如果 scenarios、variants、seed、步数或 framework hash 与原目录不一致，runner 会拒绝续跑，要求使用新输出目录，从而避免把修改前后的代码结果混在同一个 JSONL 中。

## 6. 已完成验证

### 自动测试

```bash
cd /work5/qixint/external_negotiation_envs/ANL_2024_2026
PYTHONPATH="official/negmas/src:official/anl-platform-current/src:official/anl-2025-platform/src:." \
MPLCONFIGDIR=/tmp/anl2025-mpl \
/work5/qixint/miniconda3/envs/research/bin/python -m pytest -q \
  tests/test_anl2025_framework.py
```

结果：`6 passed`。

测试覆盖：

- continuous belief 能从 edge offer 改变 value ranking；
- static belief 不改变 ranking；
- 在人工构造的跨边互补效用中，global continuation 会选择 local planner 不会选择的组合；
- 官方 dinners 场景完整运行；
- standard trace 无 evaluator truth；
- oracle preference ranking accuracy 为 1；
- baseline/framework 使用相同 paired protocol configuration。

### Smoke 结果的解释边界

目前 smoke 只用于证明代码与 causal variants 可运行，不能作为论文结论。少量 episode 中已经看到：

- `dinners` 的 oracle belief 可明显改变 center utility 和成交组合；
- `target_quantity` 中 global/static 策略与官方 Boulware 产生不同结果；
- continuous belief 的 preference ranking accuracy 在有多轮交互时高于 static 的 0.5；
- 部分场景中完整 continuous 方法尚未超过 static/global，说明当前基于报价频率的 belief 仍较弱，而不是应该隐藏的正结果。

截至 2026-08-05，保存的两组 smoke 为：

| 数据 | 每个 variant 的 episode 数 | static/global | continuous/local | 完整 continuous/global | oracle | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 三个官方固定场景 | 3 | 0.431 | 0.358 | 0.358 | 0.633 | oracle gap 为正，但完整方法未超过 static |
| `generated` + `generated_max` | 4 | 0.766 | 0.614 | 0.563 | 0.610 | 完整方法与 oracle 均低于 static |

表中是 mean center utility，仅是极小样本诊断。generated smoke 中 continuous belief 的 pairwise preference accuracy 约为 0.760，而 static 为 0.5；所以当前主要问题不是“belief 完全没有学到排序”，而是 acceptance calibration、exploration cost 和 global continuation decision 没有把更准的排序稳定转化成收益。正式昂贵实验前应先满足以下 development gate：

1. oracle 在每个主要 scenario family 上应形成稳定正 gap；
2. shuffled 应明显劣于 oracle，最好也劣于 continuous；
3. continuous/global 应在 paired utility 上不劣于 static/global；
4. global 相对 local 的提升应主要出现在 `target_quantity`、`generated_max` 或其他有跨 deal dependency 的场景，而不是只改变 action trace。

原始 smoke 分别保存在 `runs/anl2025_framework_smoke_20260805` 和 `runs/anl2025_generated_smoke_clean_20260805`。首次 generated 运行因中断产生并发追加，完整调用记录保留在 `runs/anl2025_generated_smoke_20260805`，并已用 `INTERRUPTED_RUN_DO_NOT_USE.md` 标明不得用于比较。因此当前交付应被称为“可复现的 ANL2025 evaluation harness + framework v1”，而不是已经得到正结果的最终方法。下一轮 planner v2 应优先校准 response dynamics、降低未来分支的乐观偏差，并用 oracle-policy 与 oracle-preference 两种不同干预拆分 belief error 和 planning error。

正式结论必须使用更多随机场景、edge order/opponent mixtures 和至少 30 个 paired generated scenarios。官方固定场景上的 repetition 主要改变 edge opponent assignment，不能被当成 30 个独立场景。

## 7. 运行命令

### 快速 pilot

```bash
cd /work5/qixint/external_negotiation_envs/ANL_2024_2026
REPETITIONS=3 NSTEPS=30 \
  ./scripts/run_anl2025_basic.sh runs/anl2025_framework_pilot
```

### 较完整的第一轮实验

```bash
cd /work5/qixint/external_negotiation_envs/ANL_2024_2026
SCENARIOS="dinners job_hunt target_quantity generated generated_max" \
REPETITIONS=30 NSTEPS=100 \
  ./scripts/run_anl2025_basic.sh runs/anl2025_framework_main_n30
```

建议先把这条命令视为 development run。只有达到上面的 development gate 后，才固定代码 hash，并更换 seed 做 locked test：

```bash
cd /work5/qixint/external_negotiation_envs/ANL_2024_2026
SCENARIOS="dinners job_hunt target_quantity generated generated_max" \
REPETITIONS=100 NSTEPS=100 SEED=20250901 \
  ./scripts/run_anl2025_basic.sh runs/anl2025_framework_locked_seed20250901_n100
```

不要用 locked-test seed 反复修改 planner；所有调参应只看 development 输出目录。

### 中断后恢复

一键脚本默认新建/覆盖实验。需要 resume 时直接运行模块：

```bash
PYTHONPATH="official/negmas/src:official/anl-platform-current/src:official/anl-2025-platform/src:." \
MPLCONFIGDIR=/tmp/anl2025-mpl \
/work5/qixint/miniconda3/envs/research/bin/python \
  -m belief_planner_anl.anl2025.runner \
  --scenarios dinners job_hunt target_quantity generated generated_max \
  --repetitions 30 --nsteps 100 --seed 20250805 \
  --output-dir runs/anl2025_framework_main_n30 \
  --resume
```

## 8. 下一步实验建议

1. 先跑官方 3 个固定场景作为 case study，并跑至少 30 个按 repetition 独立生成的 `generated`/`generated_max` 场景作为 paired main table；检查完整方法相对 static 和 local 的 utility delta，而不仅看 agreement；
2. 增加生成场景，分层控制 edge 数、issue 数和 center utility 类型；
3. 将 center utility 类型分为 additive、target、complementary、substitutable 四组，验证 global planner 的收益是否随 cross-deal dependency 增大；
4. 从 trajectory 构造 edge preference pair 与 accept/reject calibration data，再比较 heuristic belief 和 learned belief；
5. 必须同时报告 preference-oracle 和 shuffled intervention；如果 oracle 不提升或 shuffled 不下降，优先诊断 acceptance/continuation planner，不能直接把 oracle 当作理论上界；
6. 在当前无 LLM harness 上完成机制诊断后，按下一节加入 LLM belief updater 和 LLM planner/chooser。ANL2025 虽没有自然语言通信，但仍可用 LLM 维护结构化 belief 并进行决策；两类实验需要分别报告。

## 9. LLM-based ANL2025 Framework 扩展计划

### 9.1 为什么结构化环境仍然需要 LLM variant

ANL2025 的 agent 间通信不是自然语言，而是结构化 `Outcome`、`Accept`、`Reject` 和 `End`。这只意味着环境不测试语言说服或文本理解，不意味着 agent 内部不能使用大模型推断 belief 和规划。

如果论文声称的是一个 **LLM negotiation agent framework**，那么当前纯算法实现只能作为 mechanistic baseline 和 evaluator harness；完整方法还应包含：

1. LLM 根据结构化历史持续更新 partner belief；
2. LLM planner/chooser 显式读取 belief 和 global candidate features；
3. 程序化 action lock 将模型决策约束为合法的 ANL action。

这样可以把两个问题分开：

- ANL2025 测试 LLM 是否具备 belief tracking、uncertainty-aware exploration 和 multi-deal planning；
- 自然语言 benchmark 再测试偏好信息抽取、说服、措辞和 generator grounding。

因此 ANL2025 能验证 framework 的“认知结构”，但不能单独证明自然语言谈判能力。

### 9.2 推荐架构

```text
ANL structured history
        ↓
Public evidence builder
        ↓
LLM continuous belief updater
        ↓
Schema validation + numerical calibration
        ↓
Programmatic global candidate generator
        ↓
LLM belief-aware planner / chooser
        ↓
Candidate-ID validator + deterministic action lock
        ↓
Offer / Accept / Reject
```

保留程序化 candidate generator 很重要：`CenterUFun` 的精确计算、合法 outcome 枚举和未来组合搜索由代码完成；LLM 负责整合不确定 belief、比较候选、判断探索价值，而不是凭文本虚构一个 outcome。

### 9.3 LLM continuous belief 接口

每条 edge 维护独立 belief。收到新的 partner offer，或观察到我方 proposal 被接受/拒绝后，调用一次 updater。模型只看到：

- 公开 domain schema 和合法 issue values；
- 当前 edge 的公开 offer/response history；
- 当前时间、剩余轮数；
- 上一次 belief snapshot；
- 带稳定 ID 的新增 evidence。

建议严格输出：

```json
{
  "edge_id": 2,
  "update_index": 4,
  "issue_preferences": {
    "salary": {
      "high": 0.72,
      "medium": 0.23,
      "low": 0.05
    }
  },
  "acceptance_threshold": {
    "mean": 0.61,
    "q10": 0.44,
    "q90": 0.78
  },
  "concession_type": {
    "boulware": 0.55,
    "linear": 0.30,
    "conceder": 0.15
  },
  "uncertainty": 0.36,
  "evidence_refs": ["edge2_offer1", "edge2_reject3"]
}
```

Validator 必须检查：

- issue/value 必须来自公开 domain；
- probability 非负且归一化；
- `q10 <= mean <= q90`；
- evidence reference 必须真实存在；
- 失败时最多进行一次 schema repair，再失败则回退到上一个 snapshot；
- 不允许使用 `flexibility_delta` 之类没有绝对语义、无法校准的自由字段替代 posterior。

LLM 输出仍需映射为 planner 可使用的数值接口：`preference_score(outcome)`、`acceptance_probability(outcome,t)` 和 uncertainty/IG。LLM 生成的是结构化 posterior 参数，而不是无法监督的自然语言猜测。

### 9.4 LLM belief-aware planner / chooser 接口

每次需要 propose 或 respond 时，程序先生成 top-K 合法候选。每个候选包含：

- candidate ID 与结构化 outcome；
- 当前 deal 接受/拒绝时的 center utility；
- global continuation value；
- belief-predicted acceptance probability；
- information gain；
- optimistic、mean、pessimistic 三种 future values；
- 当前已完成 agreements 和剩余 edges 摘要。

LLM 只能返回 candidate ID 或接受当前报价：

```json
{
  "action": "OFFER",
  "selected_candidate_id": "edge2-c17",
  "accept_received_offer": false,
  "future_assumptions": {
    "edge3_success_probability": 0.64
  },
  "exploration_value": 0.18,
  "reason_codes": [
    "global_complementarity",
    "partner_acceptance_plausible"
  ]
}
```

Action lock 随后验证：

- ID 是否属于本轮 candidate set；
- `ACCEPT` 时是否确实存在 received offer；
- action 是否符合 ANL protocol；
- 输出异常时采用记录明确的 deterministic fallback。

LLM 不直接计算或改写 `CenterUFun`，也不能自由生成不在 outcome space 中的报价。

### 9.5 必须比较的 variants

| Variant | Belief | Planner/chooser | 研究问题 |
|---|---|---|---|
| `direct_llm` | 无显式 belief | LLM 直接从 history 选候选 | 普通 prompting baseline |
| `heuristic_belief_algorithmic_planner` | 当前 heuristic continuous | 当前 global planner | 无 LLM mechanistic baseline |
| `llm_belief_algorithmic_planner` | LLM continuous | 同一 algorithmic planner | LLM belief 本身是否更有用 |
| `heuristic_belief_llm_chooser` | 当前 heuristic continuous | LLM chooser | LLM planning 是否带来收益 |
| `llm_belief_llm_chooser` | LLM continuous | LLM chooser | 完整 framework |
| `llm_static_belief_llm_chooser` | 首轮 LLM snapshot 后冻结 | 同一 LLM chooser | continuous update 的贡献 |
| `no_belief_llm_chooser` | 不提供 belief | 同一 LLM chooser | 显式 belief 的贡献 |
| `oracle_belief_llm_chooser` | evaluator 注入 preference truth | 同一 LLM chooser | belief error 与 planning error 分解 |
| `shuffled_belief_llm_chooser` | 错配/置乱 belief | 同一 LLM chooser | chooser 是否因果使用 belief |

主表不必同时放下所有 variant。建议主表使用 direct、当前 algorithmic、LLM-belief+algorithmic、完整 framework；其余放 ablation 和 causal analysis。

### 9.6 公平性与 truth isolation

所有非 oracle variants 必须满足：

- 不向 LLM 提供 edge utility function、reservation value 或其数值变换；
- 不提供对手实现名称，例如 `Boulware2025`、`Linear2025`；
- prompt 不出现当前运行属于 oracle/shuffled 的提示；
- planner 可以看到己方 `CenterUFun` 计算出的 candidate utility，因为这是 center 的合法私有信息；
- evaluator truth 只能进入 episode 结束后的 calibration/ranking metrics；
- paired variants 使用同一 scenario、edge assignment、seed 和 candidate generator；
- 相同 LLM model、temperature、token limit、重试策略和调用预算。

必须同时记录每次调用的：

- 完整 request messages 与原始 response；
- model name、endpoint、temperature、seed、token limit；
- prompt/template hash 和代码 hash；
- parse/repair/fallback 结果；
- 输入 belief snapshot、candidate set 和最终 action；
- prompt/completion tokens、latency、错误信息。

这些记录应与当前 episode JSONL 分目录保存，并通过 episode/edge/turn ID 相互索引。不得只保存最终解析后的 JSON。

### 9.7 初始模型设置

第一轮使用已部署的 Qwen3-30B OpenAI-compatible 服务：

```text
base_url = http://127.0.0.1:8002/v1
model = 通过 /v1/models 查询并固定实际名称
temperature = 0.0 或 0.1
top_p = 1.0
```

建议 belief updater 与 chooser 分开限制输出长度，并使用 context-safe history compaction：

- belief updater：只携带该 edge 的 compact evidence 和上次 snapshot；
- chooser：只携带全局 agreement summary、各 edge belief 摘要和 top-K candidates；
- public raw history 单独落盘，不在每次调用中无限累积；
- 相同输入使用 hash cache，测试重跑时可选择 replay，避免不必要的模型调用。

初始实验不训练模型，先使用 prompting 验证接口和 causal behavior。只有在 belief accuracy/calibration 对最终收益确有预测力后，再从保存的 trajectory 构造 SFT 或 calibration training 数据。

### 9.8 分阶段实施与实验门槛

#### Phase A：offline belief replay

使用已经保存的 ANL trajectory，只调用 LLM belief updater，不让其控制 action。比较：

- pairwise preference accuracy；
- acceptance Brier/ECE；
- posterior 随证据增加是否改善；
- heuristic belief 与 LLM belief 的差异；
- 每次 update 的 token/latency。

这一阶段最便宜，也能在不改变 negotiation policy 的情况下判断 LLM belief 是否值得上线。

#### Phase B：online LLM belief + fixed algorithmic planner

仅替换 belief，保持 candidate generator、planner 和 acceptance policy完全相同。若 belief 指标提高而 utility 下降，优先修 planner interface/calibration，不进入 full LLM。

#### Phase C：LLM chooser

分别运行 heuristic-belief+LLM-chooser 与 LLM-belief+LLM-chooser，验证 chooser 的独立贡献，并加入 no-belief、static/frozen、shuffled 因果 variants。

#### Phase D：development main run

在 `dinners`、`job_hunt`、`target_quantity` 和独立生成的 additive/substitutable/target/complementary 场景上做至少 30 个 paired scenarios。除 utility 外必须报告 belief quality、action flip、invalid/fallback、调用成本和 latency。

#### Phase E：locked test

只有满足以下条件才冻结 prompt/code hash 并更换 seed：

1. LLM belief 相对 heuristic belief 在 held-out trajectory 上改善 ranking 或 calibration；
2. oracle/shuffled 对同一 planner 产生方向合理的差异；
3. 完整方法相对 direct LLM 和 algorithmic baseline 有稳定 paired utility improvement；
4. invalid output/fallback rate 足够低；
5. 改进不能只来自更多 token 或更多调用，必须提供 budget-matched comparison。

### 9.9 ANL2025 在论文中的定位

ANL2025 最适合支持以下论点：

> 在没有自然语言解析和生成干扰的结构化 negotiation protocol 中，显式、连续更新且带不确定性的 partner belief，能够被 LLM planner 用于跨多笔交易的全局决策；收益来自 belief-to-planning interface，而不仅是更长的 prompt 或合法 action formatting。

它不能单独支持“自然语言 negotiation 能力提升”。最终论文仍需另一个真正包含自然语言、隐藏偏好、策略性信息披露和多轮 message interaction 的 benchmark，与 ANL2025 的结构化因果实验形成互补。
