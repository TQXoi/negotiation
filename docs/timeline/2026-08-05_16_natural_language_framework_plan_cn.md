# 自然语言协商中的校准式 Belief-to-Action Planning：方法、Story 与实验计划 V2

> 日期：2026-08-05
> 本文是对 `OPENREVIEW_2026_NEGOTIATION_RELATED_WORK_AND_NOVELTY_STORY_CN.md` 的方法与 benchmark 修订。
> 核心变化：自然语言 hidden-preference negotiation 成为主实验；ANL 降为机制附录，不再承担论文主结论。
> 直接竞争方法：*Scaling Inference-Time Computation via Opponent Simulation*，arXiv:2602.19309v2。
> 作者公开材料：[论文](https://arxiv.org/html/2602.19309v2)、[prompt 与 example-output 仓库](https://github.com/llmnegotiationsubmission/llmnegotiationsubmission)。

## 1. 结论先行

用户对 ANL 的判断是正确的：ANL 2024/2025 的结构化 offer/utility protocol 很适合检查算法与 calibration，但不能单独支持“从自然语言互动中形成并使用 opponent belief”的主张。将 ANL 放在主实验位置，会让 reviewer 合理地追问：既然输入已经是结构化报价，为什么需要 LLM belief model？方法究竟是在理解语言，还是在拟合数值序列？

建议采用以下论文主线：

> **从黑盒对手模拟到可校准、可干预的 belief-to-action planning：agent 同时维护跨谈判的行为先验与单次谈判内逐轮更新的显式 posterior，并用 posterior uncertainty 决定何时获利、何时探索、何时安全成交。**

暂定方法名为 **Dual-Timescale Calibrated Belief-to-Action Planning（DC-BAP）**。名称只是工作代号，不再沿用与本 benchmark 无关的 `ASTRA` variant 命名。

最强证据组合应为：

1. **CaSiNo：主机制实验。** 多议题、自然语言、hidden priority、局内多轮更新，有真实 preference 可评价；
2. **NegotiationArena—Opponent Simulation protocol：直接竞争实验。** 完全采用其 43/63 repeated buyer–seller 与互补 resource exchange 配置，比较 direct、Opponent Simulation 和 DC-BAP；
3. **Simple Env：受控因果诊断。** 不作为“大众 benchmark”卖点，只用于 oracle/shuffle/freeze、non-stationarity 和 belief calibration ground truth；
4. **AgenticPay-expanded：推荐的外部泛化实验。** 选择 multi-product、multidimensional contract 或 sequential topology，避免又退化为单价格 buyer–seller；
5. **ANL：附录。** 用来证明 planner core 不依赖自然语言表面形式，而不是主 benchmark。

只跑 NegotiationArena 的单价格任务仍不够，因为它接近 Simple Env 的简化版；真正有用的是它作为与 Opponent Simulation **同场、同预算、同 repeated protocol 的直接 prior-art comparison**。CaSiNo 和 AgenticPay-expanded 承担“复杂语言与多议题通用性”的证据。

## 2. 旧 story 为什么不够

原报告已经正确指出不能再声称：

- 首次 opponent belief：MIND、GENEGO 已经显式推断 willingness/preferences；
- 首次 continuous adaptation：EvoTac 和 repeated-game 方法已经跨实例适应；
- 首次 belief-aware planning：Opponent Simulation 已经完成 belief formation、rollout、best response；
- 首次长程 negotiation planning：HAR 已有 agenda/tactic decomposition。

我们当前代码还存在一个更实际的问题：旧 NegotiationArena `BeliefPlannerAgent` 的 reservation interval 是 regex 加手写规则更新，数字 action 由启发式 planner 决定，LLM 只生成一句不改变行动的表述。这可以做可靠的 protocol sanity test，却不能证明“LLM 从自然语言维护 continuous belief”，也无法与 Opponent Simulation 的 inference-time method 公平比较。

因此 novelty 不能是模块清单“belief + planner + generator”，而必须是一个可证伪的科学命题：

> **黑盒 opponent-policy simulation 依赖足够多的 repeated episodes，且无法判断自己何时不确定；显式双时间尺度 posterior 能否在首次相遇、局内新证据、对手突变和复杂多议题语言中，以更少推理预算产生更高 decision value？**

## 3. 与 Opponent Simulation 的准确关系

### 3.1 对方方法做了什么

Opponent Simulation 将 repeated negotiation 看作 smooth fictitious play：

1. 读取已经完成的 episodes 与当前局部轨迹；
2. 用辅助 LLM 模仿对手的 time-averaged behavior policy；
3. acting model 每轮 brainstorm 五种策略/候选；
4. 对每个候选模拟完整未来轨迹；
5. 选择预测自身 reward 最大的候选。

作者仓库提供了 `strategic_brainstorming`、`reminder`、`opponent_model`、`evaluation_model`、`self_simulation` prompts 和每个 setting 的 10×20 示例输出，但没有公开完整的一键执行 harness。因此本地实现是 **paper-compatible reconstruction**，不能表述为“运行作者完整原代码”。

### 3.2 原论文设置

| 维度 | Buyer–Seller | Resource Exchange |
|---|---|---|
| 私有参数 | seller cost=43；buyer WTP=63 | RED `(25X,5Y)`，BLUE `(5X,25Y)` |
| 价值 | 价格决定 surplus | RED `(vX=.5,vY=2.5)`，BLUE `(vX=2.5,vY=.5)` |
| repeated protocol | 每 run 20 episodes | 每 run 20 episodes |
| 单局 horizon | 最多 10 turns | 最多 10 turns |
| random runs | 10 | 10 |
| focal 位置 | buyer/seller 均从第二手开始 | 分别报告 first/second |
| Best-of-N | N=5 | N=5 |

论文报告 Qwen3 + BoN-oppo 相对于该模型自己 direct baseline 的增益为：

| Setting | 论文增益 |
|---|---:|
| Buyer | `+10.04 ± 2.03` |
| Seller | `+18.54 ± 2.46` |
| Resource first | `+8.95 ± 6.23` |
| Resource second | `+29.65 ± 0.33` |

这些是 **delta，不是 absolute reward**。若本地用 Qwen3-30B 同时扮演 focal/opponent，只能称为 matched local comparison；论文默认 opponent/model snapshot 与本地不完全相同时，不能把本地 absolute value 直接与上表比较。严格复现需要匹配作者 acting model、Gemini opponent/simulator、完整 prompt 和 sampling config。

### 3.3 我们不能声称什么、可以声称什么

不能声称“首次用对手模型做 rollout”。可以检验并主张以下差异：

| 维度 | Opponent Simulation | DC-BAP |
|---|---|---|
| belief 表示 | 黑盒生成式、time-averaged behavior policy | typed posterior：preference/reservation/acceptance dynamics/type + uncertainty |
| 更新时间尺度 | 重点是跨完整 episodes | 跨 episode meta-prior + 每一对手 utterance 后的局内 posterior |
| 不确定性 | prompt 中乐观补全 | 显式区间/概率/confidence/evidence reliability/abstention |
| action search | 单个模拟器给未来轨迹与 reward | posterior hypotheses/ensemble 下的期望收益、downside risk、IG、future value、leakage |
| 适用条件 | repeated same opponent 最自然 | episode 1、新对手、局内变化、repeated opponent 均适用 |
| 可评价性 | simulator ranking accuracy | belief calibration + decision regret + causal intervention |
| 主要风险 | simulator hallucination、optimism bias、调用昂贵 | posterior 错配、LLM 数值不稳定；需 schema validator 与 action lock |

## 4. DC-BAP 方法设计

### 4.1 通用 latent state，而非 Env-specific JSON

对手隐变量写成：

\[
\theta = (u, r, \pi_{acc}, d, z)
\]

其中：

- `u`：issue preference/utility ranking；
- `r`：reservation value 或 acceptance threshold；
- `π_acc`：给定 offer、轮次和语境的接受概率；
- `d`：concession dynamics，包括方向、速度和 deadline sensitivity；
- `z`：较粗的行为类型混合，如 competitive/cooperative/fairness-seeking/adaptive。

环境不必拥有所有字段。没有证据时字段必须是 `null`/wide interval，而不是编造 `flexibility_delta`。语言证据存成 `(observation, hypothesis, reliability)`；数字让步与实际接受是强证据，修辞和角色名只是弱证据。

### 4.2 双时间尺度更新

第 `e` 个 episode 开始时，从之前公开互动形成 meta-prior：

\[
b^{meta}_e(\theta)=F(\tau_{1:e-1})
\]

局内每观察到对手新 utterance/action 后更新：

\[
b_{e,t}(\theta) \propto
p(o_{e,t}\mid\theta, h_{e,t-1})^{\rho_{e,t}}
b_{e,t-1}(\theta)
\]

`ρ` 是证据可靠性。新对手的 `b_meta` 可以是宽先验；因此方法不要求 repeated interaction。对手在 episode 11 突然更换策略时，引入 change-point probability，让近期强证据覆盖旧 meta-prior。

当前落地 V1 使用 LLM 每轮输出受 validator 约束的 typed JSON；下一版可把 posterior 改成粒子集合或参数分布。V1 已经与旧 heuristic agent 不同：LLM 同时参与 belief update、candidate generation 与 planning，所有原始请求/响应均落盘。

### 4.3 候选不是同质采样

每轮生成五类候选：

1. `exploit`：利用当前 posterior 的高概率假设；
2. `probe`：区分两个主要 posterior hypotheses；
3. `safe`：在不确定时保持高 agreement probability；
4. `reciprocal`：回应对方最近的合作/让步；
5. `fallback`：接近 deadline/BATNA 的稳健动作。

这保留了 Opponent Simulation strategic brainstorming 的多样性优势，但让探索动作有明确 epistemic 目的。

### 4.4 Belief-usable planner

对候选 `a` 的评分为：

\[
S(a)=
\mathbb{E}_{\theta\sim b}[P_{acc}(a,\theta)U_{self}(a)]
-\lambda\,Risk_{CVaR}(a;b)
+\beta_t IG(\theta;o'\mid a)
+\gamma V_{future}(a)
-\mu Leakage(a).
\]

- `expected payoff` 防止只追 agreement；
- `CVaR/downside risk` 防止相信一个过度乐观 simulator；
- `IG` 奖励能改变后续决策的探索，系数随 deadline 衰减；
- `future value` 同时覆盖本 episode 与 repeated game；
- `leakage` 防止为了探测而暴露自己的 reservation/preferences。

真正的 `belief-usable` 不是 prompt 里出现 belief 字段，而是更换 belief 会按预期改变候选排序。必须通过 oracle/frozen/shuffled/adversarial belief intervention 证明。

### 4.5 Generator 与 action lock

Planner 选择完整 protocol-valid candidate；generator 不得再次自由修改价格、issue allocation 或 ACCEPT/REJECT。validator 检查：

- XML/tag/JSON 格式；
- 资源与报价合法性；
- planner action 与 public realization 一致；
- 不泄露 private belief trace；
- parse failure 使用明确 fallback，并单独计入 invalid-action rate。

## 5. Benchmark 设计

### 5.1 CaSiNo：主机制与多议题语言证据

CaSiNo 应成为第一主表，因为它同时具备自然语言、多 issue、隐藏优先级和完整 dialogue。实验需要从“平均 reward”升级为：

- preference rank accuracy、top-issue accuracy；
- posterior NLL/Brier/ECE 与 credible interval coverage；
- 每轮 belief accuracy curve；
- self utility、partner utility、joint utility、agreement、Pareto distance；
- probe 后 posterior entropy 是否下降；
- belief intervention 是否导致合理 action flip。

需要防止 Env-specific：belief schema 和 planner objective 与 NegotiationArena 共用，只允许 environment adapter 提供 legal action、utility evaluator 和 parser。

### 5.2 NegotiationArena：直接 prior-art comparison

Buyer–Seller 的价值不在于复杂，而在于复刻直接竞争对手；Resource Exchange 更能测试 issue preference 与互惠交换。正式跑四个 cells：buyer、seller、resource first、resource second。

必须加入两个额外 robustness tracks：

1. **First-encounter track**：只看 episode 1–5，检验没有大量跨局历史时谁更有效；
2. **Opponent-switch track**：episode 11 更换 opponent policy/personality，检验 time-averaged simulator 与 change-aware posterior 的恢复速度。

不能用 switch track 的结果与原论文表格混在一起；它是我们为检验 scientific hypothesis 新增的 stress test。

### 5.3 Simple Env：只做可识别机制

Simple Env 保留的原因是可以穷举 utility/acceptance truth，适合报告：

- belief calibration 与 oracle regret；
- posterior recovery under controlled noise；
- static/continuous/frozen/shuffled/oracle belief；
- planner decision boundary；
- compute-free upper bound。

它不承担外部有效性，不应成为标题或摘要中唯一 benchmark。

### 5.4 AgenticPay-expanded：跨 topology 泛化

若资源允许，将其作为第三个语言环境，优先选择 multi-product、multidimensional contract、sequential/many-to-many。重点测试跨 thread opportunity cost 和 belief transfer。原论文 111-task setting 与当前扩展仓库必须分别标注 commit/config，不能混报。

### 5.5 ANL：附录定位

ANL 只回答“同一 planner 数学 core 是否也适用于结构化自动协商”。LLM-free variant 是 lower-cost control；LLM belief/planner variant 是 portability test。主文不再用 ANL 证明语言 opponent modeling。

## 6. Baseline 与消融矩阵

### 6.1 主 baseline

1. `Direct`：原 environment prompt；
2. `Structured CoT / reflection`：控制额外 reasoning；
3. `Best-of-5 evaluator`：控制 candidate sampling；
4. `Self-simulation`：没有 learned opponent policy；
5. `Opponent Simulation BoN=5`：直接竞争方法；
6. `Static belief + planner`：局内不更新；
7. `Continuous belief + direct chooser`：有 belief、没有明确 planner；
8. `DC-BAP`：完整方法。

### 6.2 因果消融

| Variant | 回答的问题 |
|---|---|
| no belief | planner 增益是否只是更长 prompt/更多计算 |
| frozen belief | 局内 continuous update 是否有用 |
| shuffled belief | planner 是否真的读取 belief |
| oracle belief | belief 错误造成的可恢复上限 |
| wrong/confident belief | uncertainty 与 risk term 是否防止灾难性行动 |
| no information gain | 主动 probe 是否贡献长期收益 |
| no risk/CVaR | 单一乐观 hypothesis 是否造成差成交 |
| no leakage penalty | 探索是否过度暴露自身信息 |
| meta-prior only | 跨局记忆和局内证据各自贡献 |
| within-episode only | repeated learning 的边际贡献 |

### 6.3 调用预算公平性

当前本地实现中：

- direct focal 每实际行动 1 个 model call；
- Opponent Simulation focal 每行动 3 calls：5 candidates、opponent model、future rollout/selection；
- DC-BAP focal 每行动 3 calls：belief update、5 candidates、belief-usable planner；
- 对手在三种方法中始终是同一 repeated direct/strategic-brainstorming policy。

正式表同时报告 calls、input/output tokens、wall time、reward-per-1K-tokens。还需增加 compute-matched direct/BoN baseline，排除“只是调用更多”的解释。当前 dependency-free client 已记录完整 request/response 和 wall time；若 endpoint 返回 usage，下一步应扩展 client 保存 token usage。

## 7. Research questions 与指标

### RQ1：belief 是否正确且校准？

- preference rank/top-item accuracy；
- reservation MAE；
- acceptance Brier/NLL/ECE；
- 50%/80%/90% interval coverage；
- confidence–accuracy correlation；
- evidence attribution precision。

### RQ2：正确 belief 是否产生 decision value？

- focal reward、agreement、joint value、Pareto distance；
- oracle action regret；
- belief accuracy 与 reward 的 conditional correlation；
- oracle/shuffle/freeze action flip 与 value gap。

### RQ3：与黑盒 opponent simulation 相比何时更强？

- episode-index learning curve；
- episode 1–5 与 16–20 reward；
- opponent switch 后恢复到 90% steady performance 所需 turns/episodes；
- performance per model call/token；
- simulator optimistic error 与 posterior calibration error。

### RQ4：是否 Env-agnostic？

固定 belief schema、planner equation 和 trace schema，只替换 parser/legal-action/utility adapter；在 CaSiNo、NegotiationArena 和 AgenticPay-expanded 复用同一 core。

## 8. 已实现的 NegotiationArena 对照代码

代码位置：`/work5/qixint/external_negotiation_envs/NegotiationArena`。

| 文件 | 作用 |
|---|---|
| `arena_integration/repeated_agents.py` | repeated direct、Opponent Simulation BoN-5、DC-BAP；完整模型调用与决策 trace |
| `arena_integration/repeated_games.py` | 43/63 buyer–seller、互补资源价值、first-mover 控制 |
| `arena_integration/run_opponent_simulation_setting.py` | 10×20×10 四 setting runner，支持 resume |
| `arena_integration/summarize_repeated_comparison.py` | paired run-level delta、bootstrap CI、paper delta reference |
| `scripts/run_opponent_simulation_qwen30b.sh` | Qwen3-30B 8002 一键 pilot/formal comparison |
| `tests/test_repeated_integration_unittest.py` | belief/planner/simulator 调用链和 resource payoff 测试 |

为了让原仓库 Resource Exchange 可运行，还做了两个最小上游兼容修复：

1. `games/trading_game/interface.py`：不存在的 `AgentMessageInterface` 改为当前 `AgentMessage`；
2. `games/trading_game/game.py`：避免父类初始化把已经创建的 parser 覆盖成 `None`。

六个离线测试已经通过。2026-08-05 检查时 `127.0.0.1:8002` 未监听，因此没有伪造在线数值；待服务恢复后运行以下命令。

## 9. 运行命令与实验阶段

### 9.1 服务与协议检查

```bash
curl http://127.0.0.1:8002/v1/models
cd /work5/qixint/external_negotiation_envs/NegotiationArena
/work5/qixint/miniconda3/envs/research/bin/python -m unittest discover \
  -s tests -p '*unittest.py' -v
```

### 9.2 Smoke：先验证 buyer 与 resource parser

```bash
cd /work5/qixint/external_negotiation_envs/NegotiationArena
SETTINGS='buyer resource_first' RUNS=1 EPISODES_PER_RUN=2 \
OUTPUT_ROOT=arena_runs/oppsim_protocol_smoke \
bash scripts/run_opponent_simulation_qwen30b.sh
```

这里的目标不是比较方法，只检查 0 error、trace 完整、reward 计算和上下文长度。

### 9.3 Pilot：估计方差与成本

```bash
SETTINGS='buyer seller resource_first resource_second' \
RUNS=3 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/oppsim_qwen30b_pilot \
bash scripts/run_opponent_simulation_qwen30b.sh
```

先检查：三种方法 parse error 是否同量级、实际 calls 是否符合 1/3/3、episode learning curve 是否存在、resource reward 是否合理。若 framework 失败来自格式而不是决策，先修 action validator，不要用 prompt 调参掩盖。

### 9.4 Formal：论文协议

```bash
SETTINGS='buyer seller resource_first resource_second' \
RUNS=10 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/oppsim_qwen30b_formal_n10x20 \
bash scripts/run_opponent_simulation_qwen30b.sh
```

每个 setting 的结果在 `comparison.json`。主统计单位应是 run，而不是把同一 run 中 20 个高度相关 episodes 当成 20 个独立样本。

### 9.5 严格 paper-opponent replication

本地默认是 Qwen×Qwen matched comparison。获得作者所用 Gemini-compatible endpoint 后，对每个 method 显式传：

```bash
python -m arena_integration.run_opponent_simulation_setting \
  --method opponent_simulation --setting buyer \
  --runs 10 --episodes-per-run 20 --max-turns 10 --candidate-count 5 \
  --model Qwen3-30B-A3B-Instruct-2507-base \
  --base-url http://127.0.0.1:8002/v1 \
  --opponent-model '作者对应的 Gemini snapshot' \
  --opponent-base-url '对应 OpenAI-compatible endpoint/v1' \
  --output-dir arena_runs/oppsim_strict/buyer/opponent_simulation --resume
```

只有在 snapshot、opponent、prompt、temperature 和采样都匹配时，才把本地 `delta over direct` 与论文 Qwen3 的四个 delta 作数值复现判断。

## 10. 预注册的成功/失败判据

不能以“framework 最终调到赢”为唯一标准。建议预注册：

1. CaSiNo 上 DC-BAP 相对 direct 和 static belief 同时提升 calibrated belief 与 reward，且 shuffled belief 显著下降；
2. NegotiationArena matched budget 下，DC-BAP 的 run-level reward delta 不劣于 Opponent Simulation，并在 episode 1–5 或 opponent-switch 至少一个预设区间显著更好；
3. 完整方法的收益不能完全由 parse-error 降低解释；另报告 valid-only performance；
4. belief confidence 与实际 accuracy 单调相关，90% interval coverage 不能严重失真；
5. 方法在至少两个自然语言、多议题环境中保持同方向效果；
6. 若只在固定 repeated opponent 的后十局有效，应诚实得出“meta-learning 有效、局内 belief 未证实”，而不是继续宣称 universal continuous belief。

## 11. 最终论文 Story

### 一句话

> **现有 negotiation agent 要么输出不可校准的对手描述，要么用黑盒模拟器在大量 repeated interaction 后搜索 best response；我们研究如何把跨局经验与局内自然语言证据统一成显式不确定 posterior，并把该 posterior 的准确性转化为可审计的 decision value。**

### 三项贡献

1. **方法**：一个 Env-agnostic 的 dual-timescale opponent posterior，逐 utterance 更新 preference/reservation/acceptance dynamics，并保留 calibration 与 change uncertainty；
2. **规划**：一个 posterior-conditioned risk/information-aware planner，显式权衡 payoff、agreement risk、information gain、future value 与 own-information leakage；
3. **评价**：一套从 belief accuracy 到 causal decision value 的实验协议，在 CaSiNo 与 paper-compatible NegotiationArena 中加入 oracle/frozen/shuffled 和 opponent-switch，并与同预算 Opponent Simulation 直接比较。

### Reviewer 最可能的质疑与回答

- **“这就是 Opponent Simulation 加 JSON。”** 回答必须来自校准指标、first-encounter/switch 优势、belief interventions，而不是架构图；
- **“提升只是 3 倍调用。”** 使用 1/3/3 调用记录、compute-matched BoN 和 reward/token；
- **“belief 是 post-hoc rationale。”** shuffled/oracle/wrong-belief intervention 必须改变 action 与 value；
- **“CaSiNo 太简单。”** 加入 repeated Resource Exchange 和 AgenticPay-expanded 多维合同；
- **“方法对 Env 特化。”** 三环境共用 schema/planner，仅 adapter 不同；
- **“LLM posterior 数字不可信。”** 报告 calibration、coverage、abstention，并将未来训练 target 设为 preference comparisons/acceptance likelihood，而不是模糊 `flexibility_delta`。

## 12. 下一步优先级

1. 恢复 8002 后跑 `1×2` smoke，检查真实 Qwen 的 JSON/XML adherence；
2. 补 action validator/repair 和 endpoint token-usage logging；
3. 跑四 setting `3×20` pilot，冻结 prompt 后再正式跑；
4. 在 NegotiationArena 加 `frozen/shuffled/no-belief/opponent-switch`，不要先围绕结果手调；
5. 将相同 typed schema 和 planner 接回 CaSiNo，重跑 existing direct/ASTRA/framework 基线；
6. 只有当 prompt-based posterior 的 calibration 与 causal usefulness 被验证后，再进入 belief calibration training；训练 label 应来自环境 truth、pairwise preference 和 accept/reject likelihood，不直接监督模糊 delta。
