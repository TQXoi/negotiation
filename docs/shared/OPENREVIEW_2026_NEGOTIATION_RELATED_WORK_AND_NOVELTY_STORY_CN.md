# 2026 OpenReview Negotiation 相关工作系统调研与 ASTRA Novelty / Benchmark 方案

> **2026-08-05 方法修订说明：** 本文的论文调研与 prior-art 判断仍有效；但“ANL 为机制层主实验”的 benchmark 层级已根据 natural-language research question 调整。新版方法、story、Opponent Simulation 对照实现与正式命令见同目录 `NATURAL_LANGUAGE_NEGOTIATION_BELIEF_FRAMEWORK_AND_EXPERIMENT_PLAN_V2_CN.md`。ANL 在新版中降为结构化机制附录，CaSiNo 与 paper-compatible NegotiationArena repeated protocol 成为主证据。

> 版本：2026-08-05
> 调研对象：用户给出的 8 个 OpenReview URL（实际对应 7 项独立工作，其中一项是同一论文的 conference 与 workshop 版本）。
> 研究目标：判断这些方法分别解决了 negotiation agent 的哪一环，哪些能够复现，哪些构成 ASTRA 的直接 prior art，并据此确定最有说服力的 benchmark 组合与论文 story。
> 状态说明：OpenReview 页面上的 “submission” 不等于主会接收。下文严格区分主会拒稿、workshop 接收/展示和 ACL ARR 在审。

## 1. Executive summary

这些论文共同说明，2026 年的 negotiation-agent 研究已经从“LLM 会不会讨价还价”转向四个更具体的问题：

1. **能否推断对方**：MIND 推断 willingness，GENEGO 推断 pairwise preference；
2. **能否积累经验**：EvoTac 从跨实例 terminal outcome 中提炼可复用 tacit heuristics；
3. **能否把推断转化为策略**：Opponent Simulation 用对手模拟和 Best-of-N 寻找 best response，HAR 用层级 agenda 维持长期目标并切换 tactic；
4. **结论能否跨环境成立**：AgenticPay 扩展市场结构，多语言工作证明仅在英语中评估会产生偏差。

这意味着 ASTRA 不能把 novelty 写成“首次使用 opponent belief”“首次 continuous update”或“首次 belief-aware planning”。这些说法分别会被 GENEGO/MIND、EvoTac、Opponent Simulation/HAR 直接挑战。

更稳健、也更符合我们已有系统的核心命题应当是：

> **现有 LLM negotiator 往往能产生关于对手的描述，却不能证明该描述是校准的，也不能稳定地把它转化为更好的交易。ASTRA 研究的是从 calibrated online belief 到 decision value 的完整因果链：在单次谈判中连续更新显式、带不确定性的对手 belief，并让 planner 以可审计的方式使用该 belief 来权衡 exploitation、information gathering、agreement risk 与 future value。**

最值得成为论文标题/主线的表述是：

> **From Belief Accuracy to Decision Value: Calibrated Online Opponent Modeling for Belief-Usable Negotiation Planning**
> 或中文：**从“猜对对方”到“用对猜测”：面向协商规划的校准式在线对手建模**。

实验上不应押注单一自然语言 benchmark，而应采用两层设计：

- **机制层主实验：ANL 2024/2025/2026**。2024 测 belief calibration，2025 测 belief 是否改善跨谈判 continuation planning，2026 测主动探索与自身信息泄露；
- **语言与市场泛化：AgenticPay 扩展版 + StrategicBench**。AgenticPay 选择 sequential/many-to-many/multi-product 或多维合同，而不是最简单 bilateral price；StrategicBench 用于长程自然语言、对手 personality 和 tactic revision。

CaSiNo、NegotiationArena、MIND/TravelPlanner 和 LLM-Deliberation 仍可保留，但应降级为诊断或 robustness，而不是支撑主 novelty 的唯一证据。

## 2. 论文清单、投稿状态与可复现性

用户给出的 8 个链接中，`pBRFXoYfhj` 与 `ywXgFYs44d` 是同一条研究线的 workshop 修订版与 ICLR conference 版本，因此科学内容应合并分析。

| OpenReview ID | 工作 | 截至 2026-08-05 的状态 | 环境/数据 | 公开代码与复现判断 |
|---|---|---|---|---|
| `pBRFXoYfhj` | Scaling Inference-Time Computation via Opponent Simulation | ICLR 2026 MALGAI workshop 版本 | NegotiationArena 的 repeated Buyer–Seller、Resource Exchange | [作者仓库](https://github.com/llmnegotiationsubmission/llmnegotiationsubmission)公开了 prompt/example outputs，但不是完整、成熟的一键 benchmark harness；可据论文重构 |
| `ywXgFYs44d` | Opponent Simulation as Inference-time Scaling for Self-improving Agent | ICLR 2026 conference rejected submission；上项的早期版本 | 同上 | 不应当作另一篇独立 baseline |
| `9DrWbVOc3h` | MIND | ICLR 2026 HCAIR workshop accepted | TravelPlanner + Stravl personas，多方旅行共识 | OpenReview 有 supplement，未找到独立公开 GitHub；可复刻 setting，但需自行实现 |
| `p144zx4bO0` | Hierarchical Agenda Reasoning (HAR) | ICLR 2026 SPOT/DATA-FM 等 workshop 版本 | 新建 StrategicBench，30 个长程 negotiation tasks | 未找到公开代码；任务和 evaluator 若未放出，严格数值复现暂不可行 |
| `3djsIClhS0` | AgenticPay | ICLR 2026 AIMS workshop | buyer–seller，多市场 topology，私有估值 | [完整仓库](https://github.com/SafeRL-Lab/AgenticPay)可用；是本组工作中最适合直接接入的语言环境 |
| `OUXeGofhKR` | GENEGO | ICML 2026 uncertainty/agentic systems workshop poster | 论文摘要只明确称 multi-issue negotiation benchmark | 未发现公开代码或 supplement；公开信息不足以确定精确任务实现，不能虚构环境名称 |
| `2cPztydemw` | EvoTac | ACL ARR 2026 May 在审；另有 ICLR 2026 Lifelong Agent workshop 版本 | 私有品牌—达人佣金率预测，跨实例 chronological stream | 数据来自商业平台且未公开，未找到代码；只能抄报论文数据或实现“同思想”消融，不能声称原样复现 |
| `UTwhu4udAg` | The Language of Bargaining | ACL ARR 2026 January / arXiv 标为 under review | NegotiationArena：Ultimatum、Buy–Sell、Resource Exchange，多语言 | 依赖公开 NegotiationArena，未找到作者独立代码；实验较易重构 |

这里有两个必须写入论文 related work 的事实：

- `ywXgFYs44d` 的 metadata 是 ICLR conference rejected submission，而 `pBRFXoYfhj` 是后续 workshop 版本；不能写成“两篇 ICLR 2026 accepted paper”。
- EvoTac 与 The Language of Bargaining 当前是 ARR/under review，不能写成 ACL 2026 accepted paper。

## 3. Scaling Inference-Time Computation via Opponent Simulation

论文与版本：[arXiv:2602.19309](https://arxiv.org/abs/2602.19309)、[OpenReview workshop 页面](https://openreview.net/forum?id=pBRFXoYfhj)、[代码/样例仓库](https://github.com/llmnegotiationsubmission/llmnegotiationsubmission)。

### 3.1 核心方法

该工作把 smooth Fictitious Play 的两步放进 LLM inference-time computation：

1. **belief formation**：辅助 LLM 读取过去完整 episodes，in-context 模仿对手的 time-averaged policy；
2. **best response**：acting LLM 生成若干候选策略/动作，让 opponent model 与候选进行未来 trajectory simulation，再选取收益最高的候选，即 BoN-oppo。

它不更新模型参数；适应发生在 repeated interaction 的上下文中。论文还加入结构化 strategic brainstorming，以提高候选策略的语义和数值多样性。典型候选包括高锚定、公平分割、tit-for-tat 等，而不是简单对同一 prompt 独立采样。

这项工作是本列表中与 ASTRA **最直接的 prior art**，因为它已经明确提出了 “belief formation → rollout → best response”。因此 ASTRA 不能只凭“我们也有 belief + planner”主张新颖性。

### 3.2 Setting

实验使用 NegotiationArena 的两种 repeated games：

- Buyer–Seller：buyer/seller 的保留边界分别约为 63 与 43，刻意不用 40/60 这类 5 的倍数；
- Resource Exchange：双方资源禀赋与价值互补；一方多 X 少 Y，另一方相反；
- 默认每个重复博弈 20 episodes，每 episode 最多 10 steps，10 个 random runs；
- acting agent 被放在不利的先手或后手位置；
- 默认 Gemini-2.5-Flash，另测 Claude Sonnet 4、Qwen3、Llama-3.3；opponent simulator 有时使用更轻的 Flash-Lite 以减少 self-modeling bias；
- Best-of-N 的主设置为 N=5。

### 3.3 Baselines 与结果

主要 baseline 包括 zero-shot、thinking、BoN-evaluator、self-simulation、iid BoN-oppo、AI Feedback、experience reflection、private-information prediction。论文报告的是相对于各模型自身 direct baseline 的收益提升：

| Acting model + BoN-oppo | Buyer | Seller | Resource first | Resource second |
|---|---:|---:|---:|---:|
| Claude Sonnet 4 | +3.02 ± 1.51 | +2.80 ± 2.06 | +30.65 ± 6.34 | +1.92 ± 0.68 |
| Qwen3 | +10.04 ± 2.03 | +18.54 ± 2.46 | +8.95 ± 6.23 | +29.65 ± 0.33 |
| Llama-3.3 | +4.80 ± 1.68 | +14.74 ± 3.13 | +13.27 ± 4.84 | +6.08 ± 5.83 |

论文的 mechanism evidence 包括：随 repeated episodes 增加，simulator 相对 oracle 的候选排序准确性提升；brainstorming 增加候选多样性；增加 N 通常进一步提升结果。

### 3.4 与 ASTRA 的重叠和边界

重叠非常实质：两者都希望根据对手模型选择动作，而不是只把 belief 当作解释文本。

可 defensibly 区分的边界是：

- 它学习的是**跨完整 episodes 的 time-averaged behavior policy**；ASTRA 应强调**单次谈判每一 turn 的在线 posterior update**，并在新对手/无 repeated games 时工作；
- 它的 opponent model 是黑盒生成式 simulator，没有显式、可打分的 preference/reservation/acceptance posterior；ASTRA 应提供 calibration、uncertainty 和 ground-truth diagnostics；
- 它以 rollout score 选 best response；ASTRA planner 应显式分解 exploitation、agreement risk、future value 与 information gain，并能在 belief intervention 后呈现可预测的 action change；
- 它只在两个简单 NegotiationArena games 中验证；ASTRA 必须在 multi-issue、long-horizon、linked negotiations 或 richer market 中证明通用性。

这不是说 ASTRA 天然已经优于它。当前 ASTRA 如果仍主要依赖 `flexibility_delta` 等非校准 JSON 和 prompt chooser，就还没有形成上述差异。Opponent Simulation 应列入正式 baseline，而不是只在 related work 中提及。

## 4. MIND: Multi-Agent Inference for Negotiation Dialogue in Travel Planning

论文：[arXiv:2603.21696](https://arxiv.org/abs/2603.21696)、[OpenReview](https://openreview.net/forum?id=9DrWbVOc3h)。arXiv 明确标注 accepted at ICLR 2026 HCAIR workshop。

### 4.1 核心方法

MIND 研究多人旅行规划中，如何让 agent 表达并推断每项约束的 willingness `w`：

- 每个 agent 只知道自己的 `w`；
- agent 根据 `w` 动态调整语言 tone；
- 其他 agent 从语言线索推断 `w`；
- Strategic Appraisal 根据推断选择 Push、Compromise 或 Yield；
- 多数投票最多进行 3 轮；如果仍未决，fallback 选择 ground-truth `w` 最高者的意见。

### 4.2 Environment 与数据构造

- 从 TravelPlanner 场景出发，引入 Stravl 风格 persona；
- 每个 scenario 从约 200–400 candidate personas 中以 MMR 选择多样组合；
- LLM 根据 20 个 survey questions 合成 persona responses，再以 MoSCoW 风格推导 1–10 的 willingness；
- 每组有共同 hard constraints，并至少有 3 个相互冲突、`w=6–8` 的 soft constraints；
- GPT-4.1-mini-2025-04-14，temperature 0.4；
- 201 个 scenarios，2–4 agents。

### 4.3 结果

| Method | Debate Hit Rate | Debate Ratio | Jain fairness | Total fidelity | Weighted satisfaction |
|---|---:|---:|---:|---:|---:|
| Base MAD | 26.51 | 82.71 | **0.6849** | **25.80** | 18.03 |
| MIND | **34.65** | **93.18** | 0.6838 | 23.87 | **19.96** |

随着人数增加，MIND 的 debate ratio 优势更明显：2/3/4 agents 下 Base 为 89.2/82.7/64.5，MIND 为 96.1/93.2/88.4。对 willingness 的推断报告 MAE 1.27、相关系数 0.69、误差不超过 1 的准确率 67.7%、不超过 2 的准确率 90.2%。LLM judge 对 rationality、fluency、overall 的 win rates 分别为 68.8%、72.4%、68.3%。

### 4.4 对 ASTRA 的启示与局限

MIND 证明了多人语言协调中显式 hidden intent 有用，也提供一个自然的 willingness calibration auxiliary task。但其 90.2% 不能直接理解为自然对手建模能力：系统主动把 `w` 编码进 tone，推断器很大程度是在解码设计好的 channel；当多数投票失败时，fallback 又直接读取 ground-truth highest `w`，部分绕开了 interaction。

因此它适合 ASTRA 的 auxiliary test：去掉 ground-truth fallback、tone encoding 或 role label clue，测试 continuous belief 是否仍能从真实让步和理由中恢复 willingness。不适合作为唯一主 benchmark。

## 5. Hierarchical Agenda Reasoning 与 StrategicBench

论文：[OpenReview SPOT 版本](https://openreview.net/forum?id=p144zx4bO0)；另一个公开 workshop 页面也描述了同一方法及 StrategicBench：[DATA-FM 页面](https://openreview.net/forum?id=3OJJcwvWk0)。

### 5.1 StrategicBench

作者认为传统 buyer–seller 或 resource division 过于短和低维，因此构建了 30 个双边、长程、现实语境 negotiation tasks：

- 题目灵感来自 Harvard Program on Negotiation 材料，覆盖艺术、企业、政策等领域；
- 每个角色有 public instructions 与 confidential instructions；
- reward 由明确的 numeric/boolean objective questions 构成，并可计算 agreement、individual reward、joint value/Pareto 条件；
- 涵盖 BATNA、anchoring、relationship、authority 等 negotiation skills；
- 任务由作者选定概念后再用 `gpt-5-chat-latest` 辅助生成/改写，因此仍需关注生成偏差与 contamination。

基础比较中，instruction model `gpt-4o-mini` 与 reasoning model `o4-mini` 的 agreement rate 分别约为 88.6% 与 96.8%，双方 normalized reward 约从 0.47/0.49 提升到 0.63/0.66。更重要的失败是：即使 reasoning model 总体更强，也不能稳定根据 sycophantic、balanced、anticompetitive、competitive、eager、receptive 等 opponent personality 改变策略。

### 5.2 HAR 方法

HAR 将策略对话分成两层：

- **what to achieve**：持久的 goal/agenda，跨轮保持；
- **how to act**：可失败、可回退和替换的 strategy/tactic。

实现上使用层级 MDP 风格的三层 reasoning，由 LLM 完成 state abstraction/history summarization、高层 policy selection 和低层 language instantiation。失败时可以回退低层 tactic，而不放弃高层目标。作者还在 Qwen3-4B 上做 multi-turn RL：在 SolWave Acquisition 训练，再测试未见 opponent personalities 和两个 held-out tasks。

主实验选 5 个代表任务，与 history-conditioned、summary-conditioned、ReAct、instruction/reasoning models 比较。论文称 HAR 总体匹配或超过强 prompting baseline，提高 agreement、joint value 与人类评价；在两个 personality adaptation 任务上最高提升约 38%。另有 36 名人类参与者参与评价。

### 5.3 与 ASTRA 的关系

HAR 主要解决“如何维护长期目标和切换 tactic”，没有以显式 calibrated opponent posterior 为中心。它与 ASTRA planner 是互补但有竞争的：HAR 可以作为一个强的 belief-free hierarchical planner baseline；ASTRA 也可以把 HAR 的 agenda/tactic decomposition 作为 planner 外壳，在每一层把 belief posterior 变成决策输入。

StrategicBench 对我们的价值很高，因为它比 NegotiationArena 更能暴露 planner 是否长期一致、是否根据对方 personality 修正行动。但在 public code/task package 未确认可获取前，不能把它设为唯一必跑 benchmark。若自行重建任务，必须明确写 “StrategicBench-inspired reimplementation”，不能报告为原 benchmark 的精确复现。

## 6. AgenticPay

论文：[arXiv:2602.06008](https://arxiv.org/abs/2602.06008)、[OpenReview](https://openreview.net/forum?id=3djsIClhS0)、[官方代码](https://github.com/SafeRL-Lab/AgenticPay)。

### 6.1 原论文 setting

AgenticPay 把自然语言 bargaining 扩展到多种市场 topology：

- buyer 有私有 willingness-to-pay，seller 有私有 minimum acceptable price；
- public side 包含商品与市场语境；
- agent 用语言轮流交互，框架抽取结构化报价；
- 只有双方在 bargaining zone 内对同一价格达成一致才算 deal；
- 支持 bilateral、one-to-many、many-to-one、many-to-many，以及 sequential/parallel interactions；
- 论文数据含 111 tasks：31 basic + 80 realistic，8 种 topology、10 类生活/服务/采购/金融资产场景；
- 每局最多 20 轮，temperature 0、seed 0；论文每个 task/model 只跑一次，这是其统计可信度的明显短板。

Global Score 综合 deal feasibility、surplus balance、quality 和 efficiency，并以 `Q = 4 r_b r_s` 强调双方同时取得 surplus；另报告 buyer/seller score、deal、timeout、rounds 等。

### 6.2 原论文结果

| Model | Global | Seller | Buyer | Deal rate | Timeout | Avg. rounds |
|---|---:|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | **86.9** | 76.1 | **63.5** | 100 | 0 | 3.7 |
| Gemini 3 Flash | 82.2 | 73.3 | 61.1 | 100 | 0 | 4.8 |
| GPT-5.2 | 81.7 | **81.1** | 58.5 | 100 | 0 | 3.8 |
| Qwen3-14B | 63.9 | 58.9 | 47.6 | 79.3 | 20.7 | 7.8 |
| Llama-3.1-8B | 32.5 | 26.3 | 25.2 | 51.4 | 48.6 | 15.0 |

论文还观察到 seller advantage、near-miss、长程收敛失败；更多参与者有时因 liquidity 反而提高 score。

### 6.3 代码版本漂移

当前官方仓库 README 已扩展到约 160 个 multimodal/multidimensional contract tasks（4 scenarios × 8 topologies × 5 tasks），包含 JSON contracts、images、user profiles，并支持 OpenAI/vLLM/SGLang。它比论文的 111 个 price-focused tasks 更适合测试通用 planner，但实验必须二选一并标清：

- **paper reproduction track**：冻结论文对应 commit/config，复现 111 tasks；
- **expanded framework track**：使用当前仓库 richer tasks，称作 AgenticPay-expanded，不拿数字直接对照论文 Table 1。

### 6.4 对 ASTRA 的价值

AgenticPay 是最容易立即接入且最适合展示 natural-language market generalization 的环境。但 bilateral single-product price setting 与我们的 Simple Env/NegotiationArena 差别不够大。主实验应选：

- multi-product 或 multidimensional contracts；
- sequential/many-to-many topology；
- hidden constraints 与跨线程 opportunity cost；
- 至少 5–10 seeds，而不是每 task 一次；
- partner pool 包含 rule agents、Qwen、闭源强模型和 adversarial/non-stationary variants。

这样 continuous belief 才有多次更新空间，planner 也必须在多个可交换属性与未来交易之间做选择。

## 7. GENEGO

论文：[OpenReview](https://openreview.net/forum?id=OUXeGofhKR)。该页面显示为 ICML 2026 uncertainty in agentic systems 相关 workshop poster。

### 7.1 方法

GENEGO 的结构与我们高度相关：

- ledger 记录所有提出和同意过的 offers；
- partner model 让 LLM 做 offer/item 的 pairwise A/B preference comparison；
- 用 Bradley–Terry 模型把成对比较聚合为全局 preference ranking 与 confidence；
- negotiation policy 使用 disciplined tit-for-tat，同时不低于 walk-away threshold。

它的重要点是：没有直接让 LLM 输出一个任意的绝对 JSON score，而是把较可靠的相对判断聚合成统计模型。这对我们此前担心的 `flexibility_delta` SFT 很有启发：训练/评估 target 应优先使用相对偏好、约束和概率，而不是把模糊 delta 当作绝对真值。

### 7.2 结果与可复现性

公开摘要报告：

- 强 prompt baseline：19.1 average points/battle；
- GENEGO：22.0，paired `p < .001`；
- 完整 preference order 恢复准确率 82%，top item 97%；
- bootstrap confidence 与准确率有明显对应：low-confidence bin 约 14%，high-confidence bin 约 94%。

目前未找到公开代码。公开页面只称环境为 multi-issue negotiation benchmark，尚不足以可靠确定原始 scenario、协议和 scoring；因此本文不猜测其具体 benchmark。要做 exact reproduction，必须获得 PDF attachment/supplement 或作者代码。

### 7.3 对 ASTRA 的挑战

GENEGO 会直接挑战 “我们首次做可用的 preference belief”。我们的优势必须建立在更完整的 belief 与更强的 planner 上：

- 不只 preference rank，还包括 reservation、acceptance/concession dynamics 与 uncertainty；
- 不只 tit-for-tat，而是对候选 deal 进行 counterfactual utility、acceptance、future value 与 information value 分析；
- 做 oracle/shuffle/freeze intervention，证明 planner 对 belief 有因果敏感性；
- 在有 ground truth 的 ANL/AgenticPay 上同时报告 calibration 和 decision regret。

GENEGO 的 pairwise + Bradley–Terry 应当成为 ASTRA belief estimator 的正式 baseline，甚至可成为 posterior construction 的一个组件。

## 8. EvoTac

论文：[OpenReview ARR 页面](https://openreview.net/forum?id=2cPztydemw)；另有 ICLR 2026 Lifelong Agent workshop 版本。它不是传统 live multi-turn bargaining benchmark。

### 8.1 问题与方法

任务是从品牌—达人匹配前的 profile/platform context 预测最终佣金率。agent 不观察谈判中的 intermediate offers，只在实例结束后看到 terminal outcome `y`。因此它更接近 sequential outcome prediction / commercial forecasting，而非 agent 现场发言、报价和接受的闭环 negotiation。

EvoTac 不 fine-tune base model，而采用 predict–reflect–update：

1. 从分层记忆检索相关经验并预测 final commission；
2. 收到 terminal outcome 后诊断预测偏差；
3. memory manager 创建、增强、改写、合并或淘汰 heuristic。

记忆分为：

- `M_self`：自身约束、guardrails、templates；
- `M_beh`：观察到的 opponent behavior 与漂移；
- `M_strat`：对手 type/stance hypothesis。

这是**跨 episode 的 lifelong memory**，不是 single negotiation 内的 continuous belief update。

### 8.2 数据与结果

- 某大型互联网营销平台 2024 年数据；
- 500 个 human-audited matches，按时间 70/30 划分：350 offline，150 online stream；
- DeepSeek-v3.1，不微调；10 次 runs；
- 目标是 final commission percentage points。

| Method | MAE ↓ | Success@0.5 ↑ |
|---|---:|---:|
| XGB-Feat | 0.6002 | 0.52 |
| simplified MARL/Q-learning | 1.0734 | 0.20 |
| LLM-MonoMem | 0.5796 | 0.56 |
| LLM-LongCtx | 0.5986 | 0.49 |
| LLM-TabRAG | 0.5895 | 0.62 |
| EvoTac offline/frozen | 0.4468 | 0.68 |
| EvoTac online | **0.4151** | **0.70** |

online 版本还报告 RMSE 0.5084、MAPE 0.0684、`R²=0.8846`。

### 8.3 与 ASTRA 的关系

EvoTac 的贡献在“从 terminal feedback 形成跨实例可复用 heuristic”，ASTRA 的核心在“单次 interaction 内更新对手 posterior 并改变 action”。两者可以形成双时间尺度：

\[
\text{within episode: } b_t = U(b_{t-1}, o_t),
\qquad
\text{across episodes: } M_{k+1}=R(M_k, \tau_k, y_k).
\]

这可以成为 future extension，但不应混淆。由于原数据私有，当前只适合在 related work 中抄报原数据，或在公开 benchmark 上实现 “EvoTac-style cross-episode memory” baseline，不能声称复现其商业结果。

## 9. The Language of Bargaining

论文：[arXiv:2601.04387](https://arxiv.org/abs/2601.04387)、[OpenReview](https://openreview.net/forum?id=UTwhu4udAg)。arXiv 当前明确标注 Under Review。

### 9.1 Setting

该工作在 NegotiationArena 中保持 game rules、model parameters 和 incentives 不变，只改变语言：

- Ultimatum；
- Buyer–Seller；
- Resource Exchange；
- GPT-4o、GPT-3.5-Turbo、Claude-3-Haiku、Claude-3.5-Haiku；
- ordered cross-play，temperature 0.7，每条件 30 runs。

需要注意版本差异：早期 arXiv PDF 的完整实验是 English + Hindi/Gujarati/Punjabi，共 4320 games；2026-07 的 arXiv/OpenReview 摘要又加入 Marwadi。引用精确数字时必须注明所用版本，不应把 v1 表格和 v2 的语言数混写。

### 9.2 主要发现

早期完整表格中：

- Ultimatum acceptance：English 93.06%，Gujarati 84.44%，Hindi 88.33%，Punjabi 86.67%；
- Buyer–Seller agreement 约 97–100%，多重比较修正后 surplus shift 不显著，但 Punjabi 往往轮数更长；
- Resource Exchange trade volume：English 16.05，Gujarati 18.77，Hindi 18.70，Punjabi 18.59，修正后 `p=0.0001`，而最终 payoff 大体稳定。

总体结论是语言影响具有 task dependence：在 distributive games 中可能降低稳定性，在 integrative games 中可能增加探索；某些影响比替换模型更大。

### 9.3 对 ASTRA 的意义

它不提供新的 opponent-model planner，但给出一个很重要的 robustness 要求：belief updater 不能把英语语言风格误当成偏好变化，也不能在翻译后 calibration 崩溃。

NegotiationArena 本身过于简单，不应重新升级为 ASTRA 主 benchmark。但可以使用它的多语言 protocol 做低成本 stress test：

- 同一 latent utility、同一 opponent policy，仅改语言；
- 比较 belief posterior、ECE/Brier、action 和 payoff 是否保持稳定；
- 单独报告 language-induced belief drift；
- 增加中文与 English，不必完全重复所有 Indic languages。

## 10. 七项工作的统一比较

| 方法 | belief 对象 | 更新时间尺度 | belief 是否可校准 | planning/acting | 最强价值 | 主要不足 |
|---|---|---|---|---|---|---|
| Opponent Simulation | 黑盒 time-averaged opponent policy | 跨 repeated episodes | 不直接可校准 | simulated BoN best response | 最接近 belief→action 闭环 | 简单 games；依赖重复对手；latent preference 不透明 |
| MIND | willingness `w` | negotiation turn | 可按 MAE/相关性评估 | Push/Compromise/Yield | 多方隐含意图 | `w` 被主动编码进 tone；fallback 读取 truth |
| HAR | 主要是 dialogue state/agenda，不是 opponent posterior | 每 turn | 否 | hierarchical goal→tactic | 长程一致性与 tactic revision | belief modeling 不是核心；代码未公开 |
| AgenticPay | benchmark 自身不规定 belief | 由参赛方法决定 | 有私有 valuation truth | 自由语言 negotiation | 多市场 topology、公开代码 | 论文任务偏 price；每 task 一次；版本漂移 |
| GENEGO | pairwise preference + BT ranking/confidence | offer history | 部分可校准 | tit-for-tat + threshold | 相对判断比模糊绝对 score 更可靠 | planner 较简单；无公开代码/setting 不完整 |
| EvoTac | opponent heuristics/memory | terminal outcome、跨实例 | 用最终预测误差评估 | 预测，不参与 live bargaining | lifelong adaptation | 私有数据；不是闭环谈判 |
| Language of Bargaining | 不建 belief | 不适用 | 不适用 | direct agents | 揭示语言 confound | game 简单；是 stress test 而非 method baseline |

从这个比较看，ASTRA 最有意义的空位不是“再做一个 belief prompt”，而是同时满足：

1. within-negotiation continuous update；
2. explicit probabilistic/uncertain belief；
3. belief calibration 可由 ground truth 打分；
4. planner 对 belief 的使用可被 intervention 验证；
5. 在 structure-rich 与 language-rich environments 间复用同一接口。

## 11. 与我们工作直接相关的额外证据：Counterparty Modeling is Not Strategy

虽然不在用户给出的 8 个 URL 中，[Counterparty Modeling is Not Strategy: The Limits of LLM Negotiators](https://arxiv.org/abs/2605.16575) 几乎直接替 ASTRA 提出了研究问题：作者在 controlled multi-attribute bargaining 中发现，LLM 能较早、较准确地描述对方偏好，却不能稳定把这些信息转化为自身收益或 Pareto efficiency；agent 经常照顾对手高价值属性，却没有同时换回自己的高价值属性，最后 deal 仍更多受 opening anchor 支配。即使 prompt 要求明确写 concession-for-reciprocity，turn 看起来更策略化，最终效率也没有可靠提升。

这篇工作非常适合被放在 introduction：

> **Opponent modeling is not automatically a strategy. The missing layer is a planner whose decisions are causally grounded in calibrated beliefs.**

但它也提高了我们的证明门槛：只展示漂亮的 belief trace、planner rationale 或更高总分不够，必须做 “belief 是否改变 action、是否降低相对 oracle 的 decision regret” 的因果实验。

## 12. 建议的论文 story 与 novelty 边界

### 12.1 一句话问题定义

> 在私有、多议题、长程谈判中，如何把随交互连续更新且带不确定性的对手 belief，转化为可验证的报价、接受、让步和探索决策？

### 12.2 方法主线

统一接口可以写为：

\[
b_t(\theta)=U(b_{t-1}(\theta), o_t, e_t),
\]

其中 `θ` 包括对手 preference weights/ranking、reservation/constraints、acceptance/concession behavior，`e_t` 是可追溯 evidence。planner 对候选动作 `a` 计算：

\[
a_t=\arg\max_a
\mathbb E_{\theta\sim b_t}
\left[
V_{self}(a,\theta)
+\lambda V_{future}(a,\theta)
-\gamma R_{disagreement}(a,\theta)
\right]
+\beta\,IG(a;b_t)
-\eta\,Leakage(a).
\]

然后 generator 只负责把锁定的结构化 action/deal 语言化，不能偷偷改价、接受或 issue allocation。

这段公式不是为了装饰。每一项都必须在 code trace 中输出 value breakdown，并通过 ablation 对应到指标，否则仍会被审稿人视为 prompt engineering。

### 12.3 可以主张的 novelty

如果实现和实验补齐，可主张：

1. **Calibrated continuous belief state**：统一表示 preference、reservation、acceptance 与 uncertainty，并在单次谈判每轮更新，支持结构化和语言环境；
2. **Belief-usable planning**：不是把 belief 拼进 prompt，而是以 posterior expectation、risk、future value 和 information gain比较候选 action；
3. **Causal belief-to-action audit**：oracle/shuffled/frozen/miscalibrated interventions 证明 action 与收益变化确由 belief 带来；
4. **Cross-environment generality**：同一 belief/planner interface 在 ANL 和 AgenticPay/StrategicBench 工作，而不是为 CaSiNo 写死字段。

### 12.4 不应主张的 novelty

- 首次 opponent modeling：MIND、GENEGO 和大量经典 automated negotiation 都已做；
- 首次从经验连续改进：EvoTac 与 Opponent Simulation 都有 online/cross-episode adaptation；
- 首次使用 belief 规划：Opponent Simulation 已直接做 simulation-based best response；
- 首次 hierarchical planning：HAR 已明确提出 agenda/tactic hierarchy；
- 在 NegotiationArena 得到更高 buyer payoff 就证明通用能力：简单 buyer–seller 很容易被 opening offer、格式控制或 rule acceptance policy 解决。

## 13. Benchmark 选择：一套能回答审稿人问题的组合

### 13.1 Tier 1：结构化、可做因果诊断的主实验

#### ANL 2024：belief calibration

用途：真实 opponent reservation value 已知于 evaluator、未知于 agent，可以精确测量 RV posterior。

回答：belief 是否随 offer history 收敛？置信度是否校准？早期误判是否被纠正？

核心指标：RV MAE/NLL、credible interval coverage、ECE、convergence speed、utility/Nash optimality、belief error 与 regret 相关性。

#### ANL 2025：belief-usable long-horizon planner

用途：多个 sequential bilateral negotiations 的最终 utility 联动，局部最优 deal 不一定全局最优。

回答：planner 是否把对当前对手的 belief 变成 continuation value、线程优先级和接受阈值？

核心指标：center utility、agreement pattern、regret vs oracle planner、belief intervention action-flip rate、跨 thread opportunity cost。

#### ANL 2026：exploration、concealment 与 second-order belief

用途：对手完整 utility/reservation 未知，同时自己也应隐藏偏好。

回答：agent 是否会为 information value 试探？是否能控制自身信息泄露？

核心指标：preference/rank recovery、IG、own utility/Pareto、opponent inference leakage、exploration–exploitation curve。

ANL 2026 赛事结果仍在演化，应把它称为 protocol-based experimental track，而不是拿未冻结榜单作最终比较。

### 13.2 Tier 2：语言与市场泛化

#### AgenticPay-expanded

选择 multi-product/multidimensional contract + sequential/many-to-many 子集。它验证语言 parsing、私有 constraints、多对手与 market topology，同时有公开代码。

不建议用最简单的 bilateral single-price 作为主结果；它与 Simple Env/NegotiationArena 过于相似。

#### StrategicBench

在任务/评估公开后使用。它验证长程 agenda、自然语言理由、对手 personality 与 tactic revision。把 HAR 当 belief-free planner baseline，测试 ASTRA、HAR、ASTRA+HAR 三者。

### 13.3 Tier 3：diagnostic/robustness/appendix

| 环境 | 建议角色 |
|---|---|
| CaSiNo | backward compatibility；证明已有结果不会因通用化而消失 |
| NegotiationArena repeated | 精确复现 Opponent Simulation；不作为主通用性证明 |
| NegotiationArena multilingual | language robustness 与 belief drift stress test |
| MIND/TravelPlanner | 多方 willingness inference 辅助实验；必须去掉 truth fallback 做严格版本 |
| LLM-Deliberation | negative control/appendix；role-name leakage 与 proposer-centric 问题使其不适合作主证据 |
| EvoTac commercial task | 只抄报原数据；可在公开环境加入 EvoTac-style cross-episode memory ablation |

## 14. Baseline 与 ablation 矩阵

为避免审稿人说性能来自更多 token、结构化输出或 stronger prompt，建议固定 acting model、token budget、candidate count 和对手池，至少比较：

### 14.1 通用 prompting baselines

1. Direct/base prompt；
2. Chain-of-thought / reasoning prompt；
3. history summarization；
4. ReAct 或 self-reflection；
5. Best-of-N + generic evaluator。

### 14.2 论文方法 baselines

1. Opponent Simulation / BoN-oppo；
2. GENEGO-style pairwise + Bradley–Terry preference inference；
3. HAR-like agenda/tactic planner；
4. EvoTac-style cross-episode memory（只在 repeated/streaming track）；
5. environment-native official baselines：ANL top agents、AgenticPay original agents。

### 14.3 ASTRA 内部消融

| Variant | 回答的问题 |
|---|---|
| static snapshot belief + direct action | 一个初始化 belief 是否已足够？ |
| continuous belief + direct/myopic action | 更新正确但不会规划时能否提升？ |
| no belief + full planner | planner 是否只靠 history/更长 prompt？ |
| static belief + full planner | 增益是否来自 continuous update？ |
| continuous belief + belief-usable planner | 完整方法 |
| oracle belief + planner | belief error 的性能上界 |
| shuffled belief + planner | planner 是否真的读取 belief，还是忽略字段？ |
| frozen-at-turn-k belief | 哪个阶段的 update 有价值？ |
| overconfident/wrong belief | 不确定性与风险控制是否有效？ |
| no information gain | 主动 probe 是否有边际价值？ |
| no chooser/action lock | 提升是否只是格式与 action locking？ |

其中 shuffled/oracle/frozen 是论文说服力的核心。仅比较 direct、ASTRA prompt 和 full framework 仍无法证明 belief 的因果贡献。

## 15. 评估指标：从总收益扩展到整条因果链

### 15.1 Belief quality

- preference rank correlation、top-k accuracy；
- reservation MAE/RMSE；
- acceptance probability NLL/Brier/ECE；
- credible interval coverage；
- belief consistency/contradiction rate；
- belief recovery speed 与最后一轮误差。

### 15.2 Decision quality

- agreement、own utility、opponent utility、joint welfare；
- Nash/Pareto distance；
- regret relative to oracle-belief planner；
- belief intervention 后 action-flip rate；
- calibration error 到 decision regret 的曲线；
- compensated-concession rate：给对方高价值 issue 时，是否换回自己的高价值 issue。

### 15.3 Process 与成本

- information gain per probe、无效 probe 比例；
- turns、tokens、latency、LLM calls、simulation budget；
- parse/constraint/action-lock error；
- privacy leakage 与对手对我方 preference recovery；
- repeated seeds 的置信区间和 paired significance test。

### 15.4 Generalization

- unseen scenarios/issues；
- unseen opponent policies/personalities；
- unseen model families；
- English/Chinese/Indic language shift；
- stationary → non-stationary opponent shift。

## 16. 当前 framework 要达到该 story 仍需补齐什么

### 16.1 从灵活 JSON 改为 typed posterior

`flexibility_delta` 可以作为 observation feature 或 model-generated evidence，但不能直接当 SFT 真值。优先训练/监督：

- pairwise preference `P(i > j)`；
- reservation interval/distribution；
- hard/soft constraint labels；
- conditional acceptance `P(accept | deal, t)`；
- evidence span 与 confidence；
- posterior update 的 consistency。

GENEGO 的 pairwise + Bradley–Terry 是一个合理 warm start；ANL 的 ground-truth utility/RV 可以生成绝对可检查 labels。CaSiNo 的模糊字段只适合作辅助或弱监督。

### 16.2 Planner 输出可审计 value breakdown

每个 candidate 至少记录：

```text
candidate deal/action
self utility
E[opponent utility | belief]
P(accept | belief)
future/continuation value
information gain
disagreement risk
own-information leakage
final planner score
```

这样才能区分“planner 使用 belief”与“LLM 看过一段 belief 文本后随意选”。

### 16.3 结构化 action 与语言 generator 解耦

planner 锁定 OFFER/ACCEPT/REJECT/END 和具体 deal；generator 只能解释与表达，不能改价、改 allocation 或偷偷接受。此前 NegotiationArena/LLM-Deliberation 结果已显示格式控制本身会带来巨大收益，因此必须单独做 action-lock ablation。

### 16.4 先诊断再 training

在进行 belief calibration training 前，先用 ANL 2024/2026 生成有 ground truth 的 supervised/evaluation set。训练目标应围绕 posterior/ranking/probability，而非环境特化的自由 JSON。之后在 ANL 2025 和 AgenticPay 评估 planner 是否从更好的 calibration 获得更低 regret。

## 17. 推荐的最小可发表实验包

如果计算预算有限，建议不是在所有环境各跑一点，而是完成下面四块：

1. **ANL 2024 calibration suite**：3–5 个 opponent families，direct/static/continuous/oracle/shuffled，至少 100+ negotiation pairs；
2. **ANL 2025 planning suite**：official finalists + rule/model opponents，完整 continuation-value 消融；
3. **AgenticPay-expanded language suite**：选择 multi-dimensional + sequential/many-party 代表子集，固定 5–10 seeds；
4. **NegotiationArena repeated reproduction**：复刻 BoN-oppo，作为最直接 prior-art baseline；附加 English/Chinese language shift。

StrategicBench 若及时公开则替换或补充 AgenticPay 的一部分；否则不要让论文进度依赖尚未公开的 evaluator。

建议至少使用两个 acting model family：Qwen3-30B（本地可重复）和一个闭源强模型；所有方法严格匹配 candidate count、总 token/call budget。Opponent Simulation 会额外消耗 rollout calls，因此除 raw performance 外还要画 payoff–inference-cost frontier。

## 18. 论文叙事结构建议

### Introduction

1. Negotiation 需要推断 hidden preferences；
2. 新研究显示 LLM 可能“能猜却不会用”，表面 counterparty modeling 不产生战略收益；
3. 现有方法分别解决 inference、memory、simulation 或 agenda，但缺少可校准、可干预、跨环境的 belief-to-action chain；
4. 提出 ASTRA continuous belief + belief-usable planner；
5. 在 calibration、decision value 和 cross-environment generalization 三层验证。

### Method

1. environment-independent observation/deal schema；
2. typed continuous belief posterior；
3. exploit/probe/safe candidate generator；
4. belief-usable counterfactual planner；
5. locked action → language generator；
6. causal trace/audit protocol。

### Experiments

1. RQ1：belief 是否准确且 calibrated？ANL 2024/2026；
2. RQ2：更准的 belief 是否真的改善 action？ANL 2025 + interventions；
3. RQ3：主动信息探索何时有用？ANL 2026；
4. RQ4：同一 interface 是否跨语言市场泛化？AgenticPay-expanded；
5. RQ5：相对 Opponent Simulation/GENEGO/HAR，收益是否超出 inference budget 与格式控制？

### 贡献表述模板

> We introduce a negotiation framework that maintains an explicit, uncertainty-aware opponent belief and updates it after every interaction. Unlike narrative opponent summaries or opaque behavior simulators, the belief is directly calibrated against latent preferences and is consumed by a counterfactual planner that decomposes exploitation, agreement risk, future value, and information gain. Through oracle, shuffled, frozen, and misspecified-belief interventions, we causally evaluate when better beliefs lead to better deals across structured automated negotiation and language-mediated markets.

中文概括：

> 我们的贡献不只是让模型“写出对手画像”，而是把对手画像变成一种可校准、可干预、可计算决策价值的状态；并首次在同一实验链条中分别验证 belief accuracy、planner usage 和最终 negotiation outcome。

最后一句中的“首次”必须限定为**同一完整实验链条和本论文检索范围内**。正式投稿前仍应做更广的 systematic literature search，避免绝对优先权表述。

## 19. 最终判断

这批 2026 工作没有否定 ASTRA，反而使问题定义更清晰，但也淘汰了一个较弱的 story：

- “continuous belief + planner 在 CaSiNo/NegotiationArena 比 direct prompt 高”不足以构成强 novelty；
- “belief model 本身预测得准”也不足，因为 GENEGO/MIND 已有显式推断，且最新证据指出 LLM 能推断却不会策略性使用；
- 真正有价值的方向是 **calibrated belief → causal planner usage → decision value**，并在 ANL 的可控 ground truth 与 AgenticPay/StrategicBench 的语言复杂度之间建立桥梁。

因此推荐把后续资源优先级设为：

1. 完成 ANL 2024 calibration protocol 与 typed belief；
2. 在 ANL 2025 实现 oracle/shuffled/frozen belief planner 实验；
3. 复刻 Opponent Simulation 作为直接 baseline，并统一 inference budget；
4. 把 ASTRA 接入 AgenticPay-expanded 的 multi-dimensional/sequential 子集；
5. 再做 CaSiNo backward comparison、多语言 stress test 与 StrategicBench/HAR 扩展。

这条路线既能回答“belief 准不准”，也能回答更关键的“belief 有什么决策价值”，同时避免 framework 对任何单一 Env 过度特化。
