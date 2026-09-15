# ANL 2024–2026 Automated Negotiation League 调研与 ASTRA Belief–Planner 实验建议

> 版本：2026-08-05
> 范围：只讨论 ANAC 的 **Automated Negotiation League (ANL)**，不混入 SCML、HAN 等其他联赛。
> 证据原则：比赛 setting 与排名采用官方 CFP、官方结果页或正式 competition paper；参赛方法采用官方 agent report 与公开源代码。ANL 2026 尚未公布决赛名次，因此本文只把冻结的线上榜称为“资格赛榜单”，不称冠军。

## 1. 结论摘要

ANL 很适合我们的研究，但不应把近三届混合为一个总分。它们构成三个互补的、难度逐步上升的实验层：

| 届次 | 隐藏变量与核心决策 | 最适合验证的模块 | 主要局限 |
|---|---|---|---|
| ANL 2024 | 已知对手效用函数形状，只隐藏对手 reservation value | continuous belief 的校准、更新速度、置信度 | belief 只是一维标量，不能充分代表通用偏好推断 |
| ANL 2025 | 顺序进行多个双边谈判；center 的最终效用依赖整组协议 | belief-usable planner、continuation value、跨 thread 资源分配 | 强启发式或悲观规划也能取得高分，必须做 belief/oracle 消融 |
| ANL 2026 | 对手完整 utility 与 reservation value 均未知；既要建模对手，又要隐藏自己的偏好 | 一阶 belief、二阶 belief、主动探索与信息泄露控制 | 赛事尚未结束；concealment 指标可能鼓励“难以预测”而非真实策略能力 |

因此建议把 ANL 定位为 ASTRA 的主要结构化 negotiation benchmark suite：

1. 用 2024 回答“belief 是否正确且校准”；
2. 用 2025 回答“planner 是否真的使用 belief 改善长期决策”；
3. 用 2026 回答“探索对方时，能否同时管理自身信息泄露”。

其中，**ANL 2025 是当前 framework 最核心的主实验，ANL 2024 是诊断性预实验，ANL 2026 是最有新意的扩展实验。**

## 2. 共同平台与比赛传统

近三届均使用 NegMAS 和轮流报价协议（Alternating Offers Protocol, AOP/SAO）。每轮收到报价的一方可以接受、给出 counteroffer，或终止谈判；截止前未达成协议则得到 reservation value。代码以 Python negotiator 形式提交，官方 agent 源码最终进入公开仓库，因而比纯自然语言 benchmark 更容易做确定的状态记录、ground-truth belief 评估和大规模复现。

官方代码仓库：[autoneg/anl-agents](https://github.com/autoneg/anl-agents)。本地镜像位于：

- `external_negotiation_envs/ANL2025/upstream/anl-agents/src/anl_agents/anl2024/`
- `external_negotiation_envs/ANL2025/upstream/anl-agents/src/anl_agents/anl2025/`
- `external_negotiation_envs/ANL2025/upstream/anl-agents/src/anl_agents/anl2026/`

三届最大的差异不是模型，而是信息结构和跨谈判依赖。因此跨届比较时不应直接比较 raw score，更不应宣称“2026 agent 比 2024 agent 强”；正确做法是在每届内部与官方 baseline、oracle 和消融版本比较。

## 3. ANL 2024：隐藏 reservation value 的双边谈判

### 3.1 Setting

官方任务是双边、多议题、单协议谈判。agent 知道：

- 自己的 utility function 和 reservation value；
- 对手 utility function 的排序/形状；
- 谈判截止轮数。

agent 不知道的关键量只有对手真实 reservation value。官方公开给 agent 的对手 utility 会把 reserved value 隐去。因此主要挑战是从对手报价随时间的变化估计该标量，再据此决定最低让步程度、目标报价与接受时机。截止长度在场景间变化，正式总结论文给出的范围为 10–10000 rounds。

### 3.2 指标与官方结果

官方使用两个赛道指标：

- **Individual Advantage**：协议效用相对 reservation value 的收益；
- **Nash Optimality**：协议相对 Nash 解的效率。

最终前列如下：

| Agent | Individual Advantage | Nash Optimality | 结论 |
|---|---:|---:|---|
| Shochan | **0.4051** | **0.8847** | 两项第一 |
| UOAgent | 0.4038 | 0.8668 | Individual Advantage 第二 |
| AgentRenting2024 | 0.3939 | 0.8654 | Individual Advantage 第三 |
| AntiAgent | 0.3849 | 0.8775 | Nash 指标强 |
| CARCAgent | 0.3130 | 0.8809 | 高 Nash、较低个体优势 |

完整排名可见 [ANL 2024 官方结果页](https://anac.cs.brown.edu/anl2024)，任务和结果也由 AAMAS 2025 competition paper 正式总结：[The Automated Negotiating Agents Competition 2024 Challenges and Results](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p3000.pdf)。

### 3.3 优秀方法

#### Shochan

Shochan 的关键不是单一的 curve fitting，而是“估计 + 场景分类 + 报价策略切换”：

- 将对手报价映射到已知对手效用空间；
- 假设对方采用随时间变化的 aspiration/concession curve，通过曲线拟合估计其 reservation value 和 concession exponent；
- 枚举双方理性 outcome，结合 Pareto/Nash 信息选择交易；
- 识别“让一点对对方收益很大、对自己损失很小”等场景，在等待最后时刻和提前让步之间切换；
- 对短 deadline 使用特殊保护，避免模型尚未收敛就错失协议。

值得注意的是，Shochan 源码注释也承认：显式学习 reservation value 并非在所有场景都稳定优于不学习版本。这恰好说明只报告总 utility 不足以证明 belief 有效。

#### UOAgent

- 以较高 reservation estimate 初始化并根据新报价在线下调；
- 主体采用非常强硬的 Boulware 式 aspiration；
- 临近截止时，在预测 reservation value 附近寻找对手可接受、同时兼顾社会效用和自身效用的 outcome；
- acceptance 主要由自身 aspiration 与最后时刻安全条件驱动。

它说明简单、保守的估计器配合稳健 deadline policy 就能非常接近冠军。

#### AgentRenting2024

- 把 reservation value 划分为区间，训练神经分类/预测模型；
- 从 offer history 提取均值、方差、最小/最大值、变化率、趋势、让步端点、谈判进度与估计 Nash utility 等特征；
- 仓库中附带 Keras 模型与 scaler，属于 2024 最直接可复现的 learning baseline。

#### 其他代表思路

正式总结将方法概括为三类：CARCAgent 的曲线拟合/线性回归、NayesianNice 的 Bayesian learning，以及 AgentRenting2024 的神经预测。大多数 agent 的 bidding 和 acceptance 仍以 time-dependent concession 为骨架。

### 3.4 对我们 framework 的意义

ANL 2024 非常适合作为 **belief calibration unit test**，因为隐藏量有精确真值，且对手 utility 形状已知。但它不能单独支持“通用 belief model”的结论，因为模型只需估计一个标量。

建议实验：

- `No-Belief`：固定 reservation prior；
- `Snapshot-Belief`：只在固定时刻估计一次；
- `Continuous-Belief`：每个新报价更新 posterior；
- `Continuous + Belief-Usable Planner`；
- `Oracle-RV Planner`：直接提供真值，给出可达上界；
- `Shuffled-Belief`：把另一局 belief 注入当前局，验证 planner 是否真的依赖 belief。

除比赛 utility 外，必须报告 RV MAE、区间覆盖率、NLL/CRPS、随回合变化的 calibration curve，以及 belief error 与 action regret 的相关性。

## 4. ANL 2025：顺序 multi-deal negotiation

### 4.1 Setting

2025 从单协议升级为 **one-to-many sequential multi-deal**：一个 center agent 依次与多个 edge agent 进行双边谈判。每个子谈判仍是 AOP，但 center 的最终 utility 是所有已达成协议组合的函数，通常不是各局 utility 的简单相加。早期接受什么会改变后续最优策略。

每个提交 agent 都必须能担任 center 和 edge。自己的 utility 可见，对手 utility 私有；同一 multi-deal episode 内 center 可保留先前 thread 的结果，但不能依赖未来对手主动告诉它偏好。

官方场景族包括：

- **Target Quantity**：center 希望多份协议合计接近目标数量，过量和不足都会损失；
- **Job Hunt**：多个工作选项，center 最终可能只取最佳 offer；
- **Dinners/Combination**：不同朋友的日期/偏好需要组合匹配；
- 更一般的 global、accumulating、max、linear-combination utility。

每个子谈判 deadline 在场景中变化。比赛通过随机场景和 round-robin 配对评估，官方总结记录 17 支队伍参加、12 支进入决赛。

### 4.2 评分与官方结果

CFP 定义最终成绩为 center 平均效用与 edge 平均效用的等权平均：

\[
S=\frac{\mu_{center}+\mu_{edge}}{2}.
\]

官方前三：

| Agent | Center | Edge | Final |
|---|---:|---:|---:|
| RUFL | 0.714 | **0.084** | **0.399** |
| SAC Agent | **0.733** | 0.064 | **0.399** |
| UFunAt | 0.686 | 0.078 | 0.382 |

RUFL 和 SAC 并列第一，但其实现哲学明显不同。结果来自 2025 官方结果及 competition summary：[ANAC 2025 Challenges and Results](https://arxiv.org/abs/2604.13914)。

### 4.3 优秀方法

#### RUFL：概率 continuation-value lookahead

RUFL 是最接近我们 belief-usable planner 的官方 baseline：

- 把未来子谈判展开为 extensive-form outcome tree；
- 对每个未来分支，根据预测效用通过带 temperature 的 softmax 构造发生概率；
- 用期望 terminal utility 评价当前接受/拒绝，而不是只看当前 deal；
- 因分支指数增长而截断树深或用近似叶节点估值；
- 在单个 thread 内用 opponent offer curve 拟合让步指数和最大可达 utility；
- 统计对手频繁提出的 issue value，优先构造对手较可能接受的 proposal；
- 通常拖到接近最后回合再接受，以收集信息并争取进一步让步。

详见 [RUFL 官方报告](https://anac.cs.brown.edu/files/anl/y2025/reports/20826_Team%20271_RUFL.pdf)。

#### SAC Agent：强化学习控制让步

- 用 Soft Actor-Critic 控制 time-dependent concession rate；
- 对未来采用悲观假设：规划时假设后续可能没有更多协议；
- 这种相对简单的 continuation model 获得最高 center score，并与 RUFL 总分并列。

这条结果非常重要：**总分高不自动等于 opponent belief 或显式 lookahead 有效。** planner 的价值必须通过 oracle、belief shuffle 和相同 bidding backbone 下的消融证明。

#### UFunAt：LSTM history model

- 同样采用偏悲观的未来估计；
- 用 LSTM 读取最近五条历史；
- 每条输入包含 propose/respond 标志、当前 offer utility、relative time 以及 center/edge role；
- 由历史预测/控制下一步行为。

#### 方法谱系

官方总结可归纳为：

- 悲观未来：UFunAt、SAC、默认策略；
- 概率未来：RUFL、ProbaBot、RivAgent、WAgent；
- 部分乐观未来：WAgent；
- dynamic target：EOH、CARC；
- expected-outcome search：RUFL、ProbaBot、RivAgent；
- RL：SAC；sampling：TheMemorizer、kAgent；dynamic programming：Astrat3m。

### 4.4 对我们 framework 的意义

2025 比 CaSiNo 和 NegotiationArena 更能检验 planner：当前 action 的好坏只有结合未来 thread 和最终 bundle 才能判断。建议把 belief 分成两层：

1. **Local opponent belief**：当前 edge 的 issue preference、reservation、acceptance probability、concession type；
2. **Continuation belief**：未来 edge 能提供哪些 deal，以及达到这些 deal 的概率。

planner 应比较：

\[
Q(a_t)=\mathbb E_{b_t}\left[U_{terminal}(D_{past}\cup D_{current}(a_t)\cup D_{future})\right]
-\lambda_{risk}\operatorname{Risk}-\lambda_{info}C_{probe}.
\]

其中 belief 不应该只生成文字分析，而要产生可供 planner 查询的 candidate distribution、acceptance probability 和 uncertainty。

## 5. ANL 2026：偏好隐藏、对手建模与二阶 belief

### 5.1 当前赛事状态

截至 2026-08-05，官方已关闭 agent/report 提交并公布 17 个 finalists，但 IJCAI 2026 的最终 winner 尚未公布。官方页面展示的是冻结的资格赛榜单，基于一个含 8 个场景、25,600 场谈判的 tournament。当前前三为：

1. ChangAgent：5,154；
2. AgentNexus_2.0：5,092；
3. Snake：5,020。

这只能视作现阶段强 baseline，不能写成最终冠亚季军。状态和榜单见 [ANL 2026 官方页面](https://anac.cs.brown.edu/anl)。

### 5.2 Setting

2026 回到 bilateral multi-issue、single-deal AOP，但双方完整 utility function 和 reservation value 都互相未知。每个 agent 在谈判结束时还必须提交对手 utility estimate。比赛同时奖励：

- 达成协议所获得的 normalized advantage；
- 让对手无法准确恢复自己偏好排序的 concealment。

已知固定 round deadline。对手会在不同谈判再次遇到，但规则禁止跨 negotiation 保存内存或磁盘状态，所以只允许单局在线学习。官方给出的标准/生成场景包括 Amsterdam、Camera、Car、Grocery、ISBTAcquisition、Laptop 和 NiceOrDie。

### 5.3 精确评分

对 agent A，协议 outcome 为 \(o\)，自身 utility/reservation 为 \(u_A,r_A\)：

\[
adv_A=\frac{u_A(o)-r_A}{\max(u_A)-r_A}.
\]

A 对 B 的估计为 \(\hat u_B\)。官方用估计排序与真实排序的 Kendall Tau-b 得到归一化模型准确度：

\[
\tau_A=\frac{1+d(\hat u_B,u_B)}{2}.
\]

A、B 再按各自对对手建模的相对准确度瓜分一个额外点：

\[
score_A=adv_A+\frac{\tau_A}{\tau_A+\tau_B}.
\]

如果 agent 不提交 opponent model，则其 \(\tau=0\)。这里存在一个容易混淆的方向：A 的 bonus 由 **A 对 B 的估计准确度**相对 **B 对 A 的估计准确度**决定。因此想得到更多 bonus，既要准确推断对手，也要让对手难以推断自己。精确定义见 [ANL 2026 CFP](https://anac.cs.brown.edu/files/anl/y2026/2026cfp.pdf)。

### 5.4 当前强 agent 方法（基于公开报告/源码，排名仍是资格赛）

#### ChangAgent

- 分段/bathtub-shaped target：早期强硬，中段降至由对手历史推断的 valley，晚期再根据风险回升或调整；
- ensemble opponent model：Smith/HardHeaded frequency、distribution-frequency、transition/recency signal，并带置信度加权；
- 在线识别对手类型并预测后续报价；
- candidate pool 同时考虑自身效用、对手曾报过的 anchor、Pareto-like 分数、未出现报价和受控重复；
- privacy-aware probe/decoy：在相近自身效用带内增加报价多样性，降低可读性；
- acceptance 分早中晚三段，用 next-offer、未来改善概率与 deadline risk 限制 regret。

这是当前最完整的“opponent belief + privacy-aware planner”传统算法 baseline。

#### AgentNexus_2.0

- 以 BOA 架构为骨架：time-based offering + ACNext acceptance；
- GSmithFrequencyModel 与 distribution-frequency model 结合，后者用窗口分布和卡方检验更新 issue 权重；
- 自维护 OwnOfferPrivacyModel，按 novelty、issue-value frequency balance、重复率和 utility 轨迹可读性为候选报价评分；
- 在高自身效用 band 内综合对手接受可能性、Pareto balance、novelty 与 deadline safety；
- 明确禁止跨局记忆，所有统计只在单局保存。

#### Snake

- 以 sliding-window frequency 和 issue entropy 估计对方 issue weights/value preference；
- 加入 offer transition、recency 和频率的 ensemble；
- 在线估计对手 Boulware exponent 并进行 persona 分类；
- 输出 `opponent_ufun` 供官方 concealment score 使用；
- bidding 主体非常强硬，并由 ACNext 类接受逻辑保护 deadline。

源码中目标函数目前存在一段提前 `return`，使后面的 persona-adaptive concession 分支不可达。这提示我们复现时应固定官方提交版本，并把“设计报告所述方法”和“实际执行代码”分别审计。

### 5.5 对我们 framework 的意义

ANL 2026 是最适合研究主动探索的赛制，因为 probe offer 同时具有三个效果：

1. 改变协议效用；
2. 获得关于对手偏好的观测；
3. 向对手泄露关于自身偏好的信息。

建议将当前 belief 扩为：

- \(b_t^{opp}=P(u_{opp},r_{opp},type\mid h_t)\)：一阶对手 belief；
- \(b_t^{self\rightarrow opp}=P(\hat u_{self}^{opp}\mid h_t)\)：对“对手如何理解我”的二阶 belief；
- planner 使用 utility、agreement、information gain 和 leakage 的多目标价值：

\[
Q(a)=E[adv+agreement]\;+
\beta IG(b^{opp})-\gamma Leakage(b^{self\rightarrow opp})-\lambda Risk.
\]

这比简单地提示 LLM “隐藏偏好”更可检验：每次 proposal 都能计算实际 information gain、真实 Kendall-Tau leakage 和最终 action regret。

## 6. 三届优秀方法横向总结

| 方法能力 | 2024 强方法 | 2025 强方法 | 2026 当前强方法 | ASTRA 应补充 |
|---|---|---|---|---|
| 隐藏变量 | scalar RV | 当前对手 + future deal | 完整 utility、RV、对手模型 | 统一 typed belief schema |
| 在线更新 | curve/Bayes/NN | offer curve、frequency、LSTM | frequency/entropy/recency ensemble | posterior + uncertainty + changepoint |
| 长期规划 | 单 thread deadline | outcome tree、RL、DP | 单 thread privacy/utility trade-off | belief-conditioned candidate evaluation |
| 探索 | 较弱、被动观察 | 延迟接受收集信息 | probe/decoy 与隐藏偏好 | 显式 value-of-information |
| 二阶建模 | 无 | 无 | 隐含 own-offer readability | 对手对我方 belief 的显式模型 |
| 可复现性 | 高，代码/部分模型齐 | 高，代码与报告齐 | 代码齐；最终成绩待公布 | 固定 NegMAS/version/scenario seeds |

总体上，冠军并非越来越依赖深度模型。传统的 frequency model、time concession、deadline-aware acceptance 和 outcome enumeration 一直非常强。我们的贡献不能只说“用了 LLM/belief”，而必须证明：

- belief 更准且校准；
- planner 的动作随 belief 合理变化；
- belief 改善能因果性地降低 decision regret；
- 在未见场景、不同 outcome-space 大小和非平稳对手上仍然成立。

## 7. 建议的正式实验方案

### Phase A：ANL 2024 belief calibration

- 场景：官方全场景 + 不同 deadline bucket；
- 对手：Shochan、UOAgent、AgentRenting、CARC、Nayesian、Boulware/Linear/Conceder；
- variants：direct、static belief、continuous belief、belief-planner、oracle、shuffled belief；
- 每个配对双向先手，至少 100 seeds；
- 指标：IA、Nash、agreement、RV MAE/NLL/coverage、action regret、runtime。

通过条件：continuous belief 在未见 agent 上显著优于 static；belief-planner 缩小相对 oracle 的 regret；shuffled belief 明显降低表现。

### Phase B：ANL 2025 belief-usable planner 主实验

- 场景分层：Target Quantity、Job Hunt、Dinners、general combination；
- baseline：RUFL、SAC、UFunAt、ProbaBot、RivAgent、官方 random/linear；
- roles：center 与 edge 分开报告，再按官方公式合并；
- planner variants：myopic、pessimistic、expected continuation、oracle continuation；
- belief variants：无 belief、local-only、future-only、两层 continuous belief；
- 指标：官方 score、terminal bundle utility、agreement composition、future-value calibration、每个 thread 的 accept/reject regret、runtime/memory。

必须固定同一 candidate generator 和 acceptance safety layer，只替换 belief/planner，避免生成质量成为混淆变量。

### Phase C：ANL 2026 exploration–concealment

- baseline：BOANeg、MAPNeg、ChangAgent、AgentNexus、Snake，以及去掉 privacy term 的各自版本；
- variants：utility-only、belief-only、belief+VOI、belief+VOI+leakage、oracle opponent utility；
- 指标：官方 score、advantage、agreement、我方 Kendall accuracy、对手对我方 Kendall accuracy、每回合 IG/leakage、proposal diversity；
- 泛化：官方七场景、生成场景、不同 rational fraction、不同 deadline；
- 压力测试：truthful、deceptive、nonstationary 和 random opponent。

2026 最好等最终排名公布后再冻结“top-3 official baseline”表；在此之前可用资格赛前三开发，但论文中必须标为 provisional。

## 8. 统计与复现规范

1. 预先固定 scenario、opponent、role、first mover、deadline 和 seed 的配对清单，所有 variant 共用同一清单。
2. 报告 paired bootstrap 95% CI 和成对显著性，而不只报告平均值。
3. 每局保存完整 offer/action trace、belief snapshot、candidate set、planner score decomposition、最终 ground truth。
4. LLM 组件固定 model snapshot、temperature、max tokens、prompt hash；解析失败单独计数，不静默丢弃。
5. 同时报 wall-clock、LLM calls、tokens 与 outcome evaluations，展示性能代价。
6. 对 official agents 锁定 Git commit、Python、NegMAS 和 dependency versions；先做 compatibility smoke test，再跑正式 tournament。
7. 2025 的 RUFL 在本地当前 NegMAS 版本上曾因访问“尚未开始的 future thread NMI”出现兼容问题；这属于 adapter/version 问题，修复前不能拿失败结果做性能比较。

## 9. 与当前代码库的衔接

现有目录 `external_negotiation_envs/ANL2025/` 已经具备 ANL 2025 的初步 adapter、belief state、planner 和 smoke script。下一步不应复制三套互不兼容 framework，而应抽出公共层：

```text
anl_integration/
├── core/
│   ├── observations.py       # NegMAS state -> canonical observation
│   ├── belief_schema.py      # RV / utility / acceptance / continuation posterior
│   ├── belief_updater.py     # continuous update
│   ├── candidates.py         # outcome candidates
│   ├── planner.py            # belief-conditioned evaluation
│   └── trace.py              # 完整审计日志
├── anl2024/                  # scalar-RV task adapter
├── anl2025/                  # multi-deal continuation adapter
└── anl2026/                  # opponent-ufun output + leakage model
```

优先顺序：

1. 先完成 2024 ground-truth calibration harness；
2. 修复并冻结 RUFL/SAC/UFunAt 的 2025 compatibility environment；
3. 将 2025 planner 从当前单 thread 候选选择升级为 bundle continuation evaluator；
4. 接入 2026 official scoring，复现 BOA/MAP/资格赛强 agent；
5. 再加入二阶 belief 和 VOI–leakage planner。

## 10. 风险与最终判断

- **ANL 2024 过窄**：已知对手 utility，只估计 RV；适合校准，不适合独立作为主结果。
- **ANL 2025 最匹配**：多交易耦合迫使 planner 使用未来信息，不容易退化为一次漂亮报价；应作为主 benchmark。
- **ANL 2026 最有创新空间**：直接度量建模准确度和偏好泄露，天然支持一阶/二阶 belief；但最终榜单未公布，且 concealment bonus 的构造效度需要消融讨论。
- **LLM 并非必要条件**：官方强方法多为轻量统计和搜索。我们的实验应包含非 LLM belief baseline，证明提升来自 continuous belief 与 belief-usable planning，而非模型规模。

最终建议是采用 **2024 calibration + 2025 planning + 2026 information control** 的三段式证据链。相比只在 CaSiNo 或简单 buyer–seller 环境报告 reward，这套组合能够分别验证“看得准、用得上、会探索且不泄露”，更贴合 ASTRA framework 的完整主张。

## 参考资料

- [ANL 2024 官方页面与最终结果](https://anac.cs.brown.edu/anl2024)
- [ANAC 2024 Challenges and Results, AAMAS 2025](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p3000.pdf)
- [ANAC 2025 Challenges and Results](https://arxiv.org/abs/2604.13914)
- [RUFL 2025 agent report](https://anac.cs.brown.edu/files/anl/y2025/reports/20826_Team%20271_RUFL.pdf)
- [ANL 2026 官方页面、finalists 与冻结资格赛榜](https://anac.cs.brown.edu/anl)
- [ANL 2026 CFP 与评分公式](https://anac.cs.brown.edu/files/anl/y2026/2026cfp.pdf)
- [ANL 2026 开发教程与官方 baseline](https://anac.cs.brown.edu/files/anl/y2026/template2026.pdf)
- [ANL agents 官方源代码仓库](https://github.com/autoneg/anl-agents)
