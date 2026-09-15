# 从对手建模到战略收益：通用协商 Belief–Planner Framework 架构与实验计划

> 文档类型：可编辑中文方法与实验计划
> 版本：V1.0，2026-08-07
> 对应汇报：NEGOTIATION_BELIEF_PLANNER_BENCHMARK_NOVELTY_EXPERIMENT_PLAN_EDITABLE_EN.pptx
> 工作方法名：**DC-BAP 2.0 / CARP-Neg**（working name，不把名称本身作为 novelty）
> 全称：**Decision-Calibrated Action-conditional Belief and Reciprocal Contingent Planning for Negotiation**

---

## 0. Executive decision

当前最有说服力的研究问题不是“LLM 能否猜测对手偏好”，而是：

> **什么样的对手 belief 才对当前决策有用，以及 planner 如何把这种 belief 稳定转化为自身收益，而不是更准确地迁就对方？**

近期工作已经覆盖 opponent belief、higher-order belief、Bayesian preference estimation、黑盒 opponent simulation、Best-of-N rollout 和 strategic-policy RL。因此不能再把以下单点当作核心 novelty：

- 首次维护 opponent belief；
- 首次逐轮更新 belief；
- 首次把 belief 放入 planner prompt；
- 首次用候选动作或未来 rollout；
- 首次进行高阶信念或 strategic reasoning training。

我们真正有机会形成闭环贡献的空位是：

1. **Action-conditional calibration**：belief 不只是对 preference/personality 的文字总结，而是对候选动作的 Accept/Counter/Reject/Exit 分布及其不确定性作可验证预测；
2. **Reciprocal contingent planning**：planner 不因“知道对方喜欢什么”就单向让利，而是把每次让步与自身回报绑定，并规划对方接受、还价、拒绝后的短程应对；
3. **Decision-value evaluation**：通过 frozen、shuffled、wrong-confident、oracle 等干预，因果检验 belief 是否改变动作、降低 regret、提高自身效用；
4. **Environment-independent interface**：同一 belief/planner core 运行于 AmazonHistoryPrice、CaSiNo 和 NegotiationArena，只替换动作、效用与语言 adapter。

建议论文主线为：

> **Counterparty models are useful only when they are calibrated for actions and coupled to reciprocal planning. We turn natural-language opponent evidence into an action-conditional posterior, then plan over reciprocal, multi-turn contingencies under uncertainty.**

中文表述：

> **对手建模只有在能够预测候选动作的后果、并受到互惠与自身效用约束时，才会成为战略。我们把自然语言交互转化为可校准的 action-conditional posterior，并在其上进行风险感知、互惠约束的短程 contingent planning。**

---

## 1. 为什么原有 framework 还不足以支撑这条 story

### 1.1 当前 NegotiationArena 实现

现有 DualTimescaleBeliefPlannerAgent 已有跨 episode meta-prior、局内逐 turn update、五类候选、risk/IG/future/leakage prompt、action lock 和 trace。但 posterior、candidate 与 planner score 主要由 LLM 在三次调用中自报：

~~~text
LLM belief JSON
    -> LLM candidate JSON
    -> LLM planner score JSON
    -> selected response
~~~

这会产生四个问题：

1. posterior 中的 confidence 数值不一定具有频率意义；
2. agreement probability 没有由数据或 response model 校准；
3. 同一个模型产生并评价候选，存在 self-consistency 偏差；
4. 动作改善也无法区分来自 belief、候选多样性、prompt 长度还是 action lock。

因此当前版本是 **prompt-based prototype**，不能直接称 calibrated belief planner。

### 1.2 当前 CaSiNo 实现

CaSiNo 版本能枚举 allocation、精确计算自身 utility，并维护 issue-priority belief。但接受概率与 belief score 仍是手工启发式：

\[
\operatorname{logit} P_{acc}
=-1.10+4.25\hat U_o+0.70f+0.45t-1.35\Delta U-0.95s .
\]

\[
S(a)=\hat U_sP_{acc}+0.12\hat U_o+0.10IG-0.28R_{walk}-0.10\Delta U\cdot t .
\]

主要局限：

- flexibility、stubbornness 等量没有 ground-truth target；
- 通用 entropy reduction 不等于能改变下一步动作的 information value；
- score 直接奖励较高 opponent utility，容易过度迁就；
- 没有跟踪“我付出多少让步、对方回报什么”；
- 没有显式规划 Accept/Counter/Reject 后续分支。

### 1.3 ANL2025 已给出的关键反例

本地正式结果是重要 negative result：

| Variant | Mean center utility | 相对 static/global |
|---|---:|---:|
| static belief + global planner | 0.5662 | reference |
| continuous belief + global planner | 0.4367 | -0.1295 |
| oracle partner preference | 0.6110 | +0.0448 |

continuous belief 的 preference ranking accuracy 高于 static，自身效用却显著更低。这说明：

> **belief formation 有效，不代表 belief utilization 有效。**

具体失败包括：

- continuous planner 预测偏好后给对方保留过多 utility；
- information gain 不能弥补即时让步成本；
- greedy future planner 在 complementary/target-quantity 场景破坏后续组合路径；
- preference oracle 不是 response-policy oracle，不能视为严格收益上界。

这与 2026 年 Counterparty Modeling is Not Strategy 的发现吻合：模型即使准确注意到对手偏好，也常没有把给对方的 concession 与自己高价值属性的 gain 绑定，最终容易被 opening anchor 主导。

---

## 2. Related work 对 novelty 的边界

| 工作 | 已经解决什么 | 对我们的冲击 | 我们必须多做什么 |
|---|---|---|---|
| Opponent Simulation（ICML 2026） | time-averaged opponent simulator + BoN rollout 求 repeated best response | 已有 belief formation -> simulated planning 闭环 | 新对手/单局内有效；显式 calibration；更低成本；belief interventions |
| Counterparty Modeling is Not Strategy（2026） | 证明了解对方不自动产生战略收益；单轮 trade-plan prompt 也无效 | 直接否定 preference accuracy -> reward 的朴素叙事 | 多轮 contingency 与 numerical reciprocity ledger |
| Preference Estimation via Opponent Modeling（Findings ACL 2026） | LLM 语义线索 + structured Bayesian preference estimation | “语言证据 + Bayesian belief”本身已不新 | action-conditional response belief + decision value |
| GENEGO | pairwise preference judgment + Bradley–Terry aggregation | 相对 preference belief 已被覆盖 | response uncertainty、reservation、change point、stronger planner |
| MIND | 从语言 tone 推断 willingness，再选 Push/Compromise/Yield | hidden-intent inference 已被覆盖 | 去掉人工 tone channel 与 truth fallback，测自然证据 |
| K-Level Reasoning（NAACL 2025） | 递归 higher-order beliefs | 不能声称首次高阶 belief | 作为 reasoning baseline，贡献放 calibration/usefulness |
| Game-theoretic LLM | game-theoretic workflow 提高不完全信息游戏 rationality | 结构化 game workflow 已有 | 在线未知对手、belief misspecification、局内更新 |
| Solver steering | 把语言策略接到 game solver，降低 exploitability | solver-guided language action 已有 | latent utility + natural-language evidence |
| EPO（ACL 2025） | process reward + self-play RL 训练 strategic policy | planner RL 不能单独作为 novelty | 作为 training baseline |
| HAR | 长程 agenda/tactic 层级规划 | planner hierarchy 已有 | belief-space contingency + reciprocal constraint |

### 2.1 不应再写的 claims

- the first continuous opponent model for negotiation；
- the first belief-aware negotiation planner；
- the first framework combining LLMs and Bayesian opponent modeling；
- the first approach to adapt across repeated negotiations；
- the first higher-order strategic reasoning framework；
- the first RL-trained strategy module。

### 2.2 可以 defend 的三项贡献

**Claim A — Actionable calibration**

> We represent opponent uncertainty as action-conditional response distributions and calibrate them on a shared candidate set, rather than relying on narrative summaries or unscored simulators.

**Claim B — Reciprocal contingency**

> We introduce a reciprocal contingent planner that links concessions to own gains, plans over counteroffers and rejections, and falls back to robust actions when the belief is uncertain or miscalibrated.

**Claim C — Causal belief evaluation**

> We separate belief accuracy, behavioral influence, and decision value through controlled posterior interventions and common-candidate counterfactual evaluation.

如果实验不能同时支持 A、B、C，就应缩减 claim，而不是靠综合 reward 掩盖链路失败。

---

## 3. 新 framework：DC-BAP 2.0 / CARP-Neg

### 3.1 总体数据流

~~~text
public dialogue + structured offers + private objective
                    |
                    v
      [1] Evidence Compiler (LLM + parser)
                    |
           typed evidence atoms + reliability
                    |
                    v
      [2] Action-Conditional Belief Updater
                    |
     latent posterior + P(response | candidate, history)
                    |
                    v
      [3] Feasible Candidate / Frontier Builder
                    |
 exploit / probe / safe / reciprocal / fallback
                    |
                    v
      [4] Reciprocal Contingent Planner
                    |
  action + Accept/Counter/Reject contingency
                    |
                    v
      [5] Action Lock + Language Realizer
                    |
           protocol-valid public response
~~~

核心原则：**LLM 处理开放语言，算法层处理可验证决策。**

### 3.2 Environment adapter：保证通用性

每个 benchmark 只实现以下 adapter，不修改 belief/planner core：

~~~python
class NegotiationDomainAdapter(Protocol):
    def parse_public_event(self, message) -> list[EvidenceAtom]: ...
    def feasible_actions(self, state) -> Iterable[StructuredAction]: ...
    def self_utility(self, action_or_outcome) -> float: ...
    def public_features(self, action, state) -> dict: ...
    def legal(self, action, state) -> bool: ...
    def terminal_value(self, outcome) -> float: ...
    def realize(self, locked_action, strategy) -> str: ...
~~~

环境特定内容仅包括 issue/attribute schema、合法动作、我方 utility、deadline/reservation/inventory 与 terminal evaluator。以下内容不得写进 core：

- Food/Water/Firewood 专属关键词；
- Buyer–Seller 固定 40/60 边界；
- 角色名称直接映射出的偏好；
- 某个 benchmark 的真实 opponent 标签。

### 3.3 Evidence Compiler：从文本变成有来源的证据

输出最小 evidence atom，而不是 personality story：

~~~json
{
  "source_turn": 4,
  "speaker": "opponent",
  "event": "CLAIM_PREF | OFFER | COUNTER | ACCEPT | REJECT | THREAT | COMMITMENT",
  "issue": "delivery_time",
  "direction": "prefers_lower",
  "strength": 0.62,
  "reliability": 0.35,
  "grounding": "exact text span or structured offer delta"
}
~~~

| Evidence | 初始可靠性原则 |
|---|---|
| accept/reject、正式报价变化 | 高 |
| 多轮一致的 concession pattern | 中高 |
| 明确偏好陈述 | 中等，可能 bluff |
| 理由、情绪、身份暗示 | 低 |
| 仅由角色名称推断 | 默认禁止，单列 role-prior ablation |

LLM 只负责提出 atom；parser 检查字段、引用原文和数值一致性。无法 grounding 的证据降权或拒绝。

### 3.4 Belief state：latent posterior + action-conditional response

保留可解释 latent state：

\[
\theta_t=(u_o,r_o,\pi_o,d_o,z_o,c_t),
\]

其中：

- \(u_o\)：对手 issue/attribute utility ranking 或 weight posterior；
- \(r_o\)：reservation / acceptance threshold interval；
- \(\pi_o\)：对手 response policy；
- \(d_o\)：concession dynamics；
- \(z_o\)：行为类型混合分布；
- \(c_t\)：change-point probability。

planner 的直接接口不是整段 \(\theta\)，而是对每个候选动作 \(a\) 预测：

\[
q_t(y\mid a,h_t),\qquad
y\in\{Accept,Counter_k,Reject,Exit\}.
\]

同时输出：

\[
\hat U_o(a),\quad
\operatorname{Var}[U_o(a)],\quad
\text{prediction set}_{1-\alpha}(y\mid a,h_t).
\]

这样，“对方重视 delivery”会转化成可检验问题：如果把 delivery 从 7 天改成 5 天，同时要求价格上调 3%，对方接受、还价或退出的概率分别是多少？

### 3.5 Dual-timescale update

\[
b^{meta}_{e}=p(\theta_e\mid \tau_{1:e-1}),\qquad
b_{e,t}=p(\theta_e\mid b^{meta}_{e},h_{e,1:t}).
\]

跨局历史是对方策略与我方旧 policy 共同产生的，不能把过去成交价直接当对方真实 reservation。meta-history 只能是弱 prior，并存储当时我方 action context 或 propensity。

### 3.6 Common-candidate calibration

不能只在实际选择的动作上评估 response prediction，否则存在 selection bias。每个 state 固定公共候选集合：

\[
\mathcal A_t^{common}
=\{a^{exploit},a^{probe},a^{safe},a^{reciprocal},a^{fallback}\}.
\]

所有 belief variants 在同一集合上输出 \(q_t(y\mid a,h_t)\)。反事实 label 来源：

1. 规则 opponent：直接执行或精确计算；
2. LLM opponent：复制同一 public state，对每个候选采样 \(M\) 次；
3. 人类语料：只对 observed action 计算 on-policy likelihood，不伪造反事实；
4. 多 policy logs：importance weighting 或 doubly robust estimation。

### 3.7 Candidate builder：结构 frontier，不完全依赖 LLM

**结构层**

- 枚举或搜索合法 outcomes；
- 精确计算自身 utility 与 reservation constraint；
- 由 posterior samples 估计 opponent response；
- 保留 self-utility frontier、robust Pareto candidates 与 diagnostic probes。

**语言层**

- LLM 为锁定动作生成 rationale、tone、question 或 conditional wording；
- LLM 不得改变价格、数量、issue value 或 accept/reject 动作。

| Candidate class | 结构定义 |
|---|---|
| exploit | posterior 下高自身效用且可接受概率过阈值 |
| probe | decision value of information 高，且即时成本有限 |
| safe | worst-case/CVaR 下不低于安全线 |
| reciprocal | 我方 concession 与明确 ask 绑定 |
| fallback | deadline-safe / reservation-safe |

### 3.8 Reciprocity ledger：防止准确地迁就

每 turn 记录我方让步成本：

\[
C^{self}_t=\max(0,U_s(a^{own}_{t-1})-U_s(a^{own}_t)),
\]

以及对方还价带给我方的增益：

\[
G^{opp}_t=\max(0,U_s(a^{opp}_t)-U_s(a^{opp}_{t-1})).
\]

互惠余额：

\[
L_t=\sum_{j\le t}G^{opp}_j-\kappa\sum_{j\le t}C^{self}_j.
\]

planner 施加：

1. **uncompensated concession budget**：连续单向让步不能超预算；
2. **give–ask binding**：让出对方高价值 issue 时，必须同时要求自身高价值 issue；
3. **deadline relaxation**：仅在剩余轮数很少且 agreement value 高于 fallback 时放松。

这不是 prompt 中的一句“请互惠”，而是可计算的跨 turn state，会直接改变可行动作或 score。

### 3.9 Reciprocal contingent planner

对每个候选构建深度 1–3 的 response tree：

~~~text
a_t
├── Accept  -> terminal utility
├── Counter -> update belief -> reciprocal response
├── Reject  -> update belief -> fallback/probe/exploit
└── Exit    -> reservation/failure payoff
~~~

目标：

\[
Q_t(a)=
\mathbb E_{y\sim q_t(\cdot\mid a,h_t)}[V_{t+1}(h_t,a,y)]
-\lambda_t\operatorname{CVaR}_{\alpha}(a)
+\beta_t\operatorname{DVoI}(a)
-\mu_t\operatorname{Leak}(a)
-\eta_t\operatorname{UncompConcession}(a).
\]

information term 从通用 entropy gain 改为 **decision value of information**：

\[
\operatorname{DVoI}(a)=
\mathbb E_y\left[\max_{a'}Q_{t+1}(a';b_{t+1}^{a,y})\right]
-\max_{a'}Q_{t+1}(a';b_t).
\]

只有可能改变未来动作排序的信息才有价值。若两个 hypothesis 会选同一动作，降低 entropy 没有决策价值。

### 3.10 Calibration-aware risk gate

当 validation ECE/Brier 过高、posterior 与新证据冲突、change probability 高、候选 response prediction 分歧大时，不继续激进 exploitation，而切换：

\[
a_t^*=\arg\max_a\min_{b\in\mathcal B_t}Q(a;b),
\]

或 lower-confidence-bound：

\[
a_t^*=\arg\max_a\left(\mathbb E[Q(a)]-\lambda\sqrt{\operatorname{Var}[Q(a)]}\right).
\]

这样 calibration 会真正影响策略，而不只是附加指标。

### 3.11 Action lock 与完整 trace

每轮保存：

~~~json
{
  "history_digest": {},
  "evidence_atoms": [],
  "prior": {},
  "posterior": {},
  "common_candidates": [],
  "response_predictions": [],
  "reciprocity_ledger": {},
  "contingency_values": [],
  "chosen_action": {},
  "realized_response": {},
  "posterior_after_response": {},
  "code_version": "...",
  "prompt_hash": "...",
  "model_config": {}
}
~~~

generator 只能实现 chosen action，不能修改数值。raw response、repair、fallback 和所有 API calls 均保留。

---

## 4. Training：训练什么，不训练什么

### 4.1 不监督 flexibility_delta

flexibility_delta、stubbornness_delta 可作为可视化或 weak feature，但不能作为主要 SFT label，因为它们没有唯一 ground truth，与具体候选动作的后果也没有一一对应关系。

### 4.2 Belief calibration targets

| Target | Label 来源 | Loss / metric |
|---|---|---|
| pairwise preference \(P(i\succ j)\) | environment truth / human order | Bradley–Terry NLL、rank loss |
| top issue / attribute | private truth | cross entropy、top-k |
| reservation interval | randomized private value | interval NLL、coverage、width |
| candidate response distribution | cloned-state samples | multiclass NLL、Brier、ECE |
| accept probability | accept/counter/reject logs | proper scoring |
| evidence reliability | numeric consistency、future validation | calibration loss |
| change point | controlled opponent switch | delay、false alarm |

训练样本按 dialogue prefix 构造：

~~~text
(task schema, public history up to t, previous posterior)
    -> posterior parameters + response predictions on common candidates
~~~

同一 dialogue 的不同 prefix 必须处于同一 split，避免 history leakage。

推荐 loss：

\[
\mathcal L_{belief}=
\lambda_{rank}\mathcal L_{BT}
+\lambda_{resp}\mathcal L_{NLL/Brier}
+\lambda_{int}\mathcal L_{interval}
+\lambda_{cal}\mathcal L_{calibration}
+\lambda_{temp}\mathcal L_{temporal}
+\lambda_{cp}\mathcal L_{change}.
\]

temporal loss 不要求 posterior 永远平滑；受控 switch 后应允许快速移动。

### 4.3 Planner training

planner 学共同候选集上的 action ranking/value，而不是自由文本 rationale：

~~~text
input  = self objective + history + calibrated posterior
         + candidate features + reciprocity ledger
target = candidate Q / pairwise preference / regret-to-best
~~~

数据来源：

1. 小动作空间 exact search；
2. oracle partner policy counterfactual rollout；
3. cloned LLM opponent state 多样本 rollout；
4. trajectory 的 doubly robust off-policy estimate。

\[
\mathcal L_{planner}
=\mathcal L_{pairwise-regret}
+\lambda_v\mathcal L_{value}
+\lambda_r\mathcal L_{reciprocity}
+\lambda_c\mathcal L_{constraint}.
\]

### 4.4 RL 进入条件

只有同时满足以下条件才进入 RL：

1. response NLL/Brier 优于 static 与 LLM self-report；
2. oracle response belief 在同一 planner 下显著优于 static；
3. shuffled/wrong belief 按预期降低表现；
4. reciprocal planner 不再出现 belief 越准、自身 utility 越低；
5. action validity 与 trace completeness 接近 100%。

否则 RL 很可能把 belief bypass 掉，或记住单一环境报价策略。

### 4.5 多环境训练

共享 EvidenceAtom schema、posterior updater、response classes、reciprocity ledger 与 planner heads；环境只提供 issue encoding、action search、self utility 和 protocol realizer。训练 batch 混合环境，并留一个 held-out domain 作 zero-shot 测试，否则 universal 只停留在接口层。

---

## 5. Benchmark portfolio

| Benchmark | 承担的证据 | 层级 | 注意事项 |
|---|---|---:|---|
| CaSiNo Offline + Interactive | 多议题自然语言、真实 preference、逐轮更新 | 机制主实验 | Interactive 必须称 derived controlled environment |
| NegotiationArena repeated | 与 Opponent Simulation 同 setting 直接比较 | 竞争对手主实验 | Resource Exchange 比 Buy–Sell 更关键 |
| AmazonHistoryPrice / SimpleEnv | reservation calibration、大规模便宜训练 | calibration 主实验 | 不能独立支撑 multi-issue claim |
| Multi-attribute car bargaining | 直接测试“知道对方但不会交换” | 强烈建议 | 无官方 code 时称 paper-specified reimplementation |
| AgenticPay multi-product/contract | richer contract/market 泛化 | 可选扩展 | paper track 与 expanded repo 分开 |
| ANL 2024/2025 | 精确效用、oracle 与 planner 单元测试 | appendix | 无自然语言 |
| LLM-Deliberation | role-name shortcut audit | appendix | 不承担 interaction 主结论 |

### 5.1 NegotiationArena 的正确用途

Buy–Sell 近似 SimpleEnv 简化版，不能独立证明能力。但 Opponent Simulation 直接在 repeated protocol 报告结果，因此必须回答：

> 在相同 20 episodes × 10 turns × 10 runs 下，显式 calibrated belief + contingent planner 是否优于 simulator + BoN，尤其在 first encounter、opponent switch 和 matched compute 下？

正式运行 buyer、seller、resource first、resource second 四个 cells。只跑 buyer 不足以对比原方法。

### 5.2 CaSiNo 的正确用途

CaSiNo 同时有三个可交换 issue、真实 high/medium/low priority、语言理由、偏好询问、offer 与 concession，适合验证：

~~~text
language evidence
 -> preference/response posterior
 -> reciprocal issue trade
 -> individual and joint utility
~~~

但原始 CaSiNo 是人类语料；我们构造的在线环境必须写作 CaSiNo-derived interactive evaluation。

### 5.3 Multi-attribute car 的特殊价值

Counterparty Modeling is Not Strategy 的多属性汽车交易直接测：

- informed agent 是否因知道对方而获益；
- 给对方高价值属性时是否拿回自己的高价值属性；
- opening anchor 是否仍主导 deal；
- one-turn trade-plan prompt 与 multi-turn planner 的差距。

建议先向作者索取 code/config；并行按论文公开 setting 重建，但明确标记 reimplementation。

---

## 6. Research questions

### RQ1：belief 是否校准？

- H1.1：action-conditional updater 的 response NLL、Brier、ECE 优于 static、last-action heuristic、LLM narrative；
- H1.2：opponent switch 后 change-aware belief 更快恢复；
- H1.3：pairwise preference calibration 能跨 domain transfer。

### RQ2：planner 是否真的使用 belief？

- H2.1：oracle/shuffled/wrong-confident belief 产生方向正确的 action flip；
- H2.2：固定 common candidates 时，better belief 降低 regret；
- H2.3：收益不是更多 tokens/candidates 导致。

### RQ3：使用 belief 是否形成战略互惠？

- H3.1：提高 compensated concession rate 与 strategic coupling；
- H3.2：相对 preference-only planner，提高 own utility 而不过度损失 agreement；
- H3.3：相对 one-step trade-plan，降低 anchor dependence。

### RQ4：相对 Opponent Simulation 的优势？

- first encounter 和 within-episode 阶段即有提升；
- opponent switch 后恢复更快；
- matched call/token budget 下有竞争力；
- Resource Exchange/multi-attribute 上优势大于 Buy–Sell。

### RQ5：是否跨环境？

- shared core + adapter 优于每环境 heuristic；
- held-out domain 仍有 calibration/utility 增益；
- 移除语言证据后下降，证明 evidence compiler 有贡献。

---

## 7. Baselines

### 7.1 通用 baselines

1. Direct prompting；
2. CoT / explicit strategy prompting；
3. Best-of-N + self-evaluator；
4. same candidates + no belief；
5. static snapshot belief + same planner；
6. continuous narrative belief + LLM chooser（当前 V1）；
7. full DC-BAP 2.0 / CARP-Neg。

### 7.2 Related-work baselines

| Baseline | 实现策略 |
|---|---|
| Opponent Simulation | 作者 prompt/examples + paper protocol；注明 official/reimplemented 部分 |
| GENEGO-style | pairwise LLM judgments + Bradley–Terry + tit-for-tat/threshold |
| MIND-style | willingness inference + Push/Compromise/Yield |
| K-Level | K=1/2/3 recursive prompt，matched calls |
| Game-theoretic workflow | incomplete-information Bayesian workflow |
| solver steering | 小动作空间 empirical payoff matrix + solver |
| EPO-style | strategy module SFT/RL |
| HAR-style | agenda–tactic hierarchy，无 explicit belief |
| one-step trade plan | 明确 give/ask 后直接行动 |

复现等级必须标为 official reproduction、paper-aligned reimplementation 或 idea-level baseline，不得混写。

### 7.3 Compute controls

每个主比较给两组结果：

1. method-native budget；
2. matched model、calls、tokens、candidates budget。

---

## 8. Causal ablations

### 8.1 Belief interventions

| Variant | 目的 |
|---|---|
| no belief | 候选结构本身贡献 |
| static prior | online update 贡献 |
| frozen after first observation | continuous update 贡献 |
| shuffled belief | 测 planner sensitivity |
| wrong-confident belief | 测 risk gate |
| oracle preference | 分离 preference error |
| oracle response | 分离 response modeling error |
| oracle full policy | 近似上界 |
| role-name prior only | shortcut test |
| behavior-only evidence | 结构动作证据 |
| language-only evidence | 自然语言增量 |

必须区分 oracle preference 与 oracle response policy。前者不保证知道对方如何行动。

### 8.2 Planner ablations

| Variant | 检验 |
|---|---|
| no reciprocity ledger | 是否重现过度迁就 |
| no contingent branch | 多步规划是否必要 |
| one-step give–ask | numerical planner 是否优于 verbal template |
| entropy IG instead of DVoI | decision-specific information value |
| no risk gate | 错 belief 的破坏 |
| no leakage penalty | 探索是否暴露自身信息 |
| LLM-only candidates | frontier builder 贡献 |
| no action lock | 格式与数值一致性贡献 |

所有 planner ablations 优先共享 candidate set。

---

## 9. Metrics：accuracy、causal use、strategic value

### 9.1 Belief

- preference Kendall/Spearman；
- pairwise preference accuracy/NLL；
- top-issue accuracy；
- reservation MAE 与 interval coverage/width；
- response multiclass NLL、Brier、ECE；
- calibration–sharpness curve；
- change-point delay 与 false alarm；
- evidence reliability calibration。

### 9.2 Planner causal use

- belief intervention action-flip rate；
- candidate rank correlation；
- regret to oracle candidate；
- decision value of information；
- oracle headroom closed；
- wrong-belief robustness；
- same-candidate policy value。

### 9.3 Strategic utilization

- counterparty-facing concession \(c_{t+1}\)；
- own high-value gain \(g_{t+1}\)；
- \(P(g>0\mid c>0)\)；
- compensated concession rate；
- concession-to-gain ratio；
- terminal reciprocity balance；
- opening-anchor dependence；
- uncompensated concession streak。

这是区分“注意到对方”与“战略性使用信息”的核心指标。

### 9.4 Outcome 与 cost

- agreement、timeout、walk-away；
- own/opponent/joint utility；
- own advantage、surplus share；
- NBS/Pareto distance；
- exploitability/adversarial regret；
- turns、calls、tokens、latency；
- invalid、repair、fallback、context truncation；
- raw validity 与 parser-safe success 分开。

不能只报告 joint utility。无条件迁就可能提高 agreement 或 partner utility，却损害自身目标。

---

## 10. 正式实验 setting

### 10.1 CaSiNo Offline

- 原始约 1,030 条人类对话；
- 以 dialogue 而非 utterance 划分 train/dev/test；
- prefix 为 25%、50%、75%、full-before-deal；
- 预测 preference order、top issue 与 next response；
- role name、rationale masking 作 shortcut test；
- 按 evidence type 分层报告 calibration。

### 10.2 CaSiNo Interactive

Development：

- 30 preference pairs × 2 role orders × 2 partner styles；
- direct、static、V1、V2、oracle response。

正式：

- 至少 100 preference pairs × 4 rollouts × 双角色；
- cooperative、competitive、stubborn、adaptive/switching partners；
- scenario/seed/partner trajectory 尽可能 paired；
- Qwen3-30B 为可复现主模型，另加一个闭源强模型。

### 10.3 NegotiationArena Opponent Simulation protocol

- 10 random runs；
- 每 run 20 repeated episodes；
- 每 episode 最多 10 steps；
- BoN N=5；
- buyer、seller、resource first、resource second；
- 保存 episode-index payoff curve；
- run 而不是 episode 作为统计单位。

Stress tests：

1. first encounter；
2. episode 11 opponent switch；
3. irrational/mixed opponent；
4. misleading language / truthful behavior；
5. matched 1/2/3-call budget。

### 10.4 AmazonHistoryPrice

- 128 products 全量；
- 每 product 至少 4 stochastic rollouts；
- 双角色或 focal buyer + 多 seller policies；
- price history visibility 作 factor；
- randomized reservation perturbation，避免记忆商品价格；
- 测 reservation interval、acceptance curve、utility regret。

### 10.5 Multi-attribute car reimplementation

- 随机线性 utility 与多属性 bargaining；
- none / buyer informed / seller informed / both 四个 information conditions；
- 每 condition 至少 100 trials；
- direct、one-step trade plan、V1、reciprocal contingent planner；
- strategic coupling、anchor dependence、individual/joint utility；
- 加 unknown/miscalibrated condition 测 risk gate。

### 10.6 ANL appendix

用于 exact self utility、action legality、oracle preference/response 区分、complementary utility 下的 contingent value、shuffled/wrong belief sanity check。ANL 不承担 natural-language 主结论。

---

## 11. Statistical protocol

1. variants 共享 scenario、private utility、role order、opponent assignment 和 seed；
2. 保存 episode paired key；
3. 报 mean、paired delta、95% bootstrap CI 与 effect size；
4. repeated setting 以 run 为 cluster bootstrap；
5. 预先指定 primary endpoints；
6. test 前冻结 prompt、snapshot、temperature、tokens 与 commit；
7. failure episode 纳入 all-episode reward；
8. parser repair 同时报告 raw-validity。

Primary endpoints：

- CaSiNo：own utility + response Brier + compensated concession rate；
- NegotiationArena：run-level cumulative focal payoff + switch recovery；
- AmazonHistoryPrice：regret + reservation coverage；
- multi-attribute car：informed-side utility gain + strategic coupling。

---

## 12. Development gates

### Gate 0：数据与 evaluator

- utility、agreement、failure payoff 单元测试；
- candidate legality 100%；
- raw/repair/fallback 分开；
- paired seeds 对齐。

### Gate 1：belief 有预测力

- continuous response predictor 优于 static；
- interval coverage 合格且不过宽；
- language evidence 在 behavior-only 之上有增量；
- controlled switch 可检测。

### Gate 2：planner 使用 belief

- oracle response > static；
- shuffled/wrong < correct；
- action flip 方向符合 utility；
- common-candidate regret 降低。

若 Gate 2 不通过，不扩张在线实验，应先修 planner。

### Gate 3：互惠战略成立

- 不再重现 ANL2025 的过度迁就；
- compensated concession rate 提升；
- own utility 提升且 agreement 不发生不可接受下降；
- contingent planner 优于 one-step trade plan。

### Gate 4：通用性与竞争力

- 至少两个自然语言 multi-turn domains 正向；
- held-out domain 仍有 calibration/decision-value 增益；
- matched compute 下与 Opponent Simulation 竞争；
- opponent switch / wrong belief 下更 robust。

如果只通过 Gate 1，只能写 belief calibration；通过 Gate 2–3 但跨域弱，应写 negotiation-specific planner，不能声称 universal。

---

## 13. 推荐实施顺序

### Phase A：把 V1 改为可测接口（1–2 周）

1. 抽取 NegotiationDomainAdapter；
2. 增加 EvidenceAtom 与 grounding validator；
3. 增加 common candidate set；
4. 输出 action-conditional response predictions；
5. 实现 calibration metrics 与 belief interventions；
6. 先 shadow evaluation，不改线上 policy。

交付：offline calibration table、V1 failure taxonomy、统一 trace。

### Phase B：ledger + depth-1 planner（1–2 周）

1. 精确计算 own concession/gain；
2. uncompensated-concession constraint；
3. entropy IG 改 one-step DVoI；
4. robust lower-confidence score；
5. ANL/CaSiNo oracle 单元测试。

交付：planner causal table，必须先过 Gate 2。

### Phase C：depth-2/3 contingent planning（1–2 周）

1. response classes 与 counteroffer abstraction；
2. short-horizon tree/beam search；
3. deadline terminal value；
4. 与 one-step trade-plan、OppSim rollout 比效果和成本。

交付：car + CaSiNo strategic coupling。

### Phase D：belief training（2–3 周）

1. 从 Amazon、CaSiNo、Arena 构造 prefix dataset；
2. SFT/LoRA evidence + response heads；
3. temperature scaling / conformal prediction；
4. held-out opponent/domain calibration；
5. 再训练 candidate ranker。

交付：calibration、decision regret、downstream utility 三联表。

### Phase E：正式 benchmark

1. CaSiNo main；
2. NegotiationArena exact OppSim protocol；
3. AmazonHistoryPrice full；
4. car 或 AgenticPay richer generalization；
5. ANL 与 LLM-Deliberation appendix。

---

## 14. 最小可行实验

| Cell | 规模 | Variants |
|---|---:|---|
| CaSiNo offline | test dialogues × 4 prefixes | static / V1 / action-conditional |
| CaSiNo interactive | 30 pairs × 2 styles × 2 seeds | direct / static / V1 / V2 / oracle-response / shuffled |
| Arena Resource Exchange | 3 runs × 10 episodes | direct / OppSim / V2 / frozen / switch-aware |
| ANL exact diagnostic | 100 paired states | no-ledger / ledger / oracle-response / wrong-confident |

停止条件：

- oracle-response 仍不改善 own utility：planner 结构错误；
- ledger 提高 utility 但 agreement 崩溃：调 deadline relaxation，不立即训练 belief；
- action-conditional model 不优于 static：先解决 label/calibration；
- V2 只在 CaSiNo 有效：检查 adapter 是否泄漏 issue-specific feature。

---

## 15. 预期论文表格

### Table 1：Outcome 与成本

| Method | CaSiNo own | CaSiNo agreement | Arena cumulative payoff | Amazon regret | Calls/turn |
|---|---:|---:|---:|---:|---:|
| Direct | | | | | |
| CoT | | | | | |
| BoN | | | | | |
| Opponent Simulation | | | | | |
| Static belief | | | | | |
| DC-BAP V1 | | | | | |
| DC-BAP 2.0 | | | | | |

### Table 2：Belief calibration

| Method | Pref. rank | Response NLL ↓ | Brier ↓ | ECE ↓ | Coverage | Switch delay ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Narrative belief | | | | | | |
| GENEGO-style | | | | | | |
| Static structured | | | | | | |
| Action-conditional | | | | | | |

### Table 3：Causal use 与 strategic coupling

| Variant | Action flip | Candidate regret ↓ | Compensated concession ↑ | Anchor dependence ↓ | Own utility |
|---|---:|---:|---:|---:|---:|
| Full | | | | | |
| Shuffled belief | | | | | |
| Wrong-confident | | | | | |
| Oracle preference | | | | | |
| Oracle response | | | | | |
| No ledger | | | | | |
| One-step trade plan | | | | | |

核心图应同时展示：

1. 随 turn 增加，response calibration 改善；
2. calibration 改善后 candidate regret 降低；
3. regret 降低后 own utility 提升、uncompensated concession 下降。

只有第一条成立时，不能写 framework 提升 negotiation。

---

## 16. 最终论文 story

### Motivation

LLM negotiators 能生成合理的 opponent profile。已有方法也能 preference estimation、higher-order reasoning 或 opponent rollout。但近期证据表明，counterparty modeling 往往不转化为自身战略收益；更丰富的信息甚至诱发 accommodation。

### Diagnosis

~~~text
latent belief accuracy
       X
candidate-level response prediction
       X
reciprocal multi-turn action value
~~~

现有方法常在第一个框生成 narrative，或从第一个框直接做黑盒 rollout，缺少 calibration、reciprocity 与 causal evaluation。

### Method

我们提出 action-conditional posterior，对统一候选预测 response distribution；再用 reciprocity ledger、decision value of information、risk gate 和 short-horizon contingency tree 选动作。LLM 负责语言证据与表达，结构层负责 utility、约束和 action lock。

### Evidence

- proper scoring + coverage；
- oracle/shuffled/wrong belief；
- compensated concessions + own gain；
- CaSiNo + Arena direct protocol + Amazon/car generalization；
- new opponent、switch、misleading language、matched compute。

### 一句话贡献

> **We show that opponent modeling becomes strategy only when beliefs are calibrated for candidate actions and consumed by a reciprocal contingent planner; our framework closes and causally evaluates this belief-to-decision loop across natural-language negotiation domains.**

---

## 17. 风险与诚实边界

1. action-conditional belief 接近 policy modeling；必须清楚说明与 opponent simulation 的关系；
2. LLM counterfactual samples 不是真实人类反事实，应在规则 opponent、人类 logs、不同模型间三角验证；
3. CaSiNo Interactive 是 derived environment；
4. NegotiationArena 主要用于直接 baseline，不单独证明 universality；
5. exact self utility 在现实中也可能有噪声，可在 appendix 测 noisy self utility；
6. own-utility optimization 可能降低 fairness，需同时报告 partner/joint/Pareto；
7. GENEGO 精确环境需由正文/supplement 确认，未核实前不写 CaSiNo 数值；
8. conference acceptance 状态在投稿前再次核验。

---

## 18. 与当前代码的对应

| 当前模块 | 保留 | 升级 |
|---|---|---|
| NegotiationArena repeated_agents.py posterior | dual-timescale schema、trace | LLM self-report -> evidence + response head |
| five candidates | 类别设计 | shared structured frontier |
| planner prompt | risk/IG/leakage 分解 | numerical contingency + reciprocity |
| CaSiNo MathPartnerBeliefModel | offer/text 双证据 | response prediction；不监督模糊 delta |
| CaSiNo RatioCandidatePlanner | allocation 枚举、自身 utility | common candidates、ledger、DVoI、branches |
| action lock/parser safe | 完整保留 | 同时报告 raw validity |
| ANL interventions | oracle/shuffled/frozen | oracle response、wrong-confident、exact tests |

建议新建通用目录：

~~~text
negotiation_belief_planner/
  core/
    evidence.py
    posterior.py
    response_model.py
    calibration.py
    reciprocity.py
    contingent_planner.py
    interventions.py
    trace.py
  adapters/
    amazon_history_price.py
    casino.py
    negotiation_arena.py
    anl.py
  training/
    build_prefix_data.py
    train_belief.py
    build_candidate_preferences.py
    train_planner.py
  evaluation/
    belief_metrics.py
    strategic_coupling.py
    paired_statistics.py
~~~

---

## 19. 主要资料

- Opponent Simulation：[arXiv:2602.19309](https://arxiv.org/abs/2602.19309)；ICML 2026 官方下载列表：[ICML Downloads 2026](https://icml.cc/Downloads/2026)。
- Counterparty Modeling is Not Strategy：[arXiv:2605.16575](https://arxiv.org/abs/2605.16575)。
- Preference Estimation via Opponent Modeling（Findings ACL 2026）：[arXiv:2604.15687](https://arxiv.org/abs/2604.15687)。
- CaSiNo（NAACL 2021）：[ACL Anthology](https://aclanthology.org/2021.naacl-main.254/)。
- NegotiationArena：[arXiv:2402.05863](https://arxiv.org/abs/2402.05863)。
- AmazonHistoryPrice：[arXiv:2402.15813](https://arxiv.org/abs/2402.15813)。
- EPO（ACL 2025）：[ACL Anthology](https://aclanthology.org/2025.acl-long.747/)。
- K-Level Reasoning（NAACL 2025）：[ACL Anthology](https://aclanthology.org/2025.naacl-long.370/)。
- Game-theoretic LLM：[arXiv:2411.05990](https://arxiv.org/abs/2411.05990)。
- Solver steering：[arXiv:2402.01704](https://arxiv.org/abs/2402.01704)。
- 本地 benchmark/related-work 总报告：../8.5/NEGOTIATION_BENCHMARK_RELATED_WORK_NOVELTY_EXPERIMENT_PLAN_EDITABLE_CN.md。
- 本地 OpenReview 调研：../8.5/OPENREVIEW_2026_NEGOTIATION_RELATED_WORK_AND_NOVELTY_STORY_CN.md。
- 本地自然语言 framework V2：../8.5/NATURAL_LANGUAGE_NEGOTIATION_BELIEF_FRAMEWORK_AND_EXPERIMENT_PLAN_V2_CN.md。
- ANL2025 主结果：../external_negotiation_envs/ANL_2024_2026/runs/anl2025_framework_main_n30/summary.json。

---

## 20. 最终建议

近期不应立即大规模 RL，而应按顺序完成三个判据：

1. **把 narrative profile 变成 action-conditional response posterior，并证明 calibration；**
2. **把 LLM 自报 planner score 变成有 reciprocity ledger 的 contingent value search；**
3. **用 oracle/shuffled/wrong belief 证明 planner 真能从正确 belief 获益，再训练 belief 与 planner。**

benchmark 采用“一个主机制环境 + 一个直接竞争协议 + 一个大规模校准环境 + 一个 richer construct test”：

~~~text
CaSiNo
  + NegotiationArena Opponent-Simulation protocol
  + AmazonHistoryPrice
  + multi-attribute car（或 AgenticPay multidimensional）
~~~

ANL 保留为数学正确性与 causal intervention 附录，LLM-Deliberation 保留为 shortcut audit。这个组合既不对 CaSiNo 过度特化，也不会因 NegotiationArena 过于简单而削弱主结论。
