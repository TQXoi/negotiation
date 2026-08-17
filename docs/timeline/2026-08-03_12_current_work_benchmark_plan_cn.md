# Belief Model–Planner–Generator Negotiation Agent：当前工作总结与 Benchmark 路线

**版本日期：** 2026-08-03
**分析范围：** `/work5/qixint` 下 negotiation 相关代码、运行结果与阶段材料，重点包括 `7.27/`、`Simple_Env/`、`Terms_Env/`、`ASTRA_Env/`、`AgenticPay_Env/` 及相应 run directories。
**核心参考：** `7.27/7.29_belief_planner_multiissue.pptx`、两份 2026-07-27 总结报告、`ASTRA_Env/BELIEF_USABLE_PLANNER_ALGORITHM_V01_CN.md`，以及截至 2026-08-03 的实际 `summary.json`。

---

## 1. 执行摘要

项目已经从“给 LLM 增加一次性 opponent-belief prompt”，逐步发展为一个模块化 negotiation agent：

```text
dialogue/history
  -> persistent belief update
  -> structured candidate generation
  -> belief-conditioned scoring / planning
  -> constrained final chooser
  -> language generator
  -> validator / environment feedback
  -> next-round belief update
```

当前最重要的研究转向不是继续单独提高 belief prediction accuracy，而是研究 **belief usability**：planner 如何在 belief 不完整、可能错误且随对话变化时，决定何时探索、何时利用、何时保守、何时成交，并把 belief 转化为更高质量的 offer。

截至目前，代码库已经形成四层实验梯度：

| 层级 | 环境 | 已解决的问题 | 当前定位 |
|---|---|---|---|
| Stage 0 | Simple Env / RLVR price bargaining | 单价格 offer、concession schedule、CEM/SFT 数据闭环 | 方法开发 sandbox |
| Stage 0.5 | 本地 `Terms_Env` | hidden type、utility、agreement、belief calibration 的统一诊断 | 自建诊断原型，不是官方 benchmark 复现 |
| Stage 1 | CaSiNo / ASTRA | 多 issue preference、logrolling、候选 allocation、belief-to-action | 下一阶段主机制 benchmark |
| Stage 2–3 | AgenticPay | contract terms、异构任务、多 seller/buyer/product | 泛化与真实 commerce stress test |

目前最强的新证据来自 CaSiNo/ASTRA Partner-Base：在 100 个实例上，`belief_usable_planner` 的 P1 全样本均分为 **22.27**，高于同批 `direct_prompt` 的 **18.73**；`belief_usable_llm_chooser` 为 **22.30**。但是其 belief top-1 accuracy 只有 **0.36–0.40**。因此当前可以说“结构化 candidate planning 显著有潜力”，但还不能说“更准确的 belief 导致了更高 negotiation utility”。

下一阶段的关键不是再增加一个复杂 prompt，而是进行 **因果可识别的 belief-usability 实验**：加入 oracle、frozen、shuffled、wrong-belief、planner-only 等干预，测量 belief 变化是否真的引起 action 变化，以及这种变化是否减少相对 oracle 的 decision regret。

推荐 benchmark 组合为：

1. **CaSiNo/ASTRA 作为近期主 benchmark**：动作空间可枚举、双方真实 utility 可计算，最适合验证 belief → candidate → action → utility 的因果链。
2. **官方 TERMS-Bench 作为 calibration/diagnosis benchmark**：其固定 stochastic simulator、evaluator-observable latent type 和 oracle-reference metrics 适合验证 continuous belief 与探索策略；需要注意它是双边价格谈判，不是 multi-issue contract 环境。
3. **AgenticPay 作为外部泛化 benchmark**：在 CaSiNo 机制成立后，先迁移到 matched contract subset，再进入 multi-agent/all-task；不宜现在直接把 noisy all-task average 当主结论。
4. Simple Env 和本地 `Terms_Env` 继续用于快速开发、回归测试和 ablation，不承担最终 universal claim。

---

## 2. 项目问题定义与当前研究主张

### 2.1 核心研究问题

在多轮、不完全信息 negotiation 中，agent 能否持续维护对对手偏好、reservation/acceptance、策略类型和让步模式的 belief，并让 planner 使用该 belief 选择更高 expected utility 的动作？

更具体地，项目需要回答四个问题：

1. **Belief learning：** agent 是否能从语言、offer、拒绝和让步轨迹中更新可校准的 posterior？
2. **Active exploration：** agent 是否会选择能区分 opponent hypotheses 的问题或 probe offer，而不只是被动总结历史？
3. **Belief usability：** belief 改变是否真正改变 candidate set、candidate ranking、accept/reject 和 concession？
4. **Robust planning：** belief 错误或高熵时，planner 是否能降低 posterior overconfidence 带来的损失？

### 2.2 当前最合适的论文式主张

建议将项目主张表述为：

> Opponent modeling is useful only when a planner can operationalize uncertain belief. We study belief-conditioned sequential planning that generates, explores, ranks, and robustly selects negotiation actions under a continuously updated opponent posterior.

这比“我们设计了更准确的 belief model”更符合现有代码和结果，也能把 Simple Env、CaSiNo/ASTRA 与 AgenticPay 串成一条一致路线。

### 2.3 当前尚不能声称的结论

- 不能声称 CaSiNo 的 score 提升已经由 belief accuracy 解释；当前 belief accuracy 很低，结构化 planner 可能是主要收益来源。
- 不能声称 CEM 已在完整 independent held-out test 上稳定显著优于手写 planner；现有 full-test 仍只完成约 31/128 instances，且差距较小。
- 不能声称本地 `Terms_Env` 是官方 TERMS-Bench 的完整复现。
- 不能声称 AgenticPay 上 full framework 优于 baselines；当前任务数不配对、存在失败记录，且 full framework 的若干 contract 指标较差。
- 不能将 smoke、dry-run、search-set 最优值与完整 evaluation 结果并列为同等级证据。

---

## 3. 当前系统与代码资产

### 3.1 模块化框架

代码已经覆盖以下模块：

- **Belief model**：prompt belief、typed/evidence-grounded belief、counterfactual response belief、Bayesian reservation posterior、CaSiNo issue-priority posterior。
- **Planner**：prompt planner、rule EV planner、CEM global schedule、contextual CEM、candidate enumeration、belief-scored chooser、LLM constrained chooser。
- **Generator**：将 structured action 转化为自然语言；在 CaSiNo 中对 allocation 做强制回写，避免 generator 偏离策略。
- **Validator**：格式、预算、contract feasibility/IR 等检查与修复。
- **Evaluation**：trajectory 保存、resume、summary、pairwise/case analysis、SFT 数据构建和 LoRA merge/eval。

### 3.2 Simple Env

`Simple_Env/` 是单 buyer、单 seller、单商品、单价格 issue 的 RLVR bargaining 环境。它的优势是 reward 可验证、迭代快、容易进行大量 rollout，并已经实现：

- direct / CoT / belief / full-framework 等 prompt baselines；
- typed、counterfactual、Bayesian belief variants；
- continuous rule-EV planner；
- CEM planner parameter search 与 contextual policy；
- planner/generator SFT、LoRA merge 与 evaluation；
- 小价格 policy + 大模型 verifier/generator 的低成本训练路线。

它已经证明框架具有工程可行性，但价格一维动作过于简单，planner 可能只学到 anchor/concession schedule，无法充分验证 multi-issue belief usability。

### 3.3 本地 Terms Env

`Terms_Env/` 提供 fixed/scripted counterpart、latent type、utility 与 calibration diagnostics，并实现了 direct、belief、planner-generator、full framework、typed、counterfactual、ASTRA-style、BOND-style 和 preference-estimation 等统一 baseline。

必须做一个术语澄清：本地 `Terms_Env` 是 **TERMS-Bench-style 的自建原型**；官方 TERMS-Bench 论文当前实例化为 bilateral price negotiation。因而本地环境适合内部诊断和方法 ablation，但如果用于论文主表，应明确标注其 synthetic/local status，并另做官方代码或严格 protocol 对齐。

### 3.4 ASTRA Env / CaSiNo

`ASTRA_Env/` 已经实现：

- CaSiNo 三 issue、每 issue 三单位的 allocation negotiation；
- Partner-Base fixed P2 setting；
- direct prompt、full framework、belief-usable deterministic planner、belief-usable LLM chooser；
- 对全部可行 allocation 的枚举；
- 多个 self/opponent utility ratio 的 candidate generation；
- belief score、accept probability、walk-away risk、information-gain proxy；
- belief accuracy、true-high probability、entropy、confidence gap；
- 完整 trace 和 planner SFT/RFT hooks。

当前 belief updater 仍主要是 lexical/rule-based：根据对手自述、保留资源、让步/强硬行为更新 issue logits、flexibility、stubbornness 和 walk-away risk。它是可解释的 v0.1，但还不是真正由 response likelihood 驱动的 Bayesian model。

### 3.5 AgenticPay Env

`AgenticPay_Env/` 已从最初 single28 wrapper 扩展出：

- fixed-seller single28；
- all-task runner；
- 1BMS、MB1S、MBMS 等 multi-agent wrapper；
- native/direct/CoT/rule/ASTRA/belief/planner/full-framework buyer variants；
- native/validated seller；
- contract feasibility、buyer/seller/global score 与 fixed-seller buyer metric；
- resume、trajectory 与 grouped summary。

当前主要价值是工程迁移基础，而不是已有结果。continuous belief/planner 尚未在 AgenticPay 的 contract/action schema 上完成稳定、配对、全任务验证。

---

## 4. 当前实验结果与可信度

### 4.1 Simple Env：CEM 有积极信号，但 held-out 证据仍弱

`simple_env_runs/continuous_rule_ev_cem_refine_full_test/summary.json` 当前结果：

| Variant | n | 已覆盖 test instances | Avg reward | Deal rate | Bargained ratio |
|---|---:|---:|---:|---:|---:|
| conservative | 121 | 31 | 0.3246 | 0.6777 | 0.4790 |
| current | 122 | 31 | 0.3642 | 0.7377 | 0.4937 |
| CEM | 122 | 31 | 0.3857 | 0.7377 | 0.5229 |

解释：

- CEM 相对 current 的 reward 增量约 **+0.0215**，并改善 bargained ratio，但没有改善 deal rate。
- run 配置目标是 128 instances × 4 rollouts，当前只覆盖约 31 instances，不能称为完整 full test。
- 早期 search-set 的 `0.6536` 是 promising search signal，不能替代 independent evaluation。
- v0.1 的全局参数缺少场景自适应；v0.2 contextual CEM 的方向合理，但尚缺完整验证结果。

另一组 `planner_generator_anchor_strategy_full` 当前也只覆盖 17 instances：full framework reward 0.4817，planner-generator 0.4445。该结果支持 belief component 可能有帮助，但样本未完成，不能做强结论。

### 4.2 本地 Terms Env：full framework 提高 buyer utility 和 calibration

每个 variant n=60 的统一结果中：

| Variant | Deal rate | Avg buyer utility | Buyer utility on deals | Bargained ratio on deals | Accept Brier |
|---|---:|---:|---:|---:|---:|
| direct | 0.7833 | 8.5010 | 10.8523 | 0.0613 | 0.3398 |
| planner-generator | 0.7833 | 10.1364 | 12.9400 | 0.1142 | 0.2307 |
| full framework | 0.7833 | **12.9182** | 16.4914 | **0.1429** | **0.1816** |
| counterfactual belief | 0.7500 | 12.4307 | **16.5742** | 0.1415 | 0.1998 |

解释：

- 在相同 deal rate 下，full framework 相比 direct 的 buyer utility 增加约 **+4.42**，Brier 降低约 **0.158**。
- counterfactual belief 同样提升 surplus extraction，但牺牲了一些 agreement。
- 这组结果很好地展示了“deal rate 饱和会掩盖策略差异”，但环境由本项目构造、seller 为 scripted，不应作为唯一外部有效性证据。

### 4.3 CaSiNo/ASTRA：当前最强机制结果

最新 100-instance Partner-Base 同批结果：

| Variant | Agreement | P1 score (all) | P1 score on agreements | P2 score (all) | Belief top-1 | True-high prob |
|---|---:|---:|---:|---:|---:|---:|
| direct prompt | 0.99 | 18.73 | 18.92 | 17.24 | — | — |
| full framework | 1.00 | 21.62 | 21.62 | 16.03 | 0.38 | 0.348 |
| belief-usable planner | 0.96 | **22.27** | 23.20 | 14.18 | 0.36 | 0.346 |
| belief-usable LLM chooser | 0.95 | **22.30** | **23.47** | 13.55 | 0.40 | 0.363 |

这组结果说明：

1. 结构化 framework 在 P1 utility 上相对 direct 有明显提升。
2. constrained LLM chooser 与 deterministic chooser 基本持平，说明当前瓶颈未必是 chooser 表达能力。
3. utility 的提升伴随 P2 utility 下降和 agreement 小幅下降，需要同时报告 joint welfare、Pareto efficiency 和 failure cost。
4. top-1 accuracy 只略高于三分类随机水平 1/3，且 true-high probability 约 0.35；因此高 score 很可能主要来自 candidate enumeration、self-weight schedule 或 generic exploitation，而不是准确 preference inference。
5. 现有 `t_statistic_agreement_p1_minus_p2` 衡量的是 P1 与 P2 score gap，不是“相对 direct baseline 的 paired significance”。下一版需要直接计算 paired treatment effect、confidence interval 和 effect size。

### 4.4 AgenticPay：wrapper 成熟度高于证据成熟度

当前 `single28_all_baselines/manual_qwen` summary 有 140 个成功 records 和 28 个 failures，各 variant 成功任务数为 20–27，不是同一任务全集上的严格配对比较。主要现象：

- deal rate 多数接近 1，单看成交率无法区分方法；
- `full_framework` avg buyer score 24.86，低于 direct 33.06、CoT 36.55 和 ASTRA-style 35.10；
- full framework contract IR violation rate 为 0.647，contract buyer utility 为负；
- 不同 variant 的 task coverage 不同，且 context-length/API errors 会造成 selection bias；
- 原始 buyer/global score、contract utility 与可行性指标之间存在不一致，必须先确定主 metric 和 task-specific semantics。

因此 AgenticPay 当前结果更像是 failure discovery：简单把 Simple Env 的 framework prompt 搬过去不会自然泛化；必须先做 schema-aware utility model、candidate generation 和 contract validator。

### 4.5 证据可靠性分级

| 证据 | 可靠性 | 用法 |
|---|---|---|
| CaSiNo Partner-Base 100-instance 同批对比 | 中高 | 当前主结果；仍需 paired seeds、更多 partner types、causal ablation |
| 本地 Terms all-baseline n=60 | 中 | 机制诊断和 ablation；不能冒充官方 benchmark |
| Simple Env partial full-test | 中低 | 方向性证据；需完成全 128×4 和统计检验 |
| Simple Env search/smoke/dry-run | 低 | 调试与假设生成 |
| AgenticPay unpaired single28 | 低 | 工程压力测试和 failure analysis，不用于排名 claim |

---

## 5. 当前工作的主要贡献

### 5.1 概念贡献

- 将研究焦点从 standalone opponent modeling 推进到 belief usability。
- 将 negotiation action selection 表述为在 posterior uncertainty 下的 sequential decision problem。
- 明确区分 belief、candidate planner、chooser、generator 和 validator，避免自然语言生成器同时承担推断、策略和执行。

### 5.2 算法贡献雏形

- persistent structured belief state；
- belief-conditioned candidate enumeration；
- self utility、estimated opponent utility、acceptance、walk-away 与 information-gain 的联合打分；
- deterministic 与 constrained-LLM chooser 对照；
- generator action locking；
- 用 trajectory 中的 belief/candidate/chosen action 形成 ranker/SFT/RFT 数据入口。

### 5.3 工程贡献

- 四类环境的统一模块化 wrapper；
- baseline/ablation factory、batch/resume、trajectory 与 summary；
- CEM、SFT、LoRA 与 small-policy 训练路径；
- fixed-seller corrected metrics 和多 agent topology 扩展。

---

## 6. 当前关键缺口

### 6.1 缺少 belief → action → outcome 的因果证据

现在看到的是 framework 与 direct 的 outcome 差异，但 framework 同时改变了 belief、candidate generation、scoring、schedule 和 generator constraint。没有 intervention 就无法识别哪一部分真正有效。

### 6.2 Belief accuracy 指标过粗

CaSiNo 当前 top-1 只检查最终最高优先级 issue，不能评价：

- 完整 High/Medium/Low permutation 是否正确；
- posterior calibration；
- belief 随轮次的收敛速度；
- belief 是否来自有效证据；
- action 是否对 belief 敏感；
- 错 belief 是否被 robust planner 抑制。

### 6.3 “Information gain” 目前主要是 heuristic

当前 score 中有 information-gain proxy，但尚未基于 `P(response | hypothesis, action)` 计算 expected entropy reduction，也未证明 probe action 实际减少了 posterior entropy或提高后续 utility。

### 6.4 Evaluation protocol 尚未统一

- 若各 variant 未使用完全相同 scenario、partner seed 和 sampling seed，就不能做高效 paired comparison。
- run 未完成时 summary 容易被当作 final result。
- 缺少 bootstrap CI、paired permutation/Wilcoxon、multiple-comparison correction。
- 不同环境的 raw score 不应拼成单一 composite score。

### 6.5 AgenticPay 缺少 schema-aware planner

CaSiNo allocation 可以精确枚举；AgenticPay contract terms 类型异构，必须先有 task schema、可行域、buyer utility 与 validator，才能生成和比较 candidates。否则 belief 再好也无法被 planner 正确使用。

---

## 7. Benchmark 选择原则与推荐

### 7.1 选择标准

一个适合本项目的 benchmark 应尽量满足：

1. opponent latent preference/type 对 evaluator 可见；
2. agent 不能直接看到 opponent private state；
3. action utility 与 feasibility 可程序化验证；
4. 可以固定 opponent policy 或至少固定 seeds；
5. 有足够丰富的 action space，使 belief 会改变最优 action；
6. 支持 agreement、own utility、joint/Pareto、calibration、compliance 等多维指标；
7. 有公开论文、代码、baseline 和明确 split；
8. 成本允许进行 paired ablation 和多 seed evaluation。

### 7.2 候选矩阵

| Benchmark | Belief 可诊断性 | Multi-issue | 外部认可 | 可控性 | 工程成本 | 推荐角色 |
|---|---:|---:|---:|---:|---:|---|
| Simple Env | 中 | 低 | 中 | 高 | 低 | 快速开发/回归 |
| 本地 Terms Env | 高 | 当前为自建扩展 | 低–中 | 高 | 低 | 内部诊断 |
| CaSiNo + ASTRA setting | 高 | 高 | 高 | 高 | 中 | **近期主机制 benchmark** |
| 官方 TERMS-Bench | 很高 | 低（价格） | 高 | 很高 | 中 | **belief/calibration 主诊断** |
| AgenticPay contract subset | 中–高 | 高 | 高 | 中 | 高 | **跨 schema 泛化** |
| AgenticPay universal | 中 | 高且多主体 | 高 | 低–中 | 很高 | 最终 stress test |
| Deal-or-No-Deal | 中 | bundle allocation | 高/经典 | 中 | 中 | 补充 sanity check |

### 7.3 推荐结论

近期不应在“CaSiNo 还是 AgenticPay”之间二选一，而应采用 **机制证明 + 诊断复核 + 泛化压力测试** 的层级设计：

```text
CaSiNo/ASTRA
  证明 multi-issue belief 是否改变 tradeoff 和 offer
        ↓
official TERMS-Bench
  证明 continuous belief、calibration、cue use 与 exploration
        ↓
AgenticPay matched contract subset
  证明 planner 能迁移到异构 contract schema
        ↓
AgenticPay multi-agent/all-task
  测试真实市场拓扑下的鲁棒性
```

其中 CaSiNo 是当前最值得投入的主线，因为它同时具备可枚举动作空间、真实双方 utility、标准数据和 ASTRA baseline。官方 TERMS-Bench 则提供更强的 evaluator-side diagnosis，可弥补 CaSiNo Partner-Base 仍是 LLM counterpart、方差较高的问题。

---

## 8. 下一步实验计划

### Phase 0：先冻结评估协议（1 周）

目标是让后续每个结果都可比较、可复现、可做统计检验。

1. 为所有环境建立统一 experiment manifest：commit、model、prompt/version、temperature、seed、scenario split、partner、max rounds、candidate K、费用/耗时。
2. 强制 paired evaluation：每个 variant 使用相同 scenario id、partner seed 和可控 sampling seed。
3. summary 增加 `expected_n`、`completed_n`、`completion_rate`；未完成结果自动标记 `partial`。
4. 输出 episode-level paired table，而不只输出 variant averages。
5. 预注册主指标与次指标，避免看完结果再挑指标。
6. 统计报告至少包含 mean、95% bootstrap CI、paired effect、win/tie/loss 和 failure rate。

**主指标建议：**

- own utility over all episodes（walk-away 计入）；
- agreement rate；
- Pareto efficiency / distance to Pareto frontier；
- oracle decision regret；
- belief log loss/Brier；
- protocol/IR/feasibility violation。

### Phase 1：CaSiNo belief-usability 主实验（2–3 周）

### 8.1.1 Baseline 与关键干预

至少运行以下 variants：

| Variant | Belief | Candidate planner | Chooser | 目的 |
|---|---|---|---|---|
| Direct | 无 | 无 | LLM | end-to-end baseline |
| Planner-only | 无/均匀 prior | 有 | deterministic | 测候选结构本身 |
| Belief-only | 有 | 无 | LLM | 测 belief prompt 本身 |
| Full | learned belief | 有 | deterministic | 当前方法 |
| Full + LLM chooser | learned belief | 有 | constrained LLM | chooser ablation |
| Oracle belief | true P2 values | 有 | 同一 chooser | 方法上界 |
| Frozen prior | 不更新 | 有 | 同一 chooser | continuous update 的增益 |
| Shuffled belief | 来自另一 episode | 有 | 同一 chooser | 检验 planner 是否真的依赖 belief |
| Adversarial/wrong belief | 反转 issue priority | 有 | 同一 chooser | 鲁棒性和错误成本 |

这是下一阶段最关键的一张表。若 Full 优于 Planner-only，Oracle 再明显优于 Full，且 Shuffled/Wrong 显著下降，才能较强地说明 planner 的收益确实来自可用 belief。

### 8.1.2 Belief metrics

- 完整 issue ranking accuracy / Kendall tau；
- per-issue posterior Brier 与 log loss；
- expected calibration error；
- entropy-by-round 与 true-hypothesis probability-by-round；
- evidence attribution precision：语言 evidence 与 allocation evidence 分开；
- belief update gain：相对 frozen prior 的 log-loss reduction。

### 8.1.3 Planning metrics

- selected-action self utility、estimated opponent utility 与真实 opponent utility；
- candidate-set oracle recall：真实最优或 Pareto candidate 是否进入 top-K；
- selection regret：candidate set 内最好动作与所选动作的差；
- end-to-end regret：全 action space oracle 与所选动作的差；
- belief-action sensitivity：固定 history，仅替换 posterior 时 action 改变率；
- robustness curve：随着 belief corruption 增加，utility 如何退化。

### 8.1.4 Outcome metrics

- P1 utility over all episodes；
- agreement rate 与 walk-away loss；
- joint utility、Nash product、Pareto efficiency/distance；
- issue-level logrolling rate；
- rounds/latency/token cost。

### 8.1.5 Partner distribution

不要只保留 Partner-Base。建议分层为：

- fixed Partner-Base：低方差主诊断；
- cooperative / competitive / stubborn partner；
- different LLM/model-family partner；
- held-out natural CaSiNo human dialogues，若 protocol 允许做离线 belief evaluation。

主表应在 frozen partner/paired seeds 上完成，partner-shift 表验证泛化。

### Phase 2：真正的 exploration vs exploitation（2–3 周）

当前 information gain 是 heuristic，建议升级为可计算的 response model：

```text
IG(a) = H(B_t) - E_{r ~ P(r|a,B_t)}[H(B_{t+1})]

V(a) = E[U_self(a)]
       + lambda_info * IG(a)
       + gamma * E[V(B_{t+1})]
       - lambda_risk * downside(a)
```

实施顺序：

1. CaSiNo hypothesis space 使用 6 种 High/Medium/Low issue permutations，可精确枚举。
2. 用 scripted/learned response model 估计 `P(response feature | hypothesis, action)`。
3. 让 planner 生成三类候选：exploit allocation、probe offer、direct preference question。
4. 对比：无探索、direct asking、heuristic probe、expected-IG probe。
5. 测量即时 utility cost、entropy reduction、后续 action regret 和最终 utility。

关键分析应按初始 entropy 分桶：低 entropy 时 exploit，高 entropy 且剩余轮数足够时 probe，临近截止时降低 exploration。这会直接验证 PPT 中“when to explore vs exploit”的核心主张。

### Phase 3：训练 planner/ranker，而不是先训练 generator（2–4 周）

推荐低成本路线：

1. 每个 state 枚举/采样 K 个 candidates。
2. 使用真实双方 utility、partner rollout 或 oracle solver 得到 candidate target。
3. 训练小型 ranker/MLP，输入 belief statistics、round、candidate utilities、risk、history features。
4. 与 hand-written math scorer、CEM contextual policy、constrained LLM chooser比较。
5. 用 held-out scenario、held-out partner type 和 belief corruption 做验证。
6. ranker 稳定后，再用 pairwise DPO/LoRA 训练 planner；generator 保持冻结或仅做 constraint-following SFT。

训练 target 不应只有最终成交 reward，还应包含：

- candidate oracle utility/regret；
- agreement probability；
- Pareto dominance；
- information gain；
- IR/feasibility penalty；
- downside/walk-away risk。

### Phase 4：接入官方 TERMS-Bench（并行准备，2–3 周）

官方 TERMS-Bench 的价值是 counterpart 为固定 stochastic policy，latent type 和 payoff 对 evaluator 可见，可测 oracle optimality gap、surplus extraction、cue use、belief calibration 和 compliance。建议：

1. 使用官方 repository/version 和固定 split，避免只在本地 Terms 原型上报告。
2. 将 belief state 改为 seller type/reservation/acceptance posterior。
3. 比较 frozen、Bayesian update、LLM update、hybrid update。
4. 加入 active probe offer，测 calibration 与 value of information。
5. 严格按官方 metrics 报告，并单列本项目新增的 belief/action diagnostics。

TERMS-Bench 与 CaSiNo 的功能不同：前者提供最干净的 Bayesian diagnosis，后者提供最直观的 multi-issue tradeoff。两者共同支持主 claim。

### Phase 5：AgenticPay matched transfer（3–5 周）

### 8.5.1 先修评估与 schema

1. 选取所有 variants 都能完整运行的 matched task subset。
2. 修复 context truncation、retry 与 failed-task selection bias。
3. 定义 canonical contract schema：price、continuous term、discrete term、hard constraint、outside option。
4. 对每个 task 实现可验证 buyer utility 与 feasibility；无法计算的 metric 不用于 planner supervision。
5. 主表按 task paired report，不比较不等 n 的 raw averages。

### 8.5.2 迁移顺序

```text
single buyer–single seller contract subset
  -> multi-seller with fixed seller pool
  -> multi-buyer / product selection
  -> selected all-task suite
```

### 8.5.3 AgenticPay 核心 ablation

- native/direct；
- schema-aware planner-only；
- belief-only；
- full belief-usable planner；
- oracle/private-info upper bound（只作 evaluator）；
- no-validator vs validator；
- single-opponent belief vs market/outside-option belief。

### 8.5.4 主指标

- buyer utility over all tasks；
- feasible agreement / IR violation；
- seller/joint utility 或 global score；
- task-normalized regret，而不是跨任务 raw score average；
- seller-selection regret、product-selection regret；
- latency/token cost/failure rate。

### Phase 6：最终跨 benchmark 结论

最终不要构造一个跨环境总分，而应回答三个分层问题：

1. **Mechanism：** 在 CaSiNo 中，continuous belief 是否改变 issue tradeoff 并降低 decision regret？
2. **Diagnosis：** 在 TERMS-Bench 中，方法是否改善 belief calibration、cue use、surplus 与 false agreement？
3. **Generalization：** 在 AgenticPay 中，schema-aware belief planner 是否在 matched contract 与 multi-agent task 上保持收益？

---

## 9. Go/No-Go 判断标准

建议预先设定以下 gates：

### Gate A：Belief 是否有用

满足以下至少三项才继续复杂化 belief model：

- Full 显著优于 Planner-only；
- Oracle belief 显著优于 uniform/frozen belief；
- Shuffled/wrong belief 显著降低 utility 或增加 regret；
- belief-action sensitivity 与 posterior correctness 正相关；
- calibration improvement 能预测 outcome improvement。

若不满足，优先改 candidate/action representation，而不是继续训练 belief updater。

### Gate B：Exploration 是否有价值

- probe 在高 entropy、足够 horizon 的 state 上显著降低 entropy；
- exploration 的最终 utility gain 大于即时 concession/probe cost；
- 临近 deadline 时 planner 自动减少 exploration。

若不满足，information gain 不应作为主贡献，只保留为辅助 heuristic。

### Gate C：是否进入 AgenticPay full suite

- CaSiNo 主实验通过 Gate A；
- contract subset 的 utility/feasibility evaluator 稳定；
- 所有 variants 在 matched tasks 上 completion rate 接近 100%；
- schema-aware planner 优于 direct 或 native baseline。

否则继续在 contract subset 修复，不进入 noisy all-task ranking。

---

## 10. 推荐的近期最短执行清单

按优先级排序：

1. 冻结 CaSiNo 100/500-instance paired protocol，并补齐 episode-level paired statistics。
2. 增加 `planner_only_uniform`、`oracle_belief`、`frozen_belief`、`shuffled_belief`、`wrong_belief` 五个关键 variants。
3. 增加 full ranking/Brier/log-loss、candidate oracle recall 和 decision regret。
4. 在相同 Partner-Base seeds 上重跑主表；再扩展 partner types。
5. 将 heuristic IG 改为 hypothesis-response model 下的 expected IG，并做 entropy-bin analysis。
6. 完成 Simple Env 128×4 partial run，但把它降级为回归/辅助表。
7. 获取并严格对齐官方 TERMS-Bench，实现 continuous posterior 与 probe-offer ablation。
8. 修复 AgenticPay matched-task/context failures，先跑 contract subset，不直接扩 all tasks。
9. ranker 数据优先来自 CaSiNo oracle/counterfactual candidates；暂不优先训练 generator。

---

## 11. 最终判断

当前项目已经越过“只有想法和 prompt”的阶段：框架、环境、baselines、训练入口和 trajectory diagnostics 都已形成，CaSiNo 上也出现了值得继续投入的强信号。项目现在最大的机会和最大的风险是同一个问题：**planner 的提升究竟来自真正可用的 opponent belief，还是来自结构化动作空间和更 aggressive 的通用策略？**

下一阶段应围绕这个问题做干净的 causal ablation。只要能证明：

```text
better / more informative belief
  -> predictably different candidate ranking and actions
  -> lower oracle regret
  -> higher own utility without uncontrolled agreement/feasibility loss
  -> transfer across partner types and contract schemas
```

项目就能从“一个表现不错的 negotiation pipeline”升级为一个清晰、可验证、有论文贡献力度的 **belief-usable sequential planner**。

---

## 12. 外部 benchmark 依据

- TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond Deal Rate, arXiv:2605.13909 — https://arxiv.org/abs/2605.13909
- ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning via Tool-integrated Action for Dynamic Offer Optimization, EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.821/
- AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions, arXiv:2602.06008 — https://arxiv.org/abs/2602.06008
