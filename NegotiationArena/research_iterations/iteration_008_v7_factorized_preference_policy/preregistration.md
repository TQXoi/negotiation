# Iteration 008 — V7 Factorized Preference–Policy Belief

状态：**预注册；尚未修改 V7 framework，尚未运行 V7 在线实验。**

## 1. 由冻结结果得到的问题定义

现有 posterior 只有一个 latent variable：opponent private utility parameter。
但自然语言 negotiation 的 public action 同时由两类原因生成：

1. `preference θ`：reservation value 或 resource weights；
2. `response policy φ`：理性温度、strategic delay/counter 倾向、接受偏置。

V4/V5 把 strategic counter 当作 θ 的证据，因而即使 acceptance calibration 改善，
continuous posterior 仍可能把 policy behavior 错归因为 preference。V6 只限制了错误探索，
没有修复 exploitation action。正式结果与 smoke 的共同证据为：

- continuous 不稳定优于 frozen；
- wrong/shuffled 有时不差于 correct；
- oracle preference 仍有 headroom；
- V5 calibration 改善不带来稳定 reward；
- V6 在线执行 0 个 probe，Buyer/Seller 仍弱于 V4。

## 2. 方法假设

维护 joint belief：

```text
b_t(θ, φ) ∝ b_{t-1}(θ, φ) · p(y_t | offer_t, θ, φ)
```

其中：

- `θ` 沿用可解释 preference particles；
- `φ = (rationality_scale, acceptance_bias, counter_propensity)`；
- response to our locked action 强更新 joint belief；
- opponent-initiated offer 只对 θ 做弱更新，不把一次 anchor 当成独立 utility sample；
- semantic claim 仍为 bounded weak θ evidence，不直接修改 φ；
- predictive surprise 可重置一部分 φ mass，但不重置 θ，从而允许 opponent-policy switch。

所有 φ 维度都是 response probability 的无量纲乘数/偏置，不含 price=43、WTP=63、
resource truth 或 setting 专属 reward threshold。

## 3. Belief-usable exploitation planner

对每个合法 structured action `a` 和每个 joint particle 计算：

```text
Q(a; θ,φ) = p_accept · self_utility
           + p_counter · continuation
           − failure_risk
           − uncompensated_concession
```

再计算 particle-wise best action 与 regret：

```text
R(a;θ,φ) = max_a' Q(a';θ,φ) − Q(a;θ,φ)

BUS(a) = E_b[Q(a)] − λ(b) · CVaR_0.9(R(a))
```

`λ(b)` 只由 posterior uncertainty 决定。这样 belief 不是生成一段 opponent description，
而是改变 proposal frontier 的 action-level regret。Observed ACCEPT 作为确定 action，与
proposal 的 BUS 和当前失败机会成本比较。V6 的 bounded DR-VOI 仅作为可选安全层保留，
不通过调 cap 强行制造 probe。

## 4. 因果 interventions

V7 必须支持：

1. correct joint continuous；
2. frozen joint belief；
3. wrong preference、correct policy；
4. shuffled preference、correct policy；
5. oracle preference、learned policy；
6. correct preference、wrong/shuffled policy（新增）；
7. oracle preference + oracle response-policy（仅诊断上限，如 evaluator 可定义）。

新增干预用于区分 preference inference 与 policy inference，避免把一个 intervention 的
reward 变化错误归因给另一层。

## 5. 实施和实验 gate

### Stage 0：代码级与 synthetic identifiability

- ACCEPT/COUNTER/REJECT 对 joint posterior 更新方向正确；
- 相同 θ、不同 φ 能被 repeated response 区分；
- 相同 φ、不同 θ 能被不同 offer 区分；
- policy surprise reset 不改变 θ marginal；
- exact oracle θ 下 BUS proposal 等于 particle-optimal frontier；
- wrong/shuffled θ 或 φ 能产生可记录 action flip；
- 不新增 LLM call，不新增非法/负 self-utility proposal。

### Stage 1：离线 trajectory replay

在 V5/V6 固定 trajectory 上做 one-step V7 shadow action：

- posterior summary reconstruction 必须通过；
- preference/policy belief 和 BUS 分量完整落盘；
- action flip 按 Buyer/Seller/resource 分层；
- 不以旧 trajectory 的 realized reward 给 counterfactual action 打分。

### Stage 2：新 seed smoke

- 4 settings × V5/V6/V7；
- 2 runs × 20 episodes；
- 与 iteration 007 不同的新 seed；
- V7 必须在至少两个 settings 产生非零 belief-conditioned exploitation flip；
- 至少两个 settings 不劣于 V5，且没有系统性降低 agreement/joint utility；
- preference wrong/shuffled smoke 至少在一个 setting 产生方向正确的 utility 损害，
  否则不进入 confirmatory。

### Stage 3：5×20 confirmatory

仅 Stage 2 通过后运行。主 contrasts：

- V7 vs direct、Opponent Simulation、V3.3、V5；
- continuous vs frozen；
- correct vs wrong/shuffled preference；
- correct vs wrong/shuffled policy；
- oracle preference headroom；
- calibration → action flip → realized utility mediation。

### Stage 4：opponent switch

仅当 Stage 3 的 preference/policy 因果链成立后运行。切换 preference 与仅切换 response
policy 要分开；主要指标仍为 no-switch-controlled DiD。

## 6. 不允许的捷径

1. 不用 true private parameter 或 test reward 选择 online action；
2. 不为 Buyer/Seller/resource 单独调 reward threshold；
3. 不增加 LLM chooser 或 rollout calls；
4. 不以两 run smoke 宣称显著提升；
5. 不因某个 setting 结果差而在事后删除；
6. 不将 response calibration、preference accuracy 与 utility improvement 混为一谈。

## 7. 预期 novelty story

> Natural-language negotiation requires separating what an opponent wants from
> how it strategically responds. We maintain a factorized, continuously updated
> preference–policy belief and use it through action-level posterior regret,
> then causally test each layer with targeted belief interventions.

如果 V7 不能使 correct preference/policy 稳定优于对应错误干预，最终贡献应收缩为：
belief-causality evaluation protocol + calibration/utility negative finding，而不是性能方法。
