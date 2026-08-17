# Universal Negotiation Belief–Planner Framework V1：经验总结、方法重构与实验计划

更新时间：2026-08-14

状态：V1 环境无关核心、Simple Env adapter、AgenticPay adapter、基础 causal variants、运行脚本与离线测试已经落地；正式 Qwen3-30B 实验尚未运行。

## 0. Executive Summary

研究重点应从“在 NegotiationArena 上继续寻找更高 payoff 的 V-number”调整为：

> 建立一个可跨谈判协议复用、能维护可校准对手 belief、并由 executable planner 真正使用 belief 的自然语言 negotiation framework。

NegotiationArena 的价值是帮助我们发现机制和失败模式，而不是定义最终方法。它已经证明：

1. belief 的内容能够引起 same-state action flip，oracle belief 仍有明显 headroom；
2. wrong/shuffled belief 会影响收益，说明 planner 可能真实使用 belief；
3. calibration 改善不自动带来 utility 改善；
4. stationary setting 下 continuous update 未稳定超过 frozen/no-cross；
5. reliability gate 能限制 wrong-confident belief 的破坏；
6. first-mover opening 上，昂贵的 Opponent Simulation rollouts 仍可能明显更强。

因此，当前最应该保留的不是 NegotiationArena 的具体粒子、资源交换规则或 V8 类继承链，而是以下方法原则：

- preference belief 与 response-policy belief 分离；
- formal action evidence 高权重，语言 self-report 低权重；
- belief 影响由 evidence reliability 控制；
- planner 对可执行候选进行显式评分；
- generator 不能改变 planner 已选的结构化动作；
- 用 frozen/wrong/shuffled/oracle 和 action flip 检验 belief 是否真的有用；
- 用 change point 和 old/new mixture 处理非平稳对手。

本次据此实现 `universal_belief_planner_v1`。同一个核心不导入 Simple Env、AgenticPay 或 NegotiationArena；每个 benchmark 只提供状态转换、候选动作、自己效用和协议渲染。

## 1. 目前实验经验应该如何理解

### 1.1 NegotiationArena：机制诊断成功，通用收益结论不足

同批次 5 runs × 20 episodes 中，best-per-setting 结果为：

| Setting | 我们最好版本 | 我们 | Opponent Simulation | Delta |
|---|---:|---:|---:|---:|
| Focal Buyer | V3.3 | 13.11 | 12.76 | +0.35 |
| Focal Seller | V3.3 | 16.95 | 19.13 | -2.18 |
| Resource-first | V5 | 18.45 | 40.57 | -22.12 |
| Resource-second | V3.3 | 22.47 | 11.88 | +10.59 |

这不是 universal improvement。V8 的意义是 scientific auditability：factorized belief、reliability gate、same-state intervention、calibration 和 switch audit。它不是所有 setting 的 payoff winner。

关键经验：

- 第二行动者能观察组合报价，belief 更容易成为 decision-relevant information；
- 第一行动者缺少当前局证据，主动 probe 或 opening search 比被动 belief update 更重要；
- Opponent Simulation 通过多候选、多 future rollout 获得 opening 优势，但调用成本高，并且与特定 action simulator 绑定；
- learned belief 不稳定优于 shuffled/frozen，说明 belief representation 和 update channel 尚未足够可靠；
- 所以不能把一个环境上的 best class 直接复制到其他环境。

### 1.2 Simple Env / AmazonHistoryPrice：适合受控识别与收益验证

现有可用结果中：

| 实验 | Variant | n | Avg reward | Deal rate | Buyer bargained ratio |
|---|---|---:|---:|---:|---:|
| planner/full prompt run | planner_generator | 68 | 0.445 | 0.809 | 0.550 |
| planner/full prompt run | full_framework | 68 | 0.482 | 0.868 | 0.555 |
| CEM partial run | continuous_rule_ev_current | 122 | 0.364 | 0.738 | 0.494 |
| CEM partial run | continuous_rule_ev_cem | 122 | 0.386 | 0.738 | 0.523 |
| planner RFT ckpt150 | full_framework | 256 | 0.399 | 0.773 | 0.516 |
| 原论文 reference | trained Qwen3-30B | paper | 0.766 | 0.920 | 0.839 |

这些 run 并非全部完成、也并非严格同 seed paired，因此不能据此排名。但它们清楚暴露出：

- prompt full framework 比 direct prompt 的早期轨迹稳定；
- rule/CEM planner 的格式安全，但 payoff 仍明显低于论文训练模型；
- belief 的贡献与 anchor/concession schedule 混在一起；
- seller 的“below cost/final”语言可能不真实，不能直接当监督标签；
- Simple Env 最适合测试 reservation calibration、active probe、change point 和 decision regret。

### 1.3 AgenticPay：最需要通用 belief–planner，但旧框架问题最大

已有 single28 结果来自不同批次，不能作为最终 paired comparison。它们至少表明：

- 一批 Qwen 结果中 direct/cot/ASTRA-style 的 buyer score 高于旧 full framework；
- 旧 full framework 即使 deal rate 高，也出现大量 contract IR violation；
- 一组 28-task run 中旧 full framework deal rate 为 0.821、fixed-seller buyer score 为 23.61、contract IR violation 为 0.563；
- planner_generator 的 contract IR violation 曾达到 0.778；
- 具体轨迹出现“环境判定 agreed，但 buyer contract utility 为负”的假成功；
- multi-agent all-task 既有尝试还受到模型服务中断影响，没有形成可靠正式结果。

旧 AgenticPay framework 的根本问题不是 prompt 不够长，而是：

1. belief 每轮由 LLM 从头生成，不能称为 persistent continuous belief；
2. belief 是描述性 JSON，没有明确 likelihood 或 response evidence；
3. planner 仍主要是自由生成 target，而不是比较相同 action space；
4. multi-seller 中同一个 buyer 没有稳定的 counterparty key，belief 可能串线；
5. contract generator 能改动价格和非价格 term；
6. validator 只在生成后修补，planner 本身仍可能偏好 buyer-IR 失败的合同。

## 2. 新的方法主张

方法暂命名为：

> **Universal Reliability-Gated Belief-Usable Continual Resolver (UR-BCR)**

更精确的论文问题是：

> How can a negotiation agent maintain behaviorally grounded beliefs over heterogeneous opponents and make those beliefs causally usable for planning across price, multi-issue, and multi-party natural-language negotiations?

Novelty 不应写成“belief + planner + generator”，而应落在四个可检验点：

1. **Canonical cross-environment belief state**：同一 belief 表示支持 price、continuous/discrete issue 和多个 counterparty，而不是给每个 benchmark 写一个 belief prompt。
2. **Reliability-gated hybrid update**：formal response 更新 response policy，opponent offer 更新 preference bounds，LLM 只抽取弱语义 evidence；语言不能直接覆盖 posterior。
3. **Belief-usable executable planning**：adapter 枚举可行 action，planner 用 belief 计算 agreement probability、own utility、risk 和 bounded information value；belief 不是附在 final prompt 中的说明文字。
4. **Action-locked realization and causal audit**：语言模型只能表达已选 action，不能改价格/合同；用 frozen/wrong/shuffled/oracle、same-state action flip 和 decision regret 证明 belief 的内容是否重要。

## 3. Universal Framework 架构

### 3.1 Canonical state

所有环境先转换成：

```text
CanonicalState
  session_id
  self_id / role
  counterparty_id
  turn / max_turns
  issues[]
  exact own utility scale / outside option
  public observations[]
```

核心不读取 benchmark 的 `[BUY]`、`<contract>`、RED/BLUE 或环境 hidden truth。

### 3.2 Factorized opponent belief

对每个 counterparty (j) 维护：

\[
b_t^j = b_t(\theta_j, \phi_j, z_t)
\]

- \(\theta_j\)：reservation、issue preference、option preference；
- \(\phi_j\)：在不同 offer compatibility 下 accept/counter/quit 的 response policy；
- \(z_t\)：old/new regime mixture 与 change probability。

每个 belief 以 `(session_id, counterparty_id)` 为 key。因此 1 buyer–2 sellers 会维护两个 posterior，而不是把两段历史写进同一自然语言 summary。

### 3.3 Evidence hierarchy

证据优先级：

```text
formal accept/reject/counter
  > repeated formal offers
  > term changes in formal contracts
  > natural-language preference/finality claims
```

LLM semantic updater只输出 evidence proposal。其最大 merge gate 随 direct response evidence 增加，在证据很少时保持很低，避免一句“这是底价”变成高置信度 reservation。

### 3.4 Belief-usable planner

adapter 给出候选 (a\in A(s_t))，并计算 exact focal utility 与协议可行性。核心 planner 使用：

\[
\text{Score}(a)=
P_{\rho}(\text{agree}\mid a,b_t)U_{self}(a)
+\lambda_t IG(a;b_t)
+V_{cont}(a)
-R_{quit}(a)
\]

其中：

\[
P_{\rho}=\rho_t P_b +(1-\rho_t)P_0
\]

- \(P_b\)：factorized belief 给出的 response prediction；
- \(P_0\)：adapter 的保守 protocol-local prior；
- \(\rho_t\)：由 preference/policy evidence reliability 决定；
- \(IG\) 只在早期、belief uncertainty 高、且 probe 保留足够 own utility 时生效；
- final action 由 planner 锁定，generator 不得更改 action fields。

这一规划式跨环境保持不变；环境只改变 action enumeration 和 exact utility。

### 3.5 Simple Env adapter

adapter 负责：

- 将 buyer budget、seller offer、round history 转成 canonical state；
- 枚举有界 price frontier 与 exact seller-offer accept action；
- 计算 \(U_b=budget-price\)；
- 输出 RLVR `[BUY]/[DEAL]/[QUIT]`；
- 调用原 validator，并检查最终 action 与 selected candidate 完全一致。

### 3.6 AgenticPay adapter

adapter 负责：

- price-only 和 contract-mode 的统一 canonicalization；
- 从 buyer private preference 计算 exact buyer contract utility；
- 枚举 own-best、seller-latest 和 one-issue tradeoff/logrolling profiles；
- 所有 negative buyer utility contract 在进入 planner 前被过滤；
- multi-seller 显式传入 seller1/seller2 counterparty id；
- 输出 AgenticPay price tag 或完整 `<contract>`；
- generator 只生成不含数字的 persuasive preamble，结构化合同由 trusted renderer 附加。

## 4. 当前已经实现的代码

### 4.1 环境无关核心

```text
negotiation/framework/
  schemas.py   canonical state/belief/candidate/trace
  belief.py    persistent store, structured update, semantic gated update
  planner.py   normalized belief-conditioned candidate scorer
  engine.py    orchestration, interventions, action locking
```

### 4.2 Simple Env

```text
Simple_Env/buyer/universal_framework.py
Simple_Env/scripts/run_universal_framework_v1.sh
```

可用 variants：

- `universal_framework_v1`
- `universal_framework_v1_frozen`
- `universal_framework_v1_wrong`
- `universal_framework_v1_shuffled`
- `universal_framework_v1_oracle`

### 4.3 AgenticPay

```text
AgenticPay_Env/buyer/universal_framework.py
AgenticPay_Env/scripts/run_universal_framework_v1.sh
```

可用 variants：

- `universal_framework_v1`
- `universal_framework_v1_frozen`
- `universal_framework_v1_wrong`
- `universal_framework_v1_shuffled`

AgenticPay oracle 暂不进入正式 registry，因为原 benchmark 不应向 buyer 暴露 seller private utility。后续只在专门的 diagnostic runner 中显式注入，并在结果中标为 upper bound。

## 5. 验证状态

已通过：

- 4 个 universal framework 单元测试；
- Simple 与 AgenticPay 使用同一个 engine；
- per-counterparty belief isolation；
- AgenticPay negative buyer-utility contract 被候选层过滤；
- action lock 与 buyer IR validation；
- 原 Simple identifiable framework 的 8 个回归测试；
- 两个环境的 CLI dry-run；
- Python compile 和 Shell syntax。

当前没有可用的 `127.0.0.1:8002` 模型服务，因此没有把 mock/offline 结果写成模型 performance，也没有开始正式 Qwen3-30B rollout。

## 6. Benchmark 角色重新分配

| Benchmark | 论文中的角色 | 主要问题 | 是否主 benchmark |
|---|---|---|---|
| Simple Env / AmazonHistoryPrice | 受控 price negotiation、calibration、probe、switch | 单 issue，belief 容量有限 | 是，机制主实验 |
| AgenticPay single28 | cross-domain price/contract transfer | 原 score 与 IR 需联合审计 | 是，泛化第一层 |
| AgenticPay multi-issue/multi-party | issue tradeoff、seller selection、competition | task heterogeneous、计算较大 | 是，泛化主结果 |
| NegotiationArena | Opponent Simulation baseline 与已完成 causal audit | action space 简单且角色不对称 | 次要兼容性实验 |
| CaSiNo | 自然语言多议题外部验证 | 数据规模和 preference observability | 后续外部验证 |

论文主表不应由 NegotiationArena 单独承担。推荐主表以 Simple/AmazonHistoryPrice + AgenticPay 层级泛化为核心，NegotiationArena 放 baseline compatibility/diagnostic 表。

## 7. Baseline 设计

### 7.1 Simple Env

必须同模型、同 scenario、同 seed 对比：

1. direct prompt；
2. CoT prompt；
3. 旧 `full_framework`；
4. 旧最佳 executable variant，例如 `continuous_rule_ev_cem`；
5. 原论文 trained Qwen3-30B reference 或可运行 checkpoint；
6. universal V1。

### 7.2 AgenticPay

1. repo-native；
2. direct prompt；
3. CoT；
4. ASTRA-style legacy baseline；
5. 旧 `full_framework`；
6. universal V1。

Opponent Simulation 只在其公开 method 能合理定义 future simulator 的 setting 比较。不能为了“baseline 齐全”把 NegotiationArena 的资源模拟 prompt 硬搬到 AgenticPay contract task。

### 7.3 推理调用预算

结果必须同时报告：

- calls/turn；
- input/output tokens；
- wall-clock latency；
- candidate count；
- rollout/search count。

Universal V1 默认一轮最多包含一次 semantic evidence extraction 和一次 constrained language realization；planner 本身不调用 LLM。应增加 `structured-only` 和 deterministic realization ablation，使相同调用预算比较成立。

## 8. 正式实验阶段

### Stage 0：协议与可行性 gate

每个环境 20–50 episodes：

- parse/format error < 1%；
- action-lock violation = 0；
- budget violation = 0；
- AgenticPay buyer IR violation = 0；
- trace 中每轮有 counterparty、belief、ranked candidates、selected action。

未通过前不扩大实验。

### Stage 1：Simple main paired comparison

设置：

- 128 test instances；
- 4 rollouts/instance；
- 5 seeds；
- 6 turns；
- 同一 Qwen3-30B buyer/seller snapshot；
- round-robin variant schedule；
- paired bootstrap 95% CI。

主指标：

- normalized reward；
- deal rate / MI deal ratio；
- buyer bargained ratio；
- overshoot/format violation；
- decision regret relative to oracle reservation planner。

### Stage 2：Simple belief causal matrix

固定 adapter、candidate set 和 planner，比较：

- learned；
- frozen；
- wrong-confident；
- shuffled；
- oracle。

belief 指标：

- signed/absolute reservation ratio error；
- interval coverage；
- Brier/NLL/ECE；
- same-state action flip；
- correct-direction action flip；
- learned–oracle decision regret。

成功标准不是“action flip 越高越好”，而是正确 belief 相对 wrong/shuffled 提高 utility，且 oracle 提供合理上界。

### Stage 3：Simple active probe 与 switch

使用已有 oracle-verified identifiable profile suite：

- stationary profile；
- episode 11 reservation switch；
- response-policy-only switch；
- mixed old/new regime；
- active probe vs no probe。

报告 detection delay、false alarm、post-switch regret、probe cost 与 cumulative reward。planner 权重保持冻结，先验证 belief/update，不立即训练。

### Stage 4：AgenticPay single28 paired comparison

先按任务类型分层：

- price-only；
- contract/multi-issue；
- 商品类别。

报告：

- deal rate；
- fixed-seller buyer score；
- global/seller score；
- buyer/seller IR；
- contract quality；
- rounds、errors、latency。

不能只报告 deal rate。`agreed + negative buyer utility` 必须视为 framework failure。

### Stage 5：AgenticPay multi-party hierarchy

按难度逐级进行：

1. 1 buyer–2 sellers：测试 per-seller belief 与 seller selection；
2. 2 buyers–1 seller：测试竞争压力和 win-probability/surplus tradeoff；
3. 2 buyers–2 sellers：测试 per-edge belief 与 market matching；
4. multi-product/multi-issue；
5. multi-buyer–multi-product–multi-seller。

每层先 smoke，再 5 seeds × 20 repeats。报告 macro-average，同时保留 task-family 分层结果，避免简单任务数量主导总体平均。

### Stage 6：跨环境 generalization

这是 novelty 最关键的一张表：

1. 在 Simple train/tune planner weights；
2. 不修改 core weights，直接迁移 AgenticPay；
3. 只允许 adapter 变化；
4. 与“在 AgenticPay 重新 prompt tune”的版本比较；
5. 反向做 AgenticPay-to-Simple transfer。

如果同一个 core 在两类 action space 上都 comparable，并且少量 adapter 能完成迁移，这比单环境 SOTA 更能支持 universal framework 主张。

## 9. 训练计划

当前不建议立即 end-to-end training。顺序应为：

1. 冻结 planner 和 adapter，收集 belief calibration 数据；
2. 训练 semantic evidence extractor，而不是直接 SFT 最终 belief JSON；
3. 用 formal responses 构造 response-policy likelihood target；
4. belief 通过 calibration + downstream decision regret 双重 gate 后，再训练 planner；
5. planner 训练目标是 canonical candidate ranking，不是 Simple Env 的绝对价格 token；
6. 最后才训练 language realization，而且它不获得修改 structured action 的权限。

通用训练样本应是：

```json
{
  "canonical_state": "...",
  "opponent_belief": "...",
  "candidate_features": ["..."],
  "formal_response_labels": ["..."],
  "selected_candidate": "...",
  "exact_own_utility": 0.0,
  "environment_id": "held out during transfer evaluation"
}
```

训练/测试必须按 scenario、seller policy 和 environment family 切分，防止记住 benchmark 名称或典型 reservation。

## 10. 可证伪的论文 claims

推荐主 claim：

> A reliability-gated, factorized opponent belief can be made causally usable by an executable candidate planner through a canonical negotiation interface, enabling the same core agent to operate across single-price, multi-issue, and multi-party natural-language negotiations.

必须由实验分别支持：

- **Universal**：相同 core、只换 adapter，Simple 与 AgenticPay 都可运行且 comparable；
- **Belief-usable**：same-state intervention 会改变 candidate rank，correct belief 优于 wrong/shuffled；
- **Continual**：switch 后检测和恢复优于 frozen；
- **Safe**：action lock、budget、contract IR 不违规；
- **Efficient**：在少于 Opponent Simulation rollout budget 下保持 competitive。

不应预先声称：

- universal SOTA；
- continuous update 一定优于 frozen；
- calibration 自动提高 negotiation utility；
- AgenticPay deal rate 高就代表策略更好。

## 11. 正式命令

### Simple smoke

```bash
RUN_BASE_URL=http://127.0.0.1:8002/v1 \
RUN_PROFILE=smoke \
bash /work5/qixint/Simple_Env/scripts/run_universal_framework_v1.sh
```

### Simple main / causal

```bash
RUN_PROFILE=main bash /work5/qixint/Simple_Env/scripts/run_universal_framework_v1.sh
RUN_PROFILE=causal bash /work5/qixint/Simple_Env/scripts/run_universal_framework_v1.sh
```

### AgenticPay smoke / single28

```bash
RUN_PROFILE=smoke bash /work5/qixint/AgenticPay_Env/scripts/run_universal_framework_v1.sh
RUN_PROFILE=single28 bash /work5/qixint/AgenticPay_Env/scripts/run_universal_framework_v1.sh
```

### AgenticPay causal / market / multiissue

```bash
RUN_PROFILE=causal bash /work5/qixint/AgenticPay_Env/scripts/run_universal_framework_v1.sh
RUN_PROFILE=market bash /work5/qixint/AgenticPay_Env/scripts/run_universal_framework_v1.sh
RUN_PROFILE=multiissue bash /work5/qixint/AgenticPay_Env/scripts/run_universal_framework_v1.sh
```

## 12. 下一轮实现优先级

1. 模型服务恢复后运行两个 smoke，逐条检查 trace；
2. 增加 structured-only、no-information-gain、no-reliability-gate ablation；
3. 给 AgenticPay adapter 增加显式 competitor/market belief；
4. 为 multi-issue candidate enumeration 增加 Pareto frontier，而不是只做 one-issue swap；
5. 统一跨环境 summary，加入 call/token/latency 与 action-flip 指标；
6. Stage 0 通过后才启动 128×4×5 或 5×20 的正式矩阵；
7. 用结果决定是否进入 belief calibration training，而不是现在直接训练 final action model。

## 13. 当前结论

研究方向调整是必要的。NegotiationArena 已经完成它最有价值的任务：揭示“belief 可改变动作，但 learned continuous belief 未必提高收益”，以及 reliability/action-lock/causal interventions 的必要性。

下一阶段的核心不是为 NegotiationArena 增加更多 posterior 类型，而是验证一个相同的 canonical belief–planner core 能否：

- 在 Simple/AmazonHistoryPrice 中准确识别 reservation 与 regime；
- 在 AgenticPay contract 中利用 issue tradeoff，同时严格保护 buyer IR；
- 在 multi-seller/multi-buyer 中维护 per-counterparty belief 并进行市场级 action selection；
- 在受控 wrong/shuffled/oracle 实验中证明 belief 内容有因果价值；
- 在固定推理预算下与 direct、旧 full framework、论文模型和可适用 baseline comparable。

只有这组结果成立，才可以把工作讲成一个具有普适性和明确 novelty 的 negotiation framework，而不是某个 benchmark 上的特化 planner。
